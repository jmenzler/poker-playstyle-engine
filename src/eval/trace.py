"""Decision-trace recorder for Eval matches (closes Blocker 3 / RESEARCH Pitfall 5).

Wraps a decision engine to record per-decision
``(hand_id, street, cluster_key, hero_action, hero_chips_in)``. At match end,
aggregate by cluster_key to compute top-5 profitable + top-5 leaky cluster_keys
by bb-delta contribution.

WHY THIS EXISTS:
    RLCard's ``env.run()`` returns only the final per-hand payoff. To attribute
    that payoff back to the clusters the engine visited during the hand
    (D-NEW-30 ``top5_profitable`` / ``top5_leaky``), we must intercept decisions
    as they happen. This is the recommended mechanism per RESEARCH Pitfall 5.

ATTRIBUTION POLICY:
    Per-hand ``bb_delta`` is split across decisions by ``hero_chips_in``
    (proportional to chip commitment). If ``hero_chips_in`` is 0 for all
    decisions in a hand (e.g., hero only checked), fall back to equal split.
    This handles both:
        - "hero stacks off river" → river decision gets most of the bb_delta
        - "hero pressure flop fold" → flop bet decision gets the bb_delta

THREAT MODEL (Plan 06-05b):
    T-06-26 (Tampering, cluster_key forging): accepted. EngineRecorder records
        what the engine *actually decided* on; an attacker who could forge
        cluster_keys already owns the engine.
    T-06-27 (Info disclosure, in-memory trace): accepted. Bounded by
        ``hands * n_decisions_per_hand``; no PII; freed on function return.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import msgspec

from src._errors import NoStrategyError
from src._log import get_logger

log = get_logger("eval.trace")


class RecordedDecision(msgspec.Struct, frozen=True, kw_only=True):
    """One engine decision in a single match hand.

    Attributes:
        hand_id: 0-indexed hand counter within the match.
        street: One of ``'preflop' | 'flop' | 'turn' | 'river'``.
        cluster_key: Canonical cluster_key the engine decided in (from the
            canonicalizer's ``hard_filter``).
        hero_action: The action string the engine returned (e.g. ``'bet_50'``,
            ``'fold'``).
        hero_chips_in: Chips hero committed on THIS action; ``0.0`` for
            check/fold. Used for proportional bb_delta attribution.
    """

    hand_id: int
    street: str
    cluster_key: str
    hero_action: str
    hero_chips_in: float = 0.0


# Coarse weight table for actions when ``state['raw_obs']['my_chips']`` is not
# exposed in the RLCard state. Values are relative; the absolute scale does not
# matter because aggregation is always done as a ratio per hand.
_ACTION_WEIGHT: dict[str, float] = {
    "call": 1.0,
    # preflop sized raises — relative bb commitment (open base ≈ raise size;
    # 3bet ≈ 3-4x the open; 4bet ≈ 2.5x the 3bet).
    "open_2_2bb": 2.2,
    "open_3bb": 3.0,
    "3bet_3x": 7.5,
    "3bet_4x": 10.0,
    "4bet_2_5x": 22.0,
    "bet_25": 0.25,
    "bet_33": 0.33,
    "bet_50": 0.5,
    "bet_75": 0.75,
    "bet_100": 1.0,
    "bet_150": 1.5,
    "bet_overbet": 2.0,
    "overbet": 2.0,
    "raise_min": 1.0,
    "raise_2_5x": 2.5,
    "raise_3x": 3.0,
    "raise_pot": 1.0,
    "allin": 5.0,
}


class DecisionTrace:
    """In-memory trace of every decision the engine made during a match."""

    def __init__(self) -> None:
        self._decisions: list[RecordedDecision] = []

    def append(self, decision: RecordedDecision) -> None:
        self._decisions.append(decision)

    def __iter__(self) -> Iterator[RecordedDecision]:
        return iter(self._decisions)

    def __len__(self) -> int:
        return len(self._decisions)

    def aggregate_by_cluster(self, payoffs_by_hand: dict[int, float]) -> dict[str, dict]:
        """Group decisions by ``cluster_key``; sum apportioned bb_delta.

        For each hand, distribute payoff across that hand's decisions weighted
        by ``hero_chips_in`` (or equal-split if zero-commitment). Each
        decision's contribution is added to its ``cluster_key`` bucket.

        Args:
            payoffs_by_hand: ``{hand_id: bb_delta}`` map from ``run_match``.
                Missing hands default to ``0.0``.

        Returns:
            ``{cluster_key: {'n_decisions': int, 'bb_total': float}}``.
        """
        # Group decisions by hand_id for proportional split.
        by_hand: dict[int, list[RecordedDecision]] = defaultdict(list)
        for d in self._decisions:
            by_hand[d.hand_id].append(d)

        cluster_totals: dict[str, dict] = defaultdict(lambda: {"n_decisions": 0, "bb_total": 0.0})
        for hand_id, decisions in by_hand.items():
            bb_delta = payoffs_by_hand.get(hand_id, 0.0)
            total_chips = sum(d.hero_chips_in for d in decisions)
            n = len(decisions)
            equal_split = 1.0 / n if n > 0 else 0.0
            for d in decisions:
                weight = d.hero_chips_in / total_chips if total_chips > 0 else equal_split
                cluster_totals[d.cluster_key]["n_decisions"] += 1
                cluster_totals[d.cluster_key]["bb_total"] += bb_delta * weight
        return dict(cluster_totals)

    def top5_profitable(self, payoffs_by_hand: dict[int, float]) -> list[dict]:
        """Top-5 cluster_keys by *positive* bb_total contribution, descending."""
        agg = self.aggregate_by_cluster(payoffs_by_hand)
        sorted_pos = sorted(agg.items(), key=lambda kv: -kv[1]["bb_total"])
        return [
            {
                "cluster_key": ck,
                "n_decisions": v["n_decisions"],
                "bb_total": v["bb_total"],
            }
            for ck, v in sorted_pos[:5]
            if v["bb_total"] > 0
        ]

    def top5_leaky(self, payoffs_by_hand: dict[int, float]) -> list[dict]:
        """Top-5 cluster_keys by *negative* bb_total contribution, most-negative first."""
        agg = self.aggregate_by_cluster(payoffs_by_hand)
        sorted_neg = sorted(agg.items(), key=lambda kv: kv[1]["bb_total"])
        return [
            {
                "cluster_key": ck,
                "n_decisions": v["n_decisions"],
                "bb_total": v["bb_total"],
            }
            for ck, v in sorted_neg[:5]
            if v["bb_total"] < 0
        ]


class EngineRecorder:
    """Wraps a decision engine; records every ``decide()`` call into a ``DecisionTrace``.

    Behaves identically to the wrapped engine's ``decide()`` from the caller's
    view — returns the bare action string for the production engine, the stub's
    return value otherwise. Adds bookkeeping to track current hand_id + current
    street and extract cluster_key + chip commitment per decision.

    Two engine shapes are supported:
        - Production ``KNNDecisionEngine``: exposes ``decide_with_encoding`` and
          its ``decide`` returns a bare action string. cluster_key is derived
          from ``enc.hard_filter`` (the only place it exists) via the canonical
          sorted ``k=v`` join — matching ``sim.harness._cluster_key_from_filter``.
        - Test stubs: define only ``decide``, returning a 4- or 5-tuple. The
          5-tuple's trailing element is the cluster_key; otherwise it is read off
          the game_state (dict or attribute). ``unknown_cluster_count`` tracks
          how often resolution falls through to ``'unknown'``.

    Usage in run_match::

        trace = DecisionTrace()
        recorder = EngineRecorder(engine, trace)
        for hand_i in range(hands):
            recorder.start_hand(hand_i)
            env.set_agents([_engine_agent(recorder), baseline_agent])
            trajectories, payoffs = env.run(...)
            ...
        top5_leaky = trace.top5_leaky(payoffs_by_hand)
    """

    def __init__(self, engine: Any, trace: DecisionTrace) -> None:
        self._engine = engine
        self._trace = trace
        self._hand_id = 0
        self._street = "preflop"
        self.unknown_cluster_count = 0
        self.nostrategy_fallbacks = 0

    def start_hand(self, hand_id: int) -> None:
        """Begin a new hand; resets street tracking to preflop."""
        self._hand_id = hand_id
        self._street = "preflop"

    def note_street(self, street: str) -> None:
        """Caller (the RLCard adapter) updates this when the street changes."""
        self._street = street

    def decide(self, game_state: Any) -> Any:
        """Pass-through to wrapped engine; record decision in trace.

        Production ``KNNDecisionEngine`` exposes ``decide_with_encoding`` and
        returns a bare action string from ``decide`` — the real cluster_key lives
        only in ``enc.hard_filter``. We prefer that path so per-cluster
        attribution works against the real engine; test stubs that define only
        ``decide`` fall through to the legacy result/game_state inspection.
        """
        if hasattr(self._engine, "decide_with_encoding"):
            try:
                action, _flag, _dist, enc = self._engine.decide_with_encoding(game_state)
                cluster_key = self._cluster_key_from_enc(enc)
            except NoStrategyError as e:
                # Completely sparse partition (zero kNN neighbors — e.g. an
                # uncovered multiway spot the HU-only solver can't fill). Fall back
                # legally: fold when facing a bet, else check; never fold for free.
                # Counted + logged so the match reports how much it leaned on the
                # fallback rather than silently scoring a degraded engine.
                facing = float(getattr(game_state, "hero_facing_bet_bb", 0) or 0)
                action = "fold" if facing > 0 else "check"
                self.nostrategy_fallbacks += 1
                try:
                    cluster_key = self._cluster_key_from_enc(self._engine._canon.encode(game_state))
                except Exception:
                    cluster_key = "nostrategy"
                log.warning(
                    "eval.trace.nostrategy_fallback",
                    hand_id=self._hand_id,
                    street=self._street,
                    action=action,
                    error=str(e),
                )
            result: Any = action
        else:
            result = self._engine.decide(game_state)
            cluster_key = self._extract_cluster_key(game_state, result)
        action_str = result[0] if isinstance(result, tuple) else str(result)
        chips_in = self._extract_chips_in(game_state, action_str)
        self._trace.append(
            RecordedDecision(
                hand_id=self._hand_id,
                street=self._street,
                cluster_key=cluster_key,
                hero_action=action_str,
                hero_chips_in=chips_in,
            )
        )
        return result

    def _cluster_key_from_enc(self, enc: Any) -> str:
        """Canonical cluster_key from ``enc.hard_filter`` (sorted ``k=v`` join).

        Identical format to ``src.sim.harness._cluster_key_from_filter`` so eval
        attribution shares the HH-ingest / sim-harness key space. Empty or absent
        hard_filter is counted + logged (fail loud) rather than silently bucketed.
        """
        hard_filter = getattr(enc, "hard_filter", None)
        if isinstance(hard_filter, dict) and hard_filter:
            return "|".join(f"{k}={v}" for k, v in sorted(hard_filter.items()))
        self.unknown_cluster_count += 1
        log.warning("eval.trace.unknown_cluster_key", hand_id=self._hand_id, street=self._street)
        return "unknown"

    def _extract_cluster_key(self, game_state: Any, result: Any) -> str:
        """Legacy path: surface cluster_key from a test-stub result/game_state.

        Priority:
            1. If ``result`` is a tuple of length >= 5, use ``result[4]`` — how
               stub engines surface the cluster_key for tests.
            2. Else read ``getattr(game_state, 'cluster_key', None)``.
            3. Else if ``game_state`` is a dict, read ``state['cluster_key']``.
            4. Fall back to ``'unknown'`` (counted + logged).
        """
        if isinstance(result, tuple) and len(result) >= 5:
            return str(result[4])
        ck = getattr(game_state, "cluster_key", None)
        if ck is not None:
            return str(ck)
        if isinstance(game_state, dict) and "cluster_key" in game_state:
            return str(game_state["cluster_key"])
        self.unknown_cluster_count += 1
        log.warning("eval.trace.unknown_cluster_key", hand_id=self._hand_id, street=self._street)
        return "unknown"

    @staticmethod
    def _extract_chips_in(game_state: Any, action_str: str) -> float:
        """Approximate chips hero committed on this action.

        - Check / fold: returns ``0.0`` (no chip commitment).
        - If ``game_state['raw_obs']['my_chips']`` is exposed, returns that
          float directly (exact RLCard chip commitment for this street so far).
        - Else returns a coarse weight from ``_ACTION_WEIGHT`` (relative
          scale; absolute value doesn't matter because aggregation is a
          per-hand ratio).
        """
        if action_str in ("fold", "check"):
            return 0.0
        if isinstance(game_state, dict):
            raw = game_state.get("raw_obs", {})
            if isinstance(raw, dict) and "my_chips" in raw:
                return float(raw["my_chips"])
        return _ACTION_WEIGHT.get(action_str, 1.0)
