"""D-NEW-30 Eval match runner — engine vs single opponent over N hands.

RLCard 'no-limit-holdem' 2-player (HU only per CONTEXT). Deterministic via seed.

CLOSES THREE BLOCKERS (plan-checker iteration 1):

1. **Blocker 2 (chip-to-BB conversion):** ``payoffs[0] / 2.0`` is inline at the
   per-hand step — chips_for_each=100, BB=2 → 1 chip = 0.5 BB. There is NO
   chip-to-bb helper function (was dead+broken in the original draft).

2. **Blocker 3 (cluster attribution):** ``EngineRecorder`` wraps the engine and
   records every ``(hand_id, street, cluster_key, hero_action, hero_chips_in)``
   decision into a ``DecisionTrace``. At match end the trace aggregates by
   ``cluster_key`` with bb_delta apportioned proportional to chip commitment;
   ``top5_profitable`` + ``top5_leaky`` are derived from this real trace —
   they are NOT empty stubs.

3. **Blocker 4 (per-street EV):** ``_per_street_attribution(bb_delta,
   hand_decisions)`` apportions each hand's bb_delta across streets weighted
   by ``hero_chips_in`` per street. Hero-only-checked edge cases attribute
   to the last street observed. NOT an equal-split heuristic.

REFERENCES:
    - RESEARCH Pattern 8 (lines 720-757): match runner + bootstrap CI + verdict.
    - RESEARCH Pitfall 2: chip-to-BB conversion.
    - RESEARCH Pitfall 5: per-cluster contribution decomposition.
    - OQ-6 RESOLVED: engine_version read from ``pyproject.toml [project].version``.
"""

from __future__ import annotations

import contextlib
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np

from src._log import get_logger
from src.eval.baselines import REGISTRY
from src.eval.result import MatchResult, persist_match
from src.eval.texture import classify_board
from src.eval.trace import DecisionTrace, EngineRecorder

log = get_logger("eval.run_match")


_STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")


def _bootstrap_ci(
    deltas: list[float],
    n_resamples: int = 1000,
    ci_alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Non-parametric bootstrap CI on the mean of per-hand bb deltas, scaled to bb/100.

    Args:
        deltas: per-hand bb_delta values.
        n_resamples: number of bootstrap resamples (RESEARCH Pattern 8 default).
        ci_alpha: significance level for the two-sided CI (default 0.05 -> 95%).
        seed: seed for the resampling RNG (deterministic).

    Returns:
        ``(ci_low, ci_high)`` in bb/100 units.
    """
    arr = np.asarray(deltas, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    resamples = rng.choice(arr, size=(n_resamples, len(arr)), replace=True)
    means = resamples.mean(axis=1) * 100  # bb/100 scaling
    return (
        float(np.quantile(means, ci_alpha / 2)),
        float(np.quantile(means, 1 - ci_alpha / 2)),
    )


def verdict(ci_low: float, ci_high: float, *, is_regression_test: bool = False) -> str:
    """Map (CI bounds, regression-flag) -> status badge per D-NEW-30.

    Mapping (CI is on bb/100):
        - ``ci_low > 0``  -> ``'won'``       (statistically significant edge)
        - ``ci_high < 0`` -> ``'lost'`` or ``'regression'`` (if vs prev-engine)
        - else            -> ``'inconclusive'``
    """
    if ci_low > 0:
        return "won"
    if ci_high < 0:
        return "regression" if is_regression_test else "lost"
    return "inconclusive"


def _engine_version() -> str | None:
    """Return ``pyproject.toml [project].version`` per OQ-6 RESOLVED.

    Returns None if pyproject.toml is missing, unreadable, or has no version key.
    This is best-effort — engine_version is informational metadata.

    NEVER accepts version from API input (T-06-22 mitigation).
    """
    try:
        import tomllib
        from pathlib import Path

        data = tomllib.loads(Path("pyproject.toml").read_text())
        return data.get("project", {}).get("version")
    except Exception:
        return None


def _classify_street(n_public_cards: int) -> str:
    """RLCard board card count -> street name."""
    if n_public_cards == 0:
        return "preflop"
    if n_public_cards == 3:
        return "flop"
    if n_public_cards == 4:
        return "turn"
    return "river"


def _extract_board(trajectories: Any) -> list[str]:
    """Extract the final public board from RLCard trajectories. Returns [] if no flop."""
    try:
        last_state = trajectories[0][-1]
        # RLCard wraps state as dict; raw_obs may be nested.
        raw = last_state.get("raw_obs", last_state) if isinstance(last_state, dict) else {}
        return list(raw.get("public_cards", []))
    except Exception:
        return []


def _per_street_attribution(
    bb_delta: float,
    hand_decisions: list,
) -> dict[str, float]:
    """Apportion a hand's bb_delta across streets proportional to hero_chips_in per street.

    Args:
        bb_delta: total bb_delta for this hand.
        hand_decisions: list of ``RecordedDecision`` for this hand.

    Returns:
        ``{preflop, flop, turn, river}`` -> apportioned bb_delta (always present,
        zero for streets hero never acted on).

    Edge cases (Blocker 4):
        - Hero never put chips in (all checks): attribute bb_delta to the LAST
          street hero acted on (rather than equal-split).
        - No decisions at all: attribute to ``'preflop'`` as a safety default
          (should not occur — RLCard always emits ≥1 decision per hand).
    """
    per_street_chips = {s: 0.0 for s in _STREETS}
    for d in hand_decisions:
        if d.street in per_street_chips:
            per_street_chips[d.street] += d.hero_chips_in
    total = sum(per_street_chips.values())
    out = {s: 0.0 for s in _STREETS}
    if total > 0:
        for s in _STREETS:
            out[s] = bb_delta * (per_street_chips[s] / total)
    elif hand_decisions:
        last_street = hand_decisions[-1].street
        if last_street in out:
            out[last_street] = bb_delta
        else:
            out["preflop"] = bb_delta
    else:
        out["preflop"] = bb_delta  # safety
    return out


def run_match(
    opponent_name: str | None = None,
    *,
    opponents: list[str] | None = None,
    hands: int = 10000,
    seed: int = 42,
    table_size: int = 2,
    persist: bool = True,
    opponent_engine: Any = None,
    opponent_label: str | None = None,
    _engine: Any = None,
    _tsdb_conn: Any = None,
    job: Any = None,
) -> MatchResult:
    """Run engine vs baseline opponent for N hands.

    Args:
        opponent_name: REGISTRY key (e.g. ``'random'``, ``'tight-passive'``).
        hands: Number of hands to play.
        seed: Deterministic seed (drives RLCard env + bootstrap CI + opponent RNG).
        table_size: Number of seats at the table. ``2`` = heads-up (default);
            ``6`` = 6-max (engine vs 5 independently-seeded opponent copies).
        persist: When True, INSERTs the MatchResult into the matches hypertable.
        _engine: Optional injected engine (tests use a stub). When None, builds
            a real ``KNNDecisionEngine`` via ``engine_from_env``.
        _tsdb_conn: Optional injected psycopg connection (tests).
        job: Optional JobRegistry job (Plan 07) — when supplied, emits progress
            events every 100 hands. Acceptable for events.put_nowait to fail.

    Returns:
        ``MatchResult`` with bb_per_100, CI bounds, per-street + per-texture
        breakdown, top5_profitable + top5_leaky.

    Raises:
        KeyError: when ``opponent_name`` is not in ``REGISTRY``.
    """
    log.info("eval.run_match.started", opponent=opponent_name, opponents=opponents, hands=hands, seed=seed)
    # Head-to-head: the opponent seat is a pinned engine, not a baseline strategy.
    if opponent_engine is not None:
        if table_size != 2:
            raise ValueError("opponent_engine is head-to-head only (table_size=2)")
        result_opponent = opponent_label or "engine"
    # Per-seat opponents (heterogeneous table) override the cloned single opponent.
    elif opponents is not None:
        if len(opponents) != table_size - 1:
            raise ValueError(
                f"opponents needs {table_size - 1} entries for table_size={table_size}, got {len(opponents)}"
            )
        for nm in opponents:
            if nm not in REGISTRY:
                raise KeyError(f"unknown opponent {nm!r}; valid: {sorted(REGISTRY)}")
        result_opponent = "+".join(opponents)
    else:
        if opponent_name not in REGISTRY:
            raise KeyError(f"unknown opponent {opponent_name!r}; valid: {sorted(REGISTRY)}")
        result_opponent = opponent_name

    started_at = datetime.now(UTC)
    opponent = REGISTRY[opponent_name](seed=seed + 1) if opponent_name is not None else None
    base_engine = _engine
    if base_engine is None:
        # Lazy import avoids requiring Milvus/TSDB at module-import time.
        from src.decision_engine.engine import engine_from_env

        base_engine = engine_from_env(rng_seed=seed)

    # Blocker 3 — wrap engine with recorder for real cluster attribution.
    trace = DecisionTrace()
    recorder = EngineRecorder(base_engine, trace)

    # RLCard env construction — chips_for_each=100, BB=2 (default) → 1 chip = 0.5 BB
    # (Pitfall 2 / Blocker 2). table_size=2 is HU; table_size=6 is 6-max 1-vs-5.
    import rlcard

    if table_size == 2:
        env = rlcard.make(
            "no-limit-holdem",
            config={"seed": seed, "game_num_players": 2, "chips_for_each": 100},
        )
        if opponent_engine is not None:
            villain_recorder = EngineRecorder(opponent_engine, DecisionTrace())
            env.set_agents([_engine_agent(recorder, env), _engine_agent(villain_recorder, env)])
        else:
            opp0 = REGISTRY[opponents[0]](seed=seed + 1) if opponents is not None else opponent
            env.set_agents([_engine_agent(recorder, env), _baseline_agent(opp0)])
    else:
        env = rlcard.make(
            "no-limit-holdem",
            config={"seed": seed, "game_num_players": table_size, "chips_for_each": 100},
        )
        if opponents is None:
            opponent_agents = [
                _baseline_agent(type(opponent)(seed=seed + 1 + i)) for i in range(table_size - 1)
            ]
        else:
            opponent_agents = [
                _baseline_agent(REGISTRY[opponents[i]](seed=seed + 1 + i)) for i in range(table_size - 1)
            ]
        env.set_agents([_engine_agent(recorder, env), *opponent_agents])

    per_hand_deltas: list[float] = []
    per_street_totals: dict[str, float] = dict.fromkeys(_STREETS, 0.0)
    per_texture_totals: dict[str, float] = {
        "dry_rainbow": 0.0,
        "wet_two_tone": 0.0,
        "paired": 0.0,
        "monotone": 0.0,
    }
    payoffs_by_hand: dict[int, float] = {}

    for hand_i in range(hands):
        recorder.start_hand(hand_i)
        trajectories, payoffs = env.run(is_training=False)

        # Blocker 2 / Pitfall 2: chips_for_each=100, BB=2 → 1 chip = 0.5 BB.
        # Inline conversion — no helper function (Pitfall 2 / Blocker 2).
        bb_delta = float(payoffs[0]) / 2.0
        per_hand_deltas.append(bb_delta)
        payoffs_by_hand[hand_i] = bb_delta

        # Per-texture attribution: classify board, accumulate by category.
        board = _extract_board(trajectories)
        if board:
            tx = classify_board(board)
            per_texture_totals[tx] += bb_delta

        # Per-street attribution (Blocker 4): proportional to hero_chips_in.
        hand_decisions = [d for d in trace if d.hand_id == hand_i]
        per_street_for_hand = _per_street_attribution(bb_delta, hand_decisions)
        for s, v in per_street_for_hand.items():
            per_street_totals[s] += v

        # JobRegistry progress events (best-effort; Plan 07 consumer).
        if job is not None and hand_i % 100 == 0:
            with contextlib.suppress(Exception):
                job.events.put_nowait(
                    {
                        "event": "progress",
                        "data": {"hands_done": hand_i, "hands_total": hands},
                    }
                )

    bb_per_100 = float(np.mean(per_hand_deltas) * 100) if per_hand_deltas else 0.0
    ci_low, ci_high = _bootstrap_ci(per_hand_deltas, seed=seed)
    finished_at = datetime.now(UTC)
    is_regr = opponent_name == "prev-engine"

    # Blocker 3 — real top-5 lists derived from trace.
    top5_profitable = trace.top5_profitable(payoffs_by_hand)
    top5_leaky = trace.top5_leaky(payoffs_by_hand)

    result = MatchResult(
        match_id=str(uuid.uuid4()),
        opponent=result_opponent,
        hands=hands,
        seed=seed,
        bb_per_100=bb_per_100,
        ci_low=ci_low,
        ci_high=ci_high,
        per_street=per_street_totals,
        per_texture=per_texture_totals,
        top5_profitable=top5_profitable,
        top5_leaky=top5_leaky,
        status=verdict(ci_low, ci_high, is_regression_test=is_regr),
        engine_version=_engine_version(),
        started_at=started_at,
        finished_at=finished_at,
    )
    if persist:
        persist_match(result, _tsdb_conn=_tsdb_conn)
    log.info(
        "eval.run_match.complete",
        match_id=result.match_id,
        status=result.status,
        bb_per_100=result.bb_per_100,
        ci_low=result.ci_low,
        ci_high=result.ci_high,
        n_profitable=len(top5_profitable),
        n_leaky=len(top5_leaky),
        nostrategy_fallbacks=recorder.nostrategy_fallbacks,
    )
    return result


def _engine_agent(recorder: EngineRecorder, env: Any = None) -> Any:
    """Adapt EngineRecorder.decide -> RLCard agent interface (.step(state) -> action).

    The recorder is fed the raw RLCard state dict (containing ``raw_obs`` with
    public cards + legal actions). The wrapped engine is responsible for
    coercing it to whatever shape it needs (real KNNDecisionEngine consumes a
    project ``GameState``; test stubs accept the dict directly).

    Street is derived from ``public_cards`` length and passed to the recorder
    via ``note_street`` before each ``decide()``.
    """
    from src.sim.adapter import (
        _AGGRESSIVE_FALLBACK_ORDER,
        _IN_HAND,
        _PASSIVE_FALLBACK_ORDER,
        PROJECT_TO_RLCARD,
        _coerce_int,
        intent_class,
        rlcard_state_to_gamestate,
    )

    def _legal_ints(state) -> list[int]:
        if not isinstance(state, dict):
            return []
        legal = state.get("legal_actions")
        if isinstance(legal, dict):
            return [_coerce_int(k) for k in legal]
        raw = state.get("raw_obs", {})
        return [_coerce_int(a) for a in raw.get("legal_actions", [])] if isinstance(raw, dict) else []

    def _snap_to_legal(mapped: int, legal: list[int], verb: str) -> int:
        """Intent-aware fallback: aggression prefers raise/ALL_IN before CHECK_CALL."""
        if not legal or mapped in legal:
            return mapped
        order = _AGGRESSIVE_FALLBACK_ORDER if intent_class(verb) == "aggressive" else _PASSIVE_FALLBACK_ORDER
        for fb in order:
            if fb in legal:
                return fb
        return legal[0]

    class _Adapter:
        use_raw = False

        def step(self, state):
            raw = state.get("raw_obs", state) if isinstance(state, dict) else {}
            n_cards = len(raw.get("public_cards", [])) if isinstance(raw, dict) else 0
            recorder.note_street(_classify_street(n_cards))
            # Production KNNDecisionEngine expects a GameState; stub engines (tests)
            # accept the raw dict. Build the GameState first; only a *construction*
            # failure falls back to feeding the raw dict (the stub shape). A failure
            # inside the engine's decide() is NOT masked — a broken engine / empty KB
            # (NoStrategyError) must surface so a degraded engine can't silently score
            # (ERR-01 fail-loud).
            dealer_id = active_count = None
            if env is not None:
                try:
                    dealer_id = env.game.dealer_id
                    active_count = sum(1 for p in env.game.players if p.status in _IN_HAND)
                except Exception:
                    dealer_id = active_count = None
            try:
                engine_input: Any = rlcard_state_to_gamestate(
                    state, dealer_id=dealer_id, active_count=active_count
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                engine_input = state
            result = recorder.decide(engine_input)
            action_str = result[0] if isinstance(result, tuple) else str(result)
            mapped = PROJECT_TO_RLCARD.get(action_str, 1)  # default CHECK_CALL on miss
            return _snap_to_legal(mapped, _legal_ints(state), action_str)

        def eval_step(self, state):
            return self.step(state), {}

    return _Adapter()


def _baseline_agent(strategy: Any) -> Any:
    """Adapt ``Strategy.decide(state) -> int`` to RLCard agent interface."""

    class _Adapter:
        use_raw = False

        def step(self, state):
            return int(strategy.decide(state))

        def eval_step(self, state):
            return self.step(state), {}

    return _Adapter()
