# rot-allow-file
"""SimHarness: RECORD-mode loop driver (SIM-01, SIM-02, SIM-03).

Drives the per-hand env loop with deterministic per-hand seeds derived from
a top-level session_seed. For each hand:

    1. session_rng.integers() -> per_hand_seed
    2. adapter._env.seed(per_hand_seed)     # MUST precede reset (Pitfall 1)
    3. adapter._env.reset()                  # consumes seed
    4. while not over and budget remains:
        a. gs = adapter.next_game_state()
        b. action, flagged_sparse, max_neighbor_distance, enc = engine.decide_with_encoding(gs)   # SINGLE encode
        c. cluster_key built from enc.hard_filter directly
        d. writer.record(cluster_key, embedding, action)
        e. rlcard_action = adapter.map_to_rlcard_action(action, adapter.current_legal_actions())
        f. adapter._env.step(rlcard_action)

Determinism contract (SIM-02):
    Two independent runs with the same session_seed produce byte-identical
    cluster_key/embedding/action_taken sequences (the only non-deterministic
    fields are obs_id and ts which are excluded from the SHA256 checksum).

Throughput contract (SIM-01):
    >= 30,000 hands/hr on PC with in-process Milvus + LRU cache pre-warm.

Count contract (SIM-03):
    Observation rows written = total decision points across all hands.
"""

from __future__ import annotations

import time
import uuid

import msgspec
import numpy as np

from src._errors import NoStrategyError
from src._log import get_logger

log = get_logger("sim.harness")

PROGRESS_EVERY = 1_000  # OBS-03 cadence
SAFETY_CAP_DECISIONS_PER_HAND = 200
FALLBACK_ACTION = "fold"


def _cluster_key_from_filter(hard_filter: dict) -> str:
    """Deterministic cluster_key from a canonicalizer hard_filter dict.

    Matches the Phase 1/2 canonical cluster_key format: sorted ``hard_filter``
    entries joined by ``"|"``, with ``key=value`` form per entry. Guarantees
    format identity between HH-ingest observations and sim-harness observations,
    so downstream consumers (auto-loop, study tool) see one canonical key space.

    Example:
        {"pot_type": "srp", "street_class": "postflop"} -> "pot_type=srp|street_class=postflop"
    """
    return "|".join(f"{k}={v}" for k, v in sorted(hard_filter.items()))


def run_record_session(
    adapter,
    engine,
    writer,
    session_seed: int,
    n_hands: int,
    *,
    session_id: str | None = None,
    max_decisions_per_hand: int = SAFETY_CAP_DECISIONS_PER_HAND,
) -> dict:
    """Drive a deterministic RECORD-mode session.

    Args:
        adapter: SimAdapter exposing ``self._env`` (RLCard env), ``has_more()``,
            ``next_game_state()``, ``current_legal_actions()``,
            ``map_to_rlcard_action(action, legal)``.
        engine: KNNDecisionEngine exposing ``decide_with_encoding(gs)`` and a
            ``_canon`` Canonicalizer attribute (for the rare NoStrategy fallback).
        writer: ObservationWriter exposing ``record(cluster_key, embedding,
            action_taken)``. The writer is owned by the caller; this function
            never closes it (the CLI uses a context manager).
        session_seed: top-level integer seed. Two runs with the same value
            produce byte-identical (cluster_key, embedding, action_taken)
            sequences (SIM-02).
        n_hands: number of hands to play in the session.
        session_id: defaults to a fresh UUID4 when None.
        max_decisions_per_hand: safety cap against infinite loops in adapter
            bugs. Real 6-max NLHE hands have <= 24 decisions; 200 catches any
            pathological loop without truncating legitimate play.

    Returns:
        Summary dict with keys:
            - total_hands (int)
            - total_decisions (int)
            - elapsed_seconds (float)
            - hands_per_hour (float)
            - nostrategy_fallbacks (int)
            - session_id (str)
            - latency_buffer (list[float]): per-decision elapsed seconds from
              time.perf_counter() wrapping engine.decide_with_encoding(). Used
              by flush_session_metrics() to compute latency_p50/p99 (METR-03).
              Timing wraps the call attempt, including failed calls that raise
              NoStrategyError. Wall-clock only — NOT part of any checksummed
              field (SIM-02 determinism unaffected).
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    session_rng = np.random.default_rng(session_seed)
    t0 = time.perf_counter()
    total_decisions = 0
    nostrategy_fallbacks = 0
    latency_buffer: list[float] = []  # METR-03: per-decision timing for flush_session_metrics

    for hand_idx in range(n_hands):
        per_hand_seed = int(session_rng.integers(0, 2**31))
        # CRITICAL: seed IMMEDIATELY before reset; no other adapter._env.* calls
        # between these two lines. RESEARCH.md Pitfall 1, verified live in Phase 3.
        adapter._env.seed(per_hand_seed)
        adapter._env.reset()

        hand_id = f"{session_id}_h{hand_idx}"
        decisions_this_hand = 0
        while adapter.has_more() and decisions_this_hand < max_decisions_per_hand:
            gs = adapter.next_game_state()
            # SINGLE encode call via the engine API (Plan 03-02 Task 2:
            # decide_with_encoding). The harness MUST NOT call
            # ``engine._canon.encode`` separately on the happy path — that would
            # double-encode every observation.
            #
            # METR-03: hoist perf_counter BEFORE the try block so the timer exists
            # when NoStrategyError is caught. Append once after the try/except to
            # capture the full call duration regardless of success or failure.
            _t_decide_start = time.perf_counter()
            try:
                action, flagged_sparse, max_neighbor_distance, enc = engine.decide_with_encoding(gs)
            except NoStrategyError as e:
                # Fall back legally: fold only when facing a bet, else check — never
                # fold for free (mirrors the engine's illegal-action mask).
                facing = float(getattr(gs, "hero_facing_bet_bb", 0) or 0)
                action = FALLBACK_ACTION if facing > 0 else "check"
                # Engine couldn't produce a strategy. Re-encode once to populate
                # the observation row. This is the ONLY place a second encode
                # happens, and only on the rare NoStrategy fallback path.
                enc = engine._canon.encode(gs)
                # NoStrategy means zero neighbors were retrieved — no max distance
                # exists. NULL is the semantically correct value (distinct from
                # "neighbors retrieved but far"). flagged_sparse=False matches the
                # migration default and avoids misleading the uncertain-spots CLI
                # with rows that have no distance signal at all.
                flagged_sparse = False
                max_neighbor_distance = None
                nostrategy_fallbacks += 1
                log.warning(
                    "harness.nostrategy_fallback",
                    hand_idx=hand_idx,
                    error=str(e),
                    session_id=session_id,
                )
            latency_buffer.append(time.perf_counter() - _t_decide_start)  # METR-03

            # cluster_key built from enc.hard_filter directly — matches the
            # Phase 1/2 canonical cluster_key format (sorted hard_filter entries
            # joined by "|"), guaranteeing format identity between HH-ingest
            # observations and sim-harness observations.
            cluster_key = _cluster_key_from_filter(enc.hard_filter)
            decision_id = f"{hand_id}_dp{decisions_this_hand}"
            felt_dict = msgspec.json.decode(msgspec.json.encode(gs))
            writer.record(
                cluster_key=cluster_key,
                embedding=list(enc.embedding),
                action_taken=action,
                flagged_sparse=flagged_sparse,
                max_neighbor_distance=max_neighbor_distance,
                hand_id=hand_id,
                decision_id=decision_id,
                felt_snapshot=felt_dict,
            )

            legal = adapter.current_legal_actions()
            if not legal:
                # Adapter signalled no legal actions but ``has_more()`` was True;
                # defensive break to avoid stalling the loop.
                log.warning(
                    "harness.no_legal_actions_break",
                    hand_idx=hand_idx,
                    decisions_this_hand=decisions_this_hand,
                    session_id=session_id,
                )
                break
            # Sizing-aware executor: drives a real raise-TO chip amount per verb,
            # with intent-aware fallback (aggression never downgraded to a call).
            adapter.execute_action(action)
            decisions_this_hand += 1
            total_decisions += 1

        if decisions_this_hand >= max_decisions_per_hand:
            log.warning(
                "harness.safety_cap_hit",
                hand_idx=hand_idx,
                max_decisions_per_hand=max_decisions_per_hand,
                session_id=session_id,
            )

        if (hand_idx + 1) % PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - t0
            rate = (hand_idx + 1) / max(elapsed, 1e-9) * 3600
            log.info(
                "harness.progress",
                hands=hand_idx + 1,
                total_target=n_hands,
                total_decisions=total_decisions,
                elapsed_s=round(elapsed, 1),
                hands_per_hr=round(rate),
                nostrategy_fallbacks=nostrategy_fallbacks,
                session_id=session_id,
            )

    elapsed = time.perf_counter() - t0
    rate = n_hands / max(elapsed, 1e-9) * 3600
    # M3: executed-vs-intended size/action divergences by intent class, so silent
    # downgrades (aggression -> call, sized bet -> clamped) are visible in metrics.
    size_divergences = adapter.divergence_counts()
    summary = {
        "total_hands": n_hands,
        "total_decisions": total_decisions,
        "elapsed_seconds": round(elapsed, 3),
        "hands_per_hour": round(rate),
        "nostrategy_fallbacks": nostrategy_fallbacks,
        "size_divergences": size_divergences,
        "session_id": session_id,
        "latency_buffer": latency_buffer,  # METR-03
    }
    log.info(
        "harness.session_complete",
        total_hands=summary["total_hands"],
        total_decisions=summary["total_decisions"],
        elapsed_seconds=summary["elapsed_seconds"],
        hands_per_hour=summary["hands_per_hour"],
        nostrategy_fallbacks=summary["nostrategy_fallbacks"],
        size_divergences=size_divergences,
        session_id=summary["session_id"],
        latency_count=len(latency_buffer),
    )
    return summary
