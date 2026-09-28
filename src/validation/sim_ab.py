"""src/validation/sim_ab.py — Paired-seed A/B validator with bootstrap CI.

VALN-01: Paired-seed determinism
    validate() runs the sim harness TWICE with the SAME session_seed — once with
    the baseline KNNDecisionEngine and once with a StrategyOverride wrapping it
    as the candidate. Identical seeds guarantee identical game sequences (SIM-02);
    only the action distributions differ for the target cluster_key.

VALN-02: ValidationResult shape
    Returns ValidationResult(seed, ev_loss_delta, n_hands, confidence_interval,
    n_cluster_hits, zero_hits). All fields required; stored as JSONB in patches table.

VALN-03: Exception propagation
    Exceptions from run_record_session propagate to the caller (the auto-loop driver
    in Plan 07 catches them and logs; validate() itself does NOT swallow them).

DryRunObservationWriter:
    Replaces the production ObservationWriter during A/B runs. API-compatible (same
    record() signature, context-manager, close()), but stores rows in a list instead
    of writing to TimescaleDB. Zero DB I/O.

bootstrap_ci:
    Non-parametric 95% CI on the mean of per-hand ev_loss deltas.
    Empty input → (0.0, 0.0) (Pitfall 5 guard).
    Deterministic with a seeded numpy RNG (VALN-01 requirement).
"""

from __future__ import annotations

import contextlib
import uuid
from math import log as math_log
from typing import Any

import numpy as np

from src._log import get_logger
from src.decision_engine.blending import CANONICAL_ACTIONS  # noqa: F401 (re-used implicitly)
from src.metrics.ev_loss import _expected_action_dist, _street_class_from_cluster_key
from src.sim.harness import run_record_session
from src.validation.override import StrategyOverride

log = get_logger("validation.sim_ab")

# Additive smoothing constant — matches ev_loss.EPSILON for apples-to-apples delta.
_EPSILON: float = 1e-6


# ---------------------------------------------------------------------------
# DryRunObservationWriter
# ---------------------------------------------------------------------------


class DryRunObservationWriter:
    """No-op writer for A/B validation. Buffers rows in-memory; zero DB I/O.

    API-compatible with ObservationWriter:
        - Same record() signature (cluster_key, embedding, action_taken, **kwargs).
        - Same context-manager protocol (__enter__ / __exit__).
        - close() returns the count of buffered rows.

    NEVER opens a DB connection. NEVER writes to TimescaleDB.
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._rows: list[dict] = []

    def record(
        self,
        cluster_key: str,
        embedding,
        action_taken: str,
        *,
        flagged_sparse: bool = False,
        max_neighbor_distance: float | None = None,
        hand_id: str | None = None,
        decision_id: str | None = None,
        felt_snapshot: dict | None = None,
    ) -> None:
        """Buffer one observation row (no DB write).

        Signature mirrors ObservationWriter.record exactly so run_record_session
        can swap writers without a TypeError. validate() only reads cluster_key,
        action_taken, and embedding; the remaining fields are buffered for symmetry.
        """
        self._rows.append(
            {
                "cluster_key": cluster_key,
                "embedding": list(embedding),
                "action_taken": action_taken,
                "flagged_sparse": flagged_sparse,
                "max_neighbor_distance": max_neighbor_distance,
                "hand_id": hand_id,
                "decision_id": decision_id,
                "felt_snapshot": felt_snapshot,
                "session_id": self._session_id,
            }
        )

    def close(self) -> int:
        """Return the total number of buffered rows. No flush — no DB to flush to."""
        return len(self._rows)

    def __enter__(self) -> DryRunObservationWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def rows(self) -> list[dict]:
        """All buffered observation row dicts."""
        return self._rows


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def bootstrap_ci(
    per_hand_deltas: list[float],
    n_resamples: int,
    alpha: float,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Non-parametric bootstrap CI on the mean of per-hand ev_loss deltas.

    From 05-RESEARCH.md Pattern 3 (locked by 05-CONTEXT.md Decision 3).

    Args:
        per_hand_deltas: list of (log P_candidate - log P_baseline) per hand that
            hit the target cluster_key. Positive → candidate is more aligned with Q.
        n_resamples: number of bootstrap resamples (default 1000 in config).
        alpha: significance level (0.05 → 95% CI).
        rng: numpy Generator (deterministic when seeded; default_rng() if None).

    Returns:
        (ci_low, ci_high). PatchEngine accepts patch only if ci_low > 0 (LOOP-03).
        Returns (0.0, 0.0) if per_hand_deltas is empty (Pitfall 5 guard).
    """
    if not per_hand_deltas:
        return 0.0, 0.0

    rng = rng if rng is not None else np.random.default_rng()
    arr = np.array(per_hand_deltas, dtype=np.float64)
    means = np.array(
        [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_resamples)]
    )
    ci_low = float(np.percentile(means, 100 * alpha / 2))
    ci_high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return ci_low, ci_high


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def validate(
    patch_spec: Any,
    config: Any,
    *,
    baseline_engine: Any,
    sim_adapter: Any,
    seed: int,
    _tsdb_conn: Any = None,
    _milvus: Any = None,
) -> Any:
    """Run paired-seed A/B; return ValidationResult.

    Steps (locked by 05-CONTEXT.md Decision 3):
      1. Baseline run: drive sim harness with baseline_engine + DryRunObservationWriter.
      2. Candidate run: same seed, StrategyOverride(baseline_engine, ...) engine.
         Both runs use the SAME session_seed (VALN-01).
      3. Filter observations that hit the target cluster_key in both runs.
         If zero hits → return zero_hits sentinel (Pitfall 5).
      4. Per-hand log-likelihood delta against Q (the expected dist from Milvus kNN).
         delta_h = log(Q[candidate_action] + ε) - log(Q[baseline_action] + ε)
         Positive → candidate action is more aligned with Q → improvement.
      5. Bootstrap CI on per-hand deltas; return populated ValidationResult.

    Args:
        patch_spec: PatchSpec with .cluster_key and .action_dist fields.
        config: AutoLoopConfig with .n_hands_validation, .bootstrap_resamples, .ci_alpha.
        baseline_engine: KNNDecisionEngine (the current production engine).
        sim_adapter: SimAdapter (RLCard wrapper) for the harness.
        seed: paired seed for VALN-01 determinism.
        _tsdb_conn: test-injection hook (unused in validate(); present for API symmetry
            with other Phase 5 functions — future callers may pass it through).
        _milvus: test-injection hook. Pre-built MilvusClient for _expected_action_dist.

    Returns:
        ValidationResult with all VALN-02 fields populated.

    Raises:
        Any exception from run_record_session — caller (Plan 07 driver) catches (VALN-03).
    """
    # Lazily import to avoid circular dependency if patch_engine imports us.
    from src.patch_engine import ValidationResult

    log_ctx = get_logger("validation.sim_ab.validate")
    n_hands = config.n_hands_validation
    target_key = patch_spec.cluster_key

    # ---- 1. Baseline run ----
    baseline_sid = f"ab-baseline-{uuid.uuid4()}"
    baseline_writer = DryRunObservationWriter(session_id=baseline_sid)
    run_record_session(
        adapter=sim_adapter,
        engine=baseline_engine,
        writer=baseline_writer,
        session_seed=seed,
        n_hands=n_hands,
        session_id=baseline_sid,
    )

    # ---- 2. Candidate run — same seed, StrategyOverride engine ----
    candidate_engine = StrategyOverride(
        baseline_engine,
        target_key,
        patch_spec.action_dist,
        rng=np.random.default_rng(seed),
    )
    candidate_sid = f"ab-candidate-{uuid.uuid4()}"
    candidate_writer = DryRunObservationWriter(session_id=candidate_sid)
    run_record_session(
        adapter=sim_adapter,
        engine=candidate_engine,
        writer=candidate_writer,
        session_seed=seed,
        n_hands=n_hands,
        session_id=candidate_sid,
    )

    # ---- 3. Filter to target cluster hits ----
    baseline_hits = [r for r in baseline_writer.rows if r["cluster_key"] == target_key]
    candidate_hits = [r for r in candidate_writer.rows if r["cluster_key"] == target_key]
    n_cluster_hits = min(len(baseline_hits), len(candidate_hits))

    if n_cluster_hits == 0:
        log_ctx.warning("validation.sim_ab.zero_hits", cluster_key=target_key, seed=seed)
        return ValidationResult(
            seed=seed,
            ev_loss_delta=0.0,
            n_hands=n_hands,
            confidence_interval=(0.0, 0.0),
            n_cluster_hits=0,
            zero_hits=True,
        )

    # ---- 4. Per-hand log-likelihood delta against Q ----
    # Q = expected action distribution (decide-time blend of kNN neighbors).
    # Use the first baseline hit's embedding as representative — same convention as ev_loss().
    street_class = _street_class_from_cluster_key(target_key)
    collection = "preflop_decisions" if street_class.lower() == "preflop" else "postflop_decisions"

    _owns_client = _milvus is None
    if _milvus is not None:
        client = _milvus
    else:
        from src.db import milvus as milvus_db

        client = milvus_db.connect_from_env()

    representative_embedding = list(baseline_hits[0]["embedding"])
    try:
        q_dist = _expected_action_dist(client, representative_embedding, collection)
    finally:
        if _owns_client:
            with contextlib.suppress(Exception):
                client.close()

    per_hand_deltas: list[float] = []
    for base_row, cand_row in zip(
        baseline_hits[:n_cluster_hits], candidate_hits[:n_cluster_hits], strict=False
    ):
        p_base = q_dist.get(base_row["action_taken"], 0.0) + _EPSILON
        p_cand = q_dist.get(cand_row["action_taken"], 0.0) + _EPSILON
        # Positive delta → candidate action has higher Q-likelihood → improvement.
        per_hand_deltas.append(math_log(p_cand) - math_log(p_base))

    # ---- 5. Bootstrap CI ----
    ci_low, ci_high = bootstrap_ci(
        per_hand_deltas,
        config.bootstrap_resamples,
        config.ci_alpha,
        rng=np.random.default_rng(seed),
    )
    mean_delta = float(np.mean(per_hand_deltas))

    log_ctx.info(
        "validation.sim_ab.complete",
        cluster_key=target_key,
        seed=seed,
        n_cluster_hits=n_cluster_hits,
        ev_loss_delta=round(mean_delta, 6),
        ci_low=round(ci_low, 6),
        ci_high=round(ci_high, 6),
    )

    return ValidationResult(
        seed=seed,
        ev_loss_delta=mean_delta,
        n_hands=n_hands,
        confidence_interval=(ci_low, ci_high),
        n_cluster_hits=n_cluster_hits,
        zero_hits=False,
    )


# ---------------------------------------------------------------------------
# Env-var helpers (production path only)
# ---------------------------------------------------------------------------
