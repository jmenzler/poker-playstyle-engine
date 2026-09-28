"""src/validation/override.py — In-memory StrategyOverride adapter for A/B validation.

Purpose (VALN-01 / 05-RESEARCH.md Pattern 2):
    Wraps a KNNDecisionEngine and intercepts decide_with_encoding() for exactly
    ONE cluster_key — the target cluster being validated. For all other
    cluster_keys the call is delegated to the underlying engine unchanged.

    The intercept path samples an action from a candidate action_dist (dict) using
    the same sample_action() helper as the production engine, but WITHOUT making
    any Milvus search call. This keeps A/B runs fully sandboxed from the KB:
    - Zero Milvus calls for the target cluster during the candidate run.
    - Zero TimescaleDB writes throughout the override's lifetime.
    - No DB connection is ever opened by this module.

Usage:
    override = StrategyOverride(
        base_engine=engine,
        cluster_key="street_class=postflop|...",
        candidate_dist={"bet_50": 0.7, "fold": 0.3},
        rng=np.random.default_rng(seed),
    )
    action, flagged_sparse, max_neighbor_distance, enc = override.decide_with_encoding(gs)
"""

from __future__ import annotations

import numpy as np

from src._log import get_logger
from src.decision_engine.blending import sample_action
from src.decision_engine.engine import KNNDecisionEngine
from src.protocols.game_state import GameState

log = get_logger("validation.override")


def _cluster_key_from_filter(hard_filter: dict) -> str:
    """Build cluster_key from a hard_filter dict (matches harness canonical form).

    Inverse of decomposing a cluster_key string — format is sorted 'key=value'
    pairs joined by '|'. Matches src.sim.harness._cluster_key_from_filter exactly.

    Example:
        {"pot_type": "srp", "street_class": "postflop"} -> "pot_type=srp|street_class=postflop"
    """
    return "|".join(f"{k}={v}" for k, v in sorted(hard_filter.items()))


class StrategyOverride:
    """A/B-sandbox intercept for one cluster_key.

    For the target cluster_key: samples from candidate_dist (no Milvus call).
    For all other cluster_keys: delegates to the underlying KNNDecisionEngine.
    NEVER writes to any DB. NEVER opens a DB connection.

    Sharing the base engine's canonicalizer (read-only) ensures the same
    embedding + hard_filter are used whether we intercept or delegate —
    guaranteeing format identity between baseline and candidate observation rows.

    Args:
        base_engine: The production KNNDecisionEngine; used for delegation and
            for its _canon.encode() method on the intercept path.
        cluster_key: The target cluster_key string (canonical sorted form).
        candidate_dist: Action probability distribution for the candidate patch.
            Must be a dict[str, float] summing to 1.0.
        rng: numpy random Generator for sampling from candidate_dist.
            Caller controls the seed for VALN-01 determinism.
    """

    def __init__(
        self,
        base_engine: KNNDecisionEngine,
        cluster_key: str,
        candidate_dist: dict[str, float],
        rng: np.random.Generator,
    ) -> None:
        self._base = base_engine
        self._target_key = cluster_key
        self._candidate_dist = candidate_dist
        self._rng = rng

    def decide_with_encoding(self, gs: GameState):
        """Return (action, flagged_sparse, max_neighbor_distance, EncodeResult).

        Matches the KNNDecisionEngine.decide_with_encoding 4-tuple contract.
        Note: Phase 3 KNNDecisionEngine.decide_with_encoding returns only a
        2-tuple (action, enc); the harness calls this via the engine API.
        StrategyOverride extends to the 4-tuple used by the harness writer
        (flagged_sparse, max_neighbor_distance are needed by DryRunObservationWriter).

        Intercept path (target cluster_key):
            - Encodes gs via base._canon to get enc (reads hard_filter).
            - Samples action from candidate_dist using sample_action().
            - Returns (action, False, None, enc) — flagged_sparse=False,
              max_neighbor_distance=None (same fallback shape as NoStrategyError
              path in harness.py lines 134-135).
            - Zero Milvus calls.

        Delegate path (non-target cluster_key):
            - Forwards to base_engine.decide_with_encoding(gs) unchanged.
            - Returns the base engine's native tuple.
        """
        enc = self._base._canon.encode(gs)
        ck = _cluster_key_from_filter(enc.hard_filter)
        if ck == self._target_key:
            # Intercept: sample from candidate_dist; skip all Milvus I/O.
            action = sample_action(self._candidate_dist, self._rng)
            log.debug(
                "validation.override.intercept",
                cluster_key=ck,
                action=action,
            )
            return action, False, None, enc

        # Non-target cluster: delegate to base engine. Preserves original tuple shape.
        # NOTE (CR-02): enc is already computed above but the base engine's
        # decide_with_encoding() re-encodes gs internally (no decide_from_encoding()
        # API exists on KNNDecisionEngine yet). This causes a double-encode on the
        # delegate path. The double-encode is benign when the canonicalizer is
        # deterministic (pure function) because both calls produce identical enc
        # objects. Marked for Phase 5.x refactor when KNNDecisionEngine exposes
        # decide_from_encoding(enc) to allow callers to pass a pre-computed enc.
        _ = enc  # silence unused-variable warning; enc was used for ck check above
        return self._base.decide_with_encoding(gs)

    def decide(self, gs: GameState) -> str:
        """Adapter for callers that use the synchronous .decide() API.

        Extracts only the action from the full 4-tuple returned by
        decide_with_encoding(). Used by sim adapters that don't need enc.
        """
        action, *_ = self.decide_with_encoding(gs)
        return action
