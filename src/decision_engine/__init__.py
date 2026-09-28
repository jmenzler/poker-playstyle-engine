"""src/decision_engine — kNN DecisionEngine (Phase 3).

Public exports:
    KNNDecisionEngine    — implements DecisionEngine protocol
    blend_distributions  — pure: list[dict] -> dict[action, prob]
    sample_action        — pure: (dist, rng) -> action_str
    CANONICAL_ACTIONS    — tuple of the 15 canonical action strings
"""

from src.decision_engine.blending import (
    CANONICAL_ACTIONS,
    blend_distributions,
    sample_action,
)
from src.decision_engine.engine import KNNDecisionEngine

__all__ = [
    "CANONICAL_ACTIONS",
    "KNNDecisionEngine",
    "blend_distributions",
    "sample_action",
]
