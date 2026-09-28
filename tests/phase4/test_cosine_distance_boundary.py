"""Boundary conversion: pymilvus COSINE similarity -> cosine distance before blend/sparseness."""

from __future__ import annotations

import pytest


def _hits(*sims: float) -> list[dict]:
    return [
        {
            "distance": s,
            "entity": {
                "decision_id": f"n{i}",
                "hero_action_type": "call",
                "confidence": 1.0,
                "gto_score": 1.0,
            },
        }
        for i, s in enumerate(sims)
    ]


def test_similarity_converted_to_distance():
    """A pymilvus similarity of 0.95 must become a cosine distance of 0.05."""
    from src.decision_engine.engine import _cosine_sim_to_distance

    out = _cosine_sim_to_distance(_hits(0.95, 0.65))
    assert out[0]["distance"] == pytest.approx(0.05), "sim 0.95 -> dist 0.05"
    assert out[1]["distance"] == pytest.approx(0.35), "sim 0.65 -> dist 0.35"


def test_best_neighbor_keeps_lowest_distance():
    """Order preserved: the most-similar hit has the smallest distance."""
    from src.decision_engine.engine import _cosine_sim_to_distance

    out = _cosine_sim_to_distance(_hits(0.9, 0.8, 0.6))
    dists = [h["distance"] for h in out]
    assert dists == sorted(dists), "higher similarity must map to lower distance, monotonically"
    assert dists[0] < dists[-1]


def test_blend_upweights_most_similar_after_conversion():
    """After conversion, the highest-similarity neighbor's action dominates the blend."""
    from src.decision_engine.blending import blend_distributions
    from src.decision_engine.engine import _cosine_sim_to_distance

    hits = [
        {
            "distance": 0.95,
            "entity": {
                "decision_id": "a",
                "hero_action_type": "raise_3x",
                "confidence": 1.0,
                "gto_score": 1.0,
            },
        },
        {
            "distance": 0.60,
            "entity": {"decision_id": "b", "hero_action_type": "fold", "confidence": 1.0, "gto_score": 1.0},
        },
    ]
    dist = blend_distributions(_cosine_sim_to_distance(hits), collection="preflop_decisions")
    assert dist["raise_3x"] > dist["fold"], f"most-similar neighbor (raise) must dominate: {dist}"
