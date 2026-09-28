from __future__ import annotations

from src.decision_engine.blending import blend_distributions


def test_solver_outranks_human():
    human_hit = {
        "distance": 0.0,
        "entity": {
            "hero_action_type": "check",
            "confidence": 1.0,
            "gto_score": 0.911,
        },
    }
    solver_hit = {
        "distance": 0.0,
        "entity": {
            "hero_action_type": "bet_50",
            "confidence": 1.0,
            "gto_score": 1.0,
        },
    }

    human_weight = 1.0 * 1.0 * 0.911
    solver_weight = 1.0 * 1.0 * 1.0

    assert solver_weight > human_weight

    dist = blend_distributions([human_hit, solver_hit])

    assert dist["bet_50"] > dist["check"]
