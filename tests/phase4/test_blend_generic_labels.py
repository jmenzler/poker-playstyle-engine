"""The Milvus index stores collapsed action labels ('raise', 'bet') with no
size, but the blend vocab only has sized actions. These tests pin the alias
that folds generic labels into one canonical bucket so raise/bet mass is not
silently dropped (which made the engine never raise or bet).
"""

from __future__ import annotations

import pytest


def _hit(action: str, distance: float = 0.05) -> dict:
    return {
        "distance": distance,
        "entity": {"decision_id": "x", "hero_action_type": action, "confidence": 1.0, "gto_score": 1.0},
    }


def test_generic_raise_not_dropped():
    """A neighbor labelled bare 'raise' must contribute mass to a raise bucket."""
    from src.decision_engine.blending import blend_distributions

    dist = blend_distributions([_hit("raise"), _hit("fold")], collection="preflop_decisions")
    raise_mass = sum(v for k, v in dist.items() if k.startswith("raise"))
    assert raise_mass > 0.0, f"generic 'raise' must map to a raise bucket, got {dist}"


def test_generic_bet_not_dropped():
    """A neighbor labelled bare 'bet' must contribute mass to a bet bucket."""
    from src.decision_engine.blending import blend_distributions

    dist = blend_distributions([_hit("bet"), _hit("check")], collection="postflop_decisions")
    bet_mass = sum(v for k, v in dist.items() if k.startswith("bet"))
    assert bet_mass > 0.0, f"generic 'bet' must map to a bet bucket, got {dist}"


def test_raise_majority_yields_raise_action():
    """All-raise neighborhood must blend to ~all raise mass (not dropped to fold)."""
    from src.decision_engine.blending import blend_distributions

    dist = blend_distributions([_hit("raise"), _hit("raise"), _hit("raise")], collection="preflop_decisions")
    raise_mass = sum(v for k, v in dist.items() if k.startswith("raise"))
    assert raise_mass == pytest.approx(1.0), f"all-raise must be ~100% raise mass, got {dist}"
