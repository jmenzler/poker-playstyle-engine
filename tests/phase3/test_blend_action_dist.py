"""DP-level patches store a full action_dist on their Milvus row (VARCHAR JSON).
The blend must split that row's weight across the distribution; base-corpus rows
(no action_dist) fall back to the single hero_action_type label.
"""

from __future__ import annotations

import json

import pytest

from src.decision_engine.blending import blend_distributions


def _hit(entity: dict, distance: float = 0.0) -> dict:
    return {"distance": distance, "entity": entity}


def test_action_dist_splits_weight_evenly() -> None:
    hit = _hit(
        {
            "hero_action_type": "fold",
            "confidence": 1.0,
            "gto_score": 1.0,
            "action_dist": json.dumps({"fold": 0.5, "call": 0.5}),
        }
    )
    dist = blend_distributions([hit], collection="postflop_decisions")
    assert dist["fold"] == pytest.approx(0.5)
    assert dist["call"] == pytest.approx(0.5)


def test_action_dist_uneven_split() -> None:
    hit = _hit(
        {
            "hero_action_type": "bet_50",
            "confidence": 1.0,
            "gto_score": 1.0,
            "action_dist": json.dumps({"bet_50": 0.75, "check": 0.25}),
        }
    )
    dist = blend_distributions([hit], collection="postflop_decisions")
    assert dist["bet_50"] == pytest.approx(0.75)
    assert dist["check"] == pytest.approx(0.25)


def test_empty_action_dist_falls_back_to_label() -> None:
    hit = _hit({"hero_action_type": "call", "confidence": 1.0, "gto_score": 1.0, "action_dist": ""})
    dist = blend_distributions([hit], collection="postflop_decisions")
    assert dist["call"] == pytest.approx(1.0)


def test_missing_action_dist_falls_back_to_label() -> None:
    hit = _hit({"hero_action_type": "call", "confidence": 1.0, "gto_score": 1.0})
    dist = blend_distributions([hit], collection="postflop_decisions")
    assert dist["call"] == pytest.approx(1.0)


def test_action_dist_row_and_label_row_blend_equally() -> None:
    # A patch row carrying {fold:1.0} contributes the same mass as a base row
    # whose single label is fold — total weight per row is identical.
    patch_hit = _hit(
        {
            "hero_action_type": "fold",
            "confidence": 1.0,
            "gto_score": 1.0,
            "action_dist": json.dumps({"call": 1.0}),
        }
    )
    base_hit = _hit({"hero_action_type": "fold", "confidence": 1.0, "gto_score": 1.0})
    dist = blend_distributions([patch_hit, base_hit], collection="postflop_decisions")
    assert dist["call"] == pytest.approx(0.5)
    assert dist["fold"] == pytest.approx(0.5)


def test_action_dist_multi_action_split_weights_row_once() -> None:
    # Three non-zero entries from a single row: total_weight is added once, so the
    # fractions are preserved (a per-entry total_weight bug would divide by 3).
    hit = _hit(
        {
            "hero_action_type": "fold",
            "confidence": 1.0,
            "gto_score": 1.0,
            "action_dist": json.dumps({"fold": 0.3, "call": 0.3, "bet_50": 0.4}),
        }
    )
    dist = blend_distributions([hit], collection="postflop_decisions")
    assert dist["fold"] == pytest.approx(0.3)
    assert dist["call"] == pytest.approx(0.3)
    assert dist["bet_50"] == pytest.approx(0.4)


def test_action_dist_invalid_json_falls_back_to_label() -> None:
    hit = _hit({"hero_action_type": "call", "confidence": 1.0, "gto_score": 1.0, "action_dist": "not json {"})
    dist = blend_distributions([hit], collection="postflop_decisions")
    assert dist["call"] == pytest.approx(1.0)


def test_action_dist_all_noncanonical_keys_contributes_nothing() -> None:
    # A present-but-all-non-canonical action_dist is treated as authoritative (no
    # fall back to the label); the row contributes zero weight → NoStrategyError.
    from src._errors import NoStrategyError

    hit = _hit(
        {
            "hero_action_type": "fold",
            "confidence": 1.0,
            "gto_score": 1.0,
            "action_dist": json.dumps({"totally_made_up_action": 1.0}),
        }
    )
    with pytest.raises(NoStrategyError):
        blend_distributions([hit], collection="postflop_decisions")
