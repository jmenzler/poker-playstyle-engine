"""Phase 7 / WARN 2 / INTG-05 — decide-time scalar filter (street + spr_x100).

Closes ENGN-02 + STOR-05. Per D-07-11b: filter-only — hard_filter unchanged,
cluster_key format preserved (282k existing PC rows remain queryable).

Cf. .planning/v1.0-MILESTONE-AUDIT.md WARN 2 + 07-CONTEXT.md D-07-11b.
"""

import pytest


def test_build_filter_includes_street_and_spr_range():
    """ENGN-02: _build_filter adds street exact + spr_x100 BETWEEN when spot_features provided."""
    from src.decision_engine.engine import _build_filter

    hard_filter = {
        "hero_pos_rel": "IP",
        "n_players_active": 2,
        "pot_type": "srp",
        "street_class": "postflop",
    }
    spot_features = {"street": "flop", "spr_x100": 250}
    expr = _build_filter(hard_filter, include_active=True, spot_features=spot_features)
    assert 'street == "flop"' in expr
    assert "spr_x100 >= 125 and spr_x100 <= 375" in expr  # rot-allow
    assert "spr_x100" not in hard_filter
    assert "street" not in hard_filter


def test_spr_differentiates():
    """STOR-05: same hard_filter + different SPR -> different filter expressions."""
    from src.decision_engine.engine import _build_filter

    hard_filter = {
        "hero_pos_rel": "IP",
        "n_players_active": 2,
        "pot_type": "srp",
        "street_class": "postflop",
    }
    expr_lo = _build_filter(hard_filter, spot_features={"street": "flop", "spr_x100": 100})
    expr_hi = _build_filter(hard_filter, spot_features={"street": "flop", "spr_x100": 800})
    assert expr_lo != expr_hi


def test_preflop_sentinel_skips_spr():
    """Preflop sentinel: spr_x100 == -1 -> no spr filter predicate emitted."""
    from src.decision_engine.engine import _build_filter

    hard_filter = {
        "hero_pos_rel": "OOP",
        "n_players_active": 2,
        "pot_type": "srp",
        "street_class": "preflop",
    }
    expr = _build_filter(hard_filter, spot_features={"street": "preflop", "spr_x100": -1})
    assert "spr_x100" not in expr, f"Preflop sentinel must skip spr filter; got expression: {expr}"
    assert 'street == "preflop"' in expr


@pytest.mark.integration
def test_existing_rows_still_matched(milvus_uri, milvus_token):
    """ENGN-02 regression: existing 282k PC rows still queryable (cluster_key format unchanged per D-07-11b)."""
    from pymilvus import MilvusClient

    client = MilvusClient(uri=milvus_uri, token=milvus_token)
    res = client.query(
        collection_name="postflop_decisions",
        filter='street_class == "postflop"',
        output_fields=["cluster_key"],
        limit=1,
    )
    assert len(res) > 0, "No existing postflop rows matched — cluster_key format may have drifted"
