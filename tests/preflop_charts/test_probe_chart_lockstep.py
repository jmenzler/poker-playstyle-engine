"""Probe lockstep: the preflop probe path mirrors the live chart lookup."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.study.probe import _preflop_chart_response


def _spot(**over) -> dict:
    base = {
        "street": "preflop",
        "hero_position": "BTN",
        "hero_hole": ("Ah", "Kd"),
        "board": (),
        "action_sequence": ("UTG:fold", "MP:fold", "CO:fold"),
        "opponents_remaining": 3,
    }
    base.update(over)
    return base


def test_chart_response_for_covered_rfi_spot():
    resp = _preflop_chart_response(_spot())
    assert resp is not None
    assert resp["source"] == "preflop_chart"
    assert resp["action_dist"] is not None
    # BTN AKo opens
    assert any(a.startswith("open") for a in resp["action_dist"])
    assert resp["flagged_sparse"] is False
    assert resp["max_neighbor_dist"] == 0.0


def test_chart_response_none_for_uncovered():
    # 72o UTG -> pure fold, absent from chart -> None (probe falls back to kNN blend)
    resp = _preflop_chart_response(_spot(hero_position="UTG", hero_hole=("7c", "2d"), action_sequence=()))
    assert resp is None


def test_chart_response_none_for_postflop():
    resp = _preflop_chart_response(_spot(street="flop", board=("Qh", "Jh", "2s")))
    assert resp is None


def test_probe_by_cluster_key_preflop_prefers_chart(monkeypatch):
    """probe_by_cluster_key tags engine_response source=preflop_chart on a covered preflop spot."""
    import src.study.probe as probe

    # No strategy_nodes row -> empty engine_response; force the no-node path.
    monkeypatch.setattr(probe, "_engine_response", lambda conn, ck: dict(probe._EMPTY_ENGINE_RESPONSE))
    monkeypatch.setattr(
        probe,
        "_knn_neighbors",
        lambda *a, **k: [{"distance": 0.1, "hero_action_type": "fold", "confidence": 1.0, "gto_score": 1.0}],
    )
    monkeypatch.setattr(probe, "_solver_truth", lambda *a, **k: None)

    cluster_key = "street_class=preflop|pot_type=limp|hero_pos_rel=OOP|n_players_active=6"
    result = probe.probe_by_cluster_key(
        cluster_key,
        spot=_spot(),
        _tsdb_conn=MagicMock(),
        _milvus=MagicMock(),
    )
    assert result["engine_response"]["source"] == "preflop_chart"
