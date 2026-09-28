"""Tests for queue_driver injection path — two-write order, per-DP Milvus write."""  # rot-allow

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_obs_row(decision_id: str = "dp-inject-001", embedding: list | None = None) -> dict:
    return {
        "obs_id": "obs-inject-001",
        "decision_id": decision_id,
        "cluster_key": "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        "embedding": embedding or [0.1] * 80,
        "max_neighbor_distance": 0.75,
        "felt_snapshot": {
            "action_sequence": ["BB:check"],
            "board_cards": ["Jh", "7c", "2d"],
            "effective_stack_bb": 95.0,
            "hero_bet_size_bb": 0.0,
            "hero_facing_bet_bb": 0.0,
            "hero_hole_cards": ["Ac", "Kh"],
            "hero_position": "BTN",
            "opponents_remaining": 1,
            "pot_size_bb": 8.0,
            "prior_street_aggressor": "BTN",
            "street": "flop",
        },
        "cluster_freq": 25,
        "ev_loss": 0.6,
        "priority_score": 25 * 0.6 * 0.75,
    }


def test_inject_calls_milvus_not_patch_engine():
    """_inject writes directly to Milvus — NOT through PatchEngine (solver distillation path)."""
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_row = _make_obs_row()
    spot = SolverSpot(
        pot=800,
        effective_stack=9500,
        board=["Jh", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    mock_result = MagicMock()
    mock_result.action_dist = {"bet_33": 0.6, "check": 0.4}
    mock_result.exploitability_pct = 0.45

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor

    mock_milvus = MagicMock()
    mock_patch_engine = MagicMock()

    with patch("src.solver.queue_driver.persist_solve"):
        _inject(
            obs_row,
            spot,
            mock_result,
            conn=mock_conn,
            milvus=mock_milvus,
            palette_lookup={},
            _patch_engine=mock_patch_engine,
        )

    mock_milvus.upsert.assert_called_once()
    mock_patch_engine.apply.assert_not_called()


def test_gto_score_formula():
    """gto_score = max(0, 1 - exploitability_pct/100) is in the Milvus node row."""
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_row = _make_obs_row()
    spot = SolverSpot(
        pot=800,
        effective_stack=9500,
        board=["Jh", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    mock_result = MagicMock()
    mock_result.action_dist = {"check": 1.0}
    mock_result.exploitability_pct = 25.0

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor

    mock_milvus = MagicMock()

    with patch("src.solver.queue_driver.persist_solve"):
        _inject(
            obs_row,
            spot,
            mock_result,
            conn=mock_conn,
            milvus=mock_milvus,
            palette_lookup={},
        )

    mock_milvus.upsert.assert_called_once()
    row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert row["gto_score"] == pytest.approx(0.75)
    assert row["confidence"] == pytest.approx(1.0)
    assert row["active"] is True


def _inject_and_get_row(obs_row: dict) -> dict:
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    spot = SolverSpot(pot=800, effective_stack=9500, board=["Jh", "7c", "2d"], range_ip="AA", range_oop="AA")
    mock_result = MagicMock()
    mock_result.action_dist = {"check": 1.0}
    mock_result.exploitability_pct = 0.45

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    mock_milvus = MagicMock()

    with patch("src.solver.queue_driver.persist_solve"):
        _inject(obs_row, spot, mock_result, conn=mock_conn, milvus=mock_milvus, palette_lookup={})
    return mock_milvus.upsert.call_args.kwargs["data"][0] if mock_milvus.upsert.called else None


def test_injected_embedding_is_zscore_weighted_not_raw():
    """The Milvus vector must be normalize((raw-mean)/std)*weights — the SAME space the
    bulk corpus loader and KNNDecisionEngine query use. A raw obs embedding lands in the
    wrong space and is unretrievable; that regression is what this guards."""
    import numpy as np

    from src.solver.queue_driver import _load_zscore_transform, normalize

    raw = [0.1] * 80
    row = _inject_and_get_row(_make_obs_row(embedding=raw))

    mean, std, weights = _load_zscore_transform("postflop_decisions")
    expected = normalize(np.asarray(raw, dtype=np.float64), mean, std, weights).tolist()

    assert row["embedding"] == pytest.approx(expected)
    assert row["embedding"] != pytest.approx(raw)  # not the raw vector


def test_inject_skips_milvus_when_embedding_absent():
    """No/empty embedding -> skip the Milvus write (solve still persisted), never store
    a malformed vector."""
    obs_row = _make_obs_row()
    obs_row["embedding"] = []
    row = _inject_and_get_row(obs_row)
    assert row is None
