"""Tests for kNN neighbor obs_id + hand_id read-only extension."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import numpy as np

import src.study.probe as probe_mod


def _build_mock_conn_for_neighbor(obs_row, n_obs: int) -> MagicMock:
    """Mock conn for the full _knn_neighbors cursor sequence (one neighbor)."""
    results = [
        None,
        ("fold_or_call", {"fold": 0.5, "call": 0.5}),
        obs_row,
        (n_obs,),
    ]
    call_idx = [0]

    def cursor_side_effect():
        cm = MagicMock()
        idx = call_idx[0]
        call_idx[0] += 1
        val = results[idx] if idx < len(results) else None
        inner = MagicMock()
        inner.fetchone.return_value = val
        cm.__enter__ = MagicMock(return_value=inner)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    mock_conn = MagicMock()
    mock_conn.cursor.side_effect = cursor_side_effect
    return mock_conn


def _make_hit(cluster_key: str, distance: float = 0.3, decision_id: str | None = None) -> MagicMock:
    hit = MagicMock()
    data = {"cluster_key": cluster_key, "decision_id": decision_id, "hero_action_type": "bet"}
    hit.entity.get.side_effect = lambda k, d=None: data.get(k, d)
    hit.distance = distance
    return hit


_QUERY_KEY = "street_class=preflop|pot_type=srp|hero_pos_rel=IP|n_players_active=2"
_NEIGHBOR_KEY = "street_class=preflop|pot_type=srp|hero_pos_rel=OOP|n_players_active=2"

_SPOT = {
    "street": "preflop",
    "hero_hole": ["Ah", "Kh"],
    "hero_position": "BB",
    "action_sequence": ["BTN:open_2.5", "SB:fold"],
}


def test_knn_neighbor_carries_obs_id_and_hand_id():
    """Each neighbor dict gains obs_id, hand_id, n_obs from a read-only observations lookup."""
    obs_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    mock_conn = _build_mock_conn_for_neighbor(obs_row=(obs_uuid, "sess_h3"), n_obs=7)

    mock_client = MagicMock()
    mock_client.search.return_value = [[_make_hit(_NEIGHBOR_KEY)]]

    fake_embedding = np.zeros(34, dtype=np.float64)
    with patch.object(probe_mod, "_encode_embedding_for_spot", return_value=fake_embedding):
        result = probe_mod._knn_neighbors(mock_conn, mock_client, _QUERY_KEY, spot=_SPOT, k=5)

    assert len(result) == 1
    neighbor = result[0]
    assert neighbor["cluster_key"] == _NEIGHBOR_KEY
    assert "distance" in neighbor
    assert "action_dist" in neighbor
    assert neighbor["obs_id"] == str(obs_uuid)
    assert neighbor["hand_id"] == "sess_h3"
    assert neighbor["n_obs"] == 7


def test_knn_neighbor_hand_id_from_decision_id_when_no_observation():
    """No observation row → obs_id is None but hand_id falls back to the corpus hit's
    decision_id (so the neighbor is still replayable via the HM3 by-hand path)."""
    mock_conn = _build_mock_conn_for_neighbor(obs_row=None, n_obs=0)

    mock_client = MagicMock()
    mock_client.search.return_value = [[_make_hit(_NEIGHBOR_KEY, decision_id="700000000001_dp2")]]

    fake_embedding = np.zeros(34, dtype=np.float64)
    with patch.object(probe_mod, "_encode_embedding_for_spot", return_value=fake_embedding):
        result = probe_mod._knn_neighbors(mock_conn, mock_client, _QUERY_KEY, spot=_SPOT, k=5)

    assert len(result) == 1
    neighbor = result[0]
    assert neighbor["obs_id"] is None
    assert neighbor["hand_id"] == "700000000001"
    assert neighbor["decision_id"] == "700000000001_dp2"
    assert neighbor["cluster_key"] == _NEIGHBOR_KEY
