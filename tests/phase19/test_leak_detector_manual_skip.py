from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from src._config import AutoLoopConfig
from src.autoloop.leak_detector import select_patch_candidates


def _make_conn_with_active_row(cluster_key: str, source: str):
    """Build a mock connection returning one ev_loss row + obs count + active node."""
    mock_conn = MagicMock()
    cursor_ctx = MagicMock()

    node_id = str(uuid.uuid4())
    ev_loss_rows = [(cluster_key, 0.5)]
    obs_count_row = (25,)
    active_node_row = (node_id, source, False, False)

    cursor_ctx.__enter__.return_value.fetchall.return_value = ev_loss_rows
    cursor_ctx.__enter__.return_value.fetchone.side_effect = [
        obs_count_row,
        active_node_row,
    ]
    mock_conn.cursor.return_value = cursor_ctx
    return mock_conn


def test_manual_source_skipped():
    """Clusters whose active node has source='manual' must not appear in candidates."""
    config = AutoLoopConfig(
        tau_leak=0.1,
        min_observations=10,
        max_patches_per_run=10,
    )
    cluster_key = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
    mock_conn = _make_conn_with_active_row(cluster_key, source="manual")

    candidates = select_patch_candidates("session_test", config, _tsdb_conn=mock_conn)

    candidate_keys = [c.cluster_key for c in candidates]
    assert cluster_key not in candidate_keys
