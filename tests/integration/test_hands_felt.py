"""get_replay felt round-trip with a mock connection (no live TSDB).

list_hands moved to a hand-level hands_index contract (see test_list_hands_index.py);
get_replay is unchanged (observations-backed, felt per dp) and covered here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.study.hands import get_replay


def _make_conn(rows, col_names):
    """Build a minimal psycopg-like mock connection returning the given rows."""
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.description = [(name,) for name in col_names]
    cursor.fetchall.return_value = rows

    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


_REPLAY_COLS = [
    "obs_id",
    "session_id",
    "ts",
    "cluster_key",
    "action_taken",
    "source",
    "solver_label",
    "max_neighbor_distance",
    "flagged_sparse",
    "embedding",
    "hand_id",
    "decision_id",
    "felt_snapshot",
]


def test_get_replay_includes_felt_snapshot_for_sim_row() -> None:
    """get_replay result dicts must include felt_snapshot for a sim row."""
    felt = {
        "street": "flop",
        "hero_position": "BTN",
        "hero_hole_cards": ["Ah", "Kd"],
        "board_cards": ["2h", "7c", "Qd"],
        "pot_size_bb": 8.0,
        "effective_stack_bb": 95.0,
    }
    rows = [
        (
            "obs-1",
            "sess-abc",
            "2026-01-01T00:00:00",
            "ck=A",
            "call",
            "sim",
            None,
            0.12,
            False,
            [0.1] * 128,
            "sess-abc_h0",
            "sess-abc_h0_dp0",
            felt,
        )
    ]
    conn = _make_conn(rows, _REPLAY_COLS)

    result = get_replay("obs-1", _tsdb_conn=conn)
    assert len(result) == 1
    row = result[0]
    assert row["felt_snapshot"]["street"] == "flop"
    assert row["hand_id"] == "sess-abc_h0"
    assert row["decision_id"] == "sess-abc_h0_dp0"


def test_get_replay_null_felt_returns_none_without_crash() -> None:
    """get_replay for a NULL-felt row returns None without crashing (D-08)."""
    rows = [
        (
            "obs-hm",
            "sess-hm",
            "2026-01-01T00:00:00",
            "ck=B",
            "fold",
            "hm",
            None,
            None,
            None,
            [0.0] * 128,
            None,
            None,
            None,
        )
    ]
    conn = _make_conn(rows, _REPLAY_COLS)

    result = get_replay("obs-hm", _tsdb_conn=conn)
    assert len(result) == 1
    assert result[0]["felt_snapshot"] is None
    assert result[0]["hand_id"] is None
