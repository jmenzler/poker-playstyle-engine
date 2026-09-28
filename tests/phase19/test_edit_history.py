"""Unit tests for get_edit_history under the DP-level coexist model.

Keyed off decision_id (not cluster_key); status reflects patches.status since
every strategy_node stays active=TRUE under coexist.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

from src.study.gaps import get_edit_history


def _mock_conn(rows, captured):
    conn = MagicMock()

    def cursor_factory():
        cm = MagicMock()
        cm.__enter__ = lambda self: cm
        cm.__exit__ = lambda *a: False

        def execute(sql, params=None):
            captured["sql"] = " ".join(sql.split())
            captured["params"] = params

        cm.execute.side_effect = execute
        cm.fetchall.return_value = list(rows)
        return cm

    conn.cursor.side_effect = cursor_factory
    return conn


def test_get_edit_history_keys_off_decision_id():
    captured: dict = {}
    ts = datetime.datetime(2026, 6, 24, 10, 0, 0)
    rows = [(ts, {"call": 0.5, "fold": 0.5}, "applied")]
    out = get_edit_history("hand1_dp3", _tsdb_conn=_mock_conn(rows, captured))

    assert "p.decision_id = %s" in captured["sql"]
    assert "p.cluster_key = %s" not in captured["sql"]
    assert captured["params"] == ("hand1_dp3",)
    assert out == [{"ts": ts.isoformat(), "action_dist": {"call": 0.5, "fold": 0.5}, "status": "applied"}]


def test_get_edit_history_status_from_patches_not_node_active():
    captured: dict = {}
    ts = datetime.datetime(2026, 6, 24, 10, 0, 0)
    rows = [(ts, {"bet_50": 1.0}, "rolled_back")]
    out = get_edit_history("hand1_dp3", _tsdb_conn=_mock_conn(rows, captured))

    assert "sn.active" not in captured["sql"]
    assert out[0]["status"] == "rolled_back"
