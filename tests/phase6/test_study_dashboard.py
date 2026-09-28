"""Unit tests for src/study/dashboard.py and src/study/probe.py.

All mocked — no live DB / Milvus needed.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

from src.study.dashboard import dashboard_snapshot
from src.study.probe import probe_by_cluster_key


def _mock_dashboard_conn(
    *,
    trend_rows=(),
    activity_rows=(),
    health_select_ok=True,
    session_row=None,
    by_source_rows=(),
    delta_rows=(),
    recent_mean=None,
    prior_mean=None,
):
    """Mock psycopg conn that dispatches by SQL content for all 6 dashboard sections."""
    conn = MagicMock()

    def cursor_factory():
        cm = MagicMock()
        cm.__enter__ = lambda self: cm
        cm.__exit__ = lambda *a: False
        cm.description = []
        cm.fetchall.return_value = []
        cm.fetchone.return_value = None

        def execute(sql, params=None):
            s = " ".join(sql.split()).lower()
            if "select 1" == s.rstrip(";"):
                cm.fetchone.return_value = (1,) if health_select_ok else None
            elif "metric_name = 'ev_loss'" in s and "group by session_id" in s:
                # ev_loss trend
                cm.fetchall.return_value = list(trend_rows)
            elif "metric_name = 'ev_loss'" in s and "in (select distinct" in s:
                # recent_mean for loop_health
                cm.fetchone.return_value = (recent_mean,)
            elif "metric_name = 'ev_loss'" in s and "ts <" in s:
                # prior_mean for loop_health
                cm.fetchone.return_value = (prior_mean,)
            elif "from patches" in s and "union all" in s:
                # recent_activity union
                cm.fetchall.return_value = list(activity_rows)
            elif "from observations" in s and "group by session_id" in s:
                # session in-flight row
                cm.fetchone.return_value = session_row
            elif "from strategy_nodes" in s and "group by source" in s and "created_at" in s:
                # last_7d_delta
                cm.fetchall.return_value = list(delta_rows)
            elif "from strategy_nodes" in s and "group by source" in s:
                # by_source
                cm.fetchall.return_value = list(by_source_rows)
            return None

        cm.execute.side_effect = execute
        return cm

    conn.cursor.side_effect = cursor_factory
    return conn


def test_dashboard_snapshot_has_required_sections():
    conn = _mock_dashboard_conn(
        trend_rows=[("sess-1", dt.datetime(2026, 5, 19), 0.42)],
        activity_rows=[(dt.datetime(2026, 5, 19), "patch", "patch xyz on ck-1 source=manual")],
        session_row=("sess-1", 100, dt.datetime(2026, 5, 19), dt.datetime(2026, 5, 19)),
        by_source_rows=[("autoloop", 50), ("manual", 12)],
        delta_rows=[("autoloop", 5)],
        recent_mean=0.40,
        prior_mean=0.50,
    )
    snap = dashboard_snapshot(_tsdb_conn=conn)
    required = {"loop_health", "ev_loss_trend", "recent_activity", "health", "session", "kb_growth"}
    assert set(snap.keys()) >= required


def test_loop_health_verdict_improving():
    conn = _mock_dashboard_conn(recent_mean=0.30, prior_mean=0.50)
    snap = dashboard_snapshot(_tsdb_conn=conn)
    assert snap["loop_health"]["verdict"] == "improving"
    assert isinstance(snap["loop_health"]["narrative"], str)


def test_loop_health_verdict_stalled():
    conn = _mock_dashboard_conn(recent_mean=0.60, prior_mean=0.40)
    snap = dashboard_snapshot(_tsdb_conn=conn)
    assert snap["loop_health"]["verdict"] == "stalled"


def test_loop_health_verdict_mixed_when_insufficient_history():
    conn = _mock_dashboard_conn(recent_mean=None, prior_mean=None)
    snap = dashboard_snapshot(_tsdb_conn=conn)
    assert snap["loop_health"]["verdict"] == "mixed"


def test_ev_loss_trend_length_capped_at_30():
    rows = [(f"sess-{i}", dt.datetime(2026, 5, 19), 0.5) for i in range(30)]
    conn = _mock_dashboard_conn(trend_rows=rows)
    snap = dashboard_snapshot(_tsdb_conn=conn)
    assert len(snap["ev_loss_trend"]) <= 30
    # Each entry has expected keys
    if snap["ev_loss_trend"]:
        assert set(snap["ev_loss_trend"][0].keys()) >= {"session_id", "ts", "ev_loss"}


def test_recent_activity_capped_at_limit():
    rows = [(dt.datetime(2026, 5, 19), "patch", f"sum-{i}") for i in range(15)]
    conn = _mock_dashboard_conn(activity_rows=rows[:10])  # mock returns first 10
    snap = dashboard_snapshot(_tsdb_conn=conn)
    assert len(snap["recent_activity"]) <= 10
    if snap["recent_activity"]:
        assert set(snap["recent_activity"][0].keys()) >= {"ts", "event_type", "summary"}


def test_kb_growth_section_shape():
    conn = _mock_dashboard_conn(
        by_source_rows=[("autoloop", 50), ("manual", 12), ("seed", 100)],
        delta_rows=[("autoloop", 5), ("manual", 2)],
    )
    snap = dashboard_snapshot(_tsdb_conn=conn)
    kb = snap["kb_growth"]
    assert "total_nodes" in kb
    assert "by_source" in kb
    assert "last_7d_delta" in kb
    assert kb["total_nodes"] == 162  # 50 + 12 + 100
    assert kb["by_source"]["autoloop"] == 50
    assert kb["last_7d_delta"]["autoloop"] == 5


def test_health_section_db_healthy_when_select_succeeds():
    conn = _mock_dashboard_conn(health_select_ok=True)
    snap = dashboard_snapshot(_tsdb_conn=conn)
    assert snap["health"]["db"] == "healthy"


# ---------------------- probe tests ----------------------


def _hit(cluster_key, distance):
    hit = MagicMock()
    hit.distance = distance
    fields = {"cluster_key": cluster_key}
    entity = MagicMock()
    entity.get = lambda k, default=None: fields.get(k, default)
    hit.entity = entity
    return hit


def _mock_probe_conn(
    *,
    engine_row=None,
    n_obs=42,
    sparse_row=(False, None),
    ev_mean=0.25,
    neighbor_action_dist=None,
    solver_row=None,
    neighbor_embedding=None,
):
    """Mock TSDB conn for probe_by_cluster_key."""
    conn = MagicMock()
    state = {"i": 0}  # tracks which query we're on

    def cursor_factory():
        cm = MagicMock()
        cm.__enter__ = lambda self: cm
        cm.__exit__ = lambda *a: False
        cm.description = []
        cm.fetchall.return_value = []
        cm.fetchone.return_value = None

        def execute(sql, params=None):
            s = " ".join(sql.split()).lower()
            # Check solver-truth FIRST (it also matches "select action_dist
            # from strategy_nodes ... active = true").
            if "source in ('solver','solver_verify')" in s:
                cm.fetchone.return_value = solver_row
            elif "select source, action_dist from strategy_nodes" in s:
                # engine_response source+action_dist
                cm.fetchone.return_value = engine_row
            elif "count(*) from observations" in s:
                cm.fetchone.return_value = (n_obs,)
            elif "bool_or(flagged_sparse)" in s:
                cm.fetchone.return_value = sparse_row
            elif "avg(value) from metrics" in s and "ev_loss" in s:
                cm.fetchone.return_value = (ev_mean,)
            elif "select embedding from strategy_nodes" in s:
                cm.fetchone.return_value = (neighbor_embedding,) if neighbor_embedding is not None else None
            elif "select action_dist from strategy_nodes" in s and "active = true" in s:
                # used by knn neighbor lookup loop
                cm.fetchone.return_value = (neighbor_action_dist,) if neighbor_action_dist else None
            state["i"] += 1
            return None

        cm.execute.side_effect = execute
        return cm

    conn.cursor.side_effect = cursor_factory
    return conn


def test_probe_by_cluster_key_returns_three_sections():
    engine_row = ("autoloop", {"fold": 0.3, "call": 0.4, "bet_50": 0.3})
    conn = _mock_probe_conn(
        engine_row=engine_row,
        neighbor_embedding=[0.1] * 80,
    )
    client = MagicMock()
    client.search.return_value = [[_hit("street_class=flop|pot_type=srp|n1", 0.05)]]
    result = probe_by_cluster_key("street_class=flop|pot_type=srp|x", k=5, _tsdb_conn=conn, _milvus=client)
    assert set(result.keys()) >= {"engine_response", "knn_neighbors", "solver_truth"}


def test_probe_engine_response_shape():
    engine_row = ("manual", {"call": 1.0})
    conn = _mock_probe_conn(engine_row=engine_row, n_obs=42, sparse_row=(True, 0.9), ev_mean=0.18)
    client = MagicMock()
    client.search.return_value = [[]]
    result = probe_by_cluster_key("street_class=flop|pot_type=srp|x", _tsdb_conn=conn, _milvus=client)
    er = result["engine_response"]
    expected = {
        "source",
        "action_dist",
        "n_obs",
        "ev_loss",
        "ci_low",
        "ci_high",
        "flagged_sparse",
        "max_neighbor_dist",
    }
    assert set(er.keys()) >= expected
    assert er["source"] == "manual"
    assert er["n_obs"] == 42
    assert er["flagged_sparse"] is True


def test_probe_solver_truth_none_when_no_solver_row():
    engine_row = ("autoloop", {"fold": 0.5, "call": 0.5})
    conn = _mock_probe_conn(engine_row=engine_row, solver_row=None, neighbor_embedding=[0.1] * 80)
    client = MagicMock()
    client.search.return_value = [[]]
    result = probe_by_cluster_key("street_class=flop|pot_type=srp|x", _tsdb_conn=conn, _milvus=client)
    assert result["solver_truth"] is None


def test_probe_solver_truth_dict_when_solver_row_exists():
    engine_row = ("autoloop", {"fold": 0.5, "call": 0.5})
    solver_dist = {"fold": 0.4, "call": 0.6}
    conn = _mock_probe_conn(
        engine_row=engine_row,
        solver_row=(solver_dist,),
        neighbor_embedding=[0.1] * 80,
    )
    client = MagicMock()
    client.search.return_value = [[]]
    result = probe_by_cluster_key("street_class=flop|pot_type=srp|x", _tsdb_conn=conn, _milvus=client)
    assert result["solver_truth"] is not None
    assert result["solver_truth"]["action_dist"] == solver_dist
    assert "kl_actual_vs_solver" in result["solver_truth"]


def test_probe_engine_response_when_cluster_missing():
    """When strategy_nodes has no row, engine_response returns the empty sentinel shape."""
    conn = _mock_probe_conn(engine_row=None)
    client = MagicMock()
    client.search.return_value = [[]]
    result = probe_by_cluster_key("nonexistent_ck", _tsdb_conn=conn, _milvus=client)
    er = result["engine_response"]
    assert er["source"] is None
    assert er["n_obs"] == 0
    assert er["action_dist"] is None


def test_probe_knn_neighbors_skip_self_and_cap_at_k():
    engine_row = ("autoloop", {"fold": 0.5, "call": 0.5})
    conn = _mock_probe_conn(
        engine_row=engine_row,
        neighbor_embedding=[0.1] * 80,
        neighbor_action_dist={"call": 1.0},
    )
    source_ck = "street_class=flop|pot_type=srp|self"
    client = MagicMock()
    client.search.return_value = [
        [
            _hit(source_ck, 0.0),
            _hit("street_class=flop|pot_type=srp|a", 0.05),
            _hit("street_class=flop|pot_type=srp|b", 0.10),
            _hit("street_class=flop|pot_type=srp|c", 0.15),
        ]
    ]
    result = probe_by_cluster_key(source_ck, k=2, _tsdb_conn=conn, _milvus=client)
    assert all(n["cluster_key"] != source_ck for n in result["knn_neighbors"])
    assert len(result["knn_neighbors"]) <= 2
