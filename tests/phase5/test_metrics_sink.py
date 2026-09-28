"""Tests for src/metrics/sink.py flush_session_metrics() (METR-01..04).

Unit tests use mock_tsdb_conn and mock_milvus_client injection.
Integration test (test_metr_04_active_node_count_query_uses_index) is PC-gated.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Deterministic session start for ts-pinning per D-07-11c — required kwarg of
# flush_session_metrics. Phase 5 tests do not assert ts values, but the kwarg
# is mandatory (no default — that's the bug-fix invariant).
_TS_PIN = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_cursor(fetchall_rows=None, fetchone_row=None):
    """Build a mock cursor that tracks execute() calls."""
    cur = MagicMock()
    cur.fetchall.return_value = fetchall_rows or []
    cur.fetchone.return_value = fetchone_row
    return cur


def _make_tsdb_conn(fetchall_rows=None, fetchone_row=None):
    """Build a psycopg-shaped mock connection.

    Configures fetchall() for _TOUCHED_CLUSTERS_SQL and fetchone() for
    _NOVEL_SPOT_RATE_SQL. All cursor.execute() calls are tracked on the
    connection's execute_calls list via a recording cursor.
    """
    conn = MagicMock()
    execute_calls = []
    conn._execute_calls = execute_calls

    def make_cursor_cm():
        cur = MagicMock()
        cur.fetchall.return_value = fetchall_rows or []
        cur.fetchone.return_value = fetchone_row

        # Record all execute calls so tests can inspect them
        def record_execute(sql, params=None):
            execute_calls.append((sql, params))

        cur.execute.side_effect = record_execute
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cur)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    conn.cursor.side_effect = make_cursor_cm
    conn.commit.return_value = None
    return conn


def _make_milvus():
    """Build a mock Milvus client that returns a plausible kNN result."""
    client = MagicMock()
    client.search.return_value = [
        [
            {
                "distance": 0.1,
                "entity": {
                    "decision_id": "dp1",
                    "hero_action_type": "call",
                    "confidence": 1.0,
                    "gto_score": 1.0,
                },
            },
        ]
    ]
    return client


# ---------------------------------------------------------------------------
# Test 1 (METR-01): writes one ev_loss row per touched cluster
# ---------------------------------------------------------------------------


def test_writes_one_ev_loss_row_per_touched_cluster(monkeypatch):
    """METR-01: one INSERT with metric_name='ev_loss' per touched cluster_key."""
    from src.metrics.sink import flush_session_metrics

    conn = _make_tsdb_conn(
        fetchall_rows=[("ck1",), ("ck2",)],  # _TOUCHED_CLUSTERS_SQL
        fetchone_row=(0.05,),  # _NOVEL_SPOT_RATE_SQL
    )

    # Monkeypatch ev_loss so it doesn't hit DB
    monkeypatch.setattr("src.metrics.sink.ev_loss", lambda ck, session_id=None, **kw: 0.42)

    flush_session_metrics(
        "session_abc",
        [0.001, 0.002, 0.003],
        session_started_at=_TS_PIN,
        _tsdb_conn=conn,
        _milvus=_make_milvus(),
    )

    # Find ev_loss INSERT calls
    ev_loss_inserts = [
        (sql, params)
        for sql, params in conn._execute_calls
        if params and len(params) >= 5 and params[3] == "ev_loss"
    ]
    assert len(ev_loss_inserts) == 2, (
        f"Expected 2 ev_loss INSERT calls (one per cluster), got {len(ev_loss_inserts)}"
    )
    cluster_keys_inserted = {p[2] for _, p in ev_loss_inserts}
    assert "ck1" in cluster_keys_inserted
    assert "ck2" in cluster_keys_inserted


# ---------------------------------------------------------------------------
# Test 2 (METR-02): writes novel_spot_rate with cluster_key=NULL
# ---------------------------------------------------------------------------


def test_writes_novel_spot_rate_with_null_cluster_key(monkeypatch):
    """METR-02: one INSERT with metric_name='novel_spot_rate', cluster_key=None."""
    from src.metrics.sink import flush_session_metrics

    conn = _make_tsdb_conn(
        fetchall_rows=[],  # no touched clusters
        fetchone_row=(0.15,),  # novel_spot_rate
    )
    monkeypatch.setattr("src.metrics.sink.ev_loss", lambda ck, session_id=None, **kw: 0.0)

    flush_session_metrics(
        "sess1",
        [],  # empty timing buffer
        session_started_at=_TS_PIN,
        _tsdb_conn=conn,
        _milvus=_make_milvus(),
    )

    novel_inserts = [
        (sql, params)
        for sql, params in conn._execute_calls
        if params and len(params) >= 5 and params[3] == "novel_spot_rate"
    ]
    assert len(novel_inserts) == 1, f"Expected 1 novel_spot_rate INSERT, got {len(novel_inserts)}"
    _, p = novel_inserts[0]
    assert p[2] is None, f"cluster_key for novel_spot_rate must be None, got {p[2]}"
    assert abs(float(p[4]) - 0.15) < 1e-6, f"novel_spot_rate value mismatch: {p[4]}"


# ---------------------------------------------------------------------------
# Test 3 (METR-03): writes latency_p50 and latency_p99 with cluster_key=NULL
# ---------------------------------------------------------------------------


def test_writes_latency_p50_p99_when_timing_buffer_present(monkeypatch):
    """METR-03: two INSERTs for latency_p50 and latency_p99 with cluster_key=None."""
    from src.metrics.sink import flush_session_metrics

    conn = _make_tsdb_conn(
        fetchall_rows=[],  # no touched clusters
        fetchone_row=(0.0,),  # novel_spot_rate
    )
    monkeypatch.setattr("src.metrics.sink.ev_loss", lambda ck, session_id=None, **kw: 0.0)

    timing = [0.001, 0.002, 0.003, 0.004, 0.005]
    flush_session_metrics(
        "sess2",
        timing,
        session_started_at=_TS_PIN,
        _tsdb_conn=conn,
        _milvus=_make_milvus(),
    )

    p50_inserts = [p for _, p in conn._execute_calls if p and len(p) >= 5 and p[3] == "latency_p50"]
    p99_inserts = [p for _, p in conn._execute_calls if p and len(p) >= 5 and p[3] == "latency_p99"]

    assert len(p50_inserts) == 1, f"Expected 1 latency_p50 INSERT, got {len(p50_inserts)}"
    assert len(p99_inserts) == 1, f"Expected 1 latency_p99 INSERT, got {len(p99_inserts)}"

    p50_val = float(p50_inserts[0][4])
    p99_val = float(p99_inserts[0][4])
    expected_p50 = float(np.percentile(timing, 50))
    expected_p99 = float(np.percentile(timing, 99))

    assert abs(p50_val - expected_p50) < 1e-9, f"latency_p50: {p50_val} != {expected_p50}"
    assert abs(p99_val - expected_p99) < 1e-9, f"latency_p99: {p99_val} != {expected_p99}"

    # Both must have cluster_key=None
    assert p50_inserts[0][2] is None, "latency_p50 cluster_key must be None"
    assert p99_inserts[0][2] is None, "latency_p99 cluster_key must be None"


# ---------------------------------------------------------------------------
# Test 4: skips latency rows when timing_buffer is empty
# ---------------------------------------------------------------------------


def test_skips_latency_when_timing_buffer_empty(monkeypatch, caplog):
    """Empty timing_buffer: no latency_p50/p99 INSERTs; warning logged."""
    import logging

    from src.metrics.sink import flush_session_metrics

    conn = _make_tsdb_conn(
        fetchall_rows=[],
        fetchone_row=(0.0,),
    )
    monkeypatch.setattr("src.metrics.sink.ev_loss", lambda ck, session_id=None, **kw: 0.0)

    with caplog.at_level(logging.WARNING):
        flush_session_metrics(
            "sess3",
            [],  # empty buffer
            session_started_at=_TS_PIN,
            _tsdb_conn=conn,
            _milvus=_make_milvus(),
        )

    latency_inserts = [
        p for _, p in conn._execute_calls if p and len(p) >= 5 and p[3] in ("latency_p50", "latency_p99")
    ]
    assert len(latency_inserts) == 0, (
        f"Expected 0 latency INSERTs for empty buffer, got {len(latency_inserts)}"
    )


# ---------------------------------------------------------------------------
# Test 5: NoStrategyError for a cluster skips that ev_loss row
# ---------------------------------------------------------------------------


def test_handles_no_strategy_error_for_low_observation_cluster(monkeypatch):
    """METR-01 fault tolerance: NoStrategyError for one cluster skips its row."""
    from src._errors import NoStrategyError
    from src.metrics.sink import flush_session_metrics

    conn = _make_tsdb_conn(
        fetchall_rows=[("ck_ok",), ("ck_fail",)],
        fetchone_row=(0.05,),
    )

    def mock_ev_loss(ck, session_id=None, **kw):
        if ck == "ck_fail":
            raise NoStrategyError("too few obs for ck_fail")
        return 0.42

    monkeypatch.setattr("src.metrics.sink.ev_loss", mock_ev_loss)

    n = flush_session_metrics(
        "sess4",
        [0.001, 0.002],
        session_started_at=_TS_PIN,
        _tsdb_conn=conn,
        _milvus=_make_milvus(),
    )

    ev_loss_inserts = [p for _, p in conn._execute_calls if p and len(p) >= 5 and p[3] == "ev_loss"]
    assert len(ev_loss_inserts) == 1, f"Expected 1 ev_loss INSERT (ck_ok only), got {len(ev_loss_inserts)}"
    assert ev_loss_inserts[0][2] == "ck_ok"

    # total_rows = 1 ev_loss + 1 novel_spot_rate + 2 latency = 4
    # Since ck_fail was skipped, we get: 1 ev_loss + 1 novel_spot_rate + 2 latency
    assert n == 4, f"Expected 4 rows written, got {n}"


# ---------------------------------------------------------------------------
# Test 6: novel_rate=0.0 when session has zero observations (NULLIF → NULL)
# ---------------------------------------------------------------------------


def test_novel_rate_zero_when_no_observations_in_session(monkeypatch):
    """NULLIF(COUNT(*), 0) returns NULL → Python converts to 0.0 (no div-by-zero)."""
    from src.metrics.sink import flush_session_metrics

    conn = _make_tsdb_conn(
        fetchall_rows=[],
        fetchone_row=(None,),  # NULLIF returns NULL → row[0] is None
    )
    monkeypatch.setattr("src.metrics.sink.ev_loss", lambda ck, session_id=None, **kw: 0.0)

    flush_session_metrics(
        "sess5",
        [],
        session_started_at=_TS_PIN,
        _tsdb_conn=conn,
        _milvus=_make_milvus(),
    )

    novel_inserts = [p for _, p in conn._execute_calls if p and len(p) >= 5 and p[3] == "novel_spot_rate"]
    assert len(novel_inserts) == 1
    assert float(novel_inserts[0][4]) == 0.0, f"NULL novel_rate should become 0.0, got {novel_inserts[0][4]}"


# ---------------------------------------------------------------------------
# Test 7: returns correct rows_written count
# ---------------------------------------------------------------------------


def test_returns_correct_rows_written_count(monkeypatch):
    """rows_written = N_clusters + 1 (novel_rate) + 2 (latencies) when all succeed."""
    from src.metrics.sink import flush_session_metrics

    conn = _make_tsdb_conn(
        fetchall_rows=[("ck1",), ("ck2",), ("ck3",)],  # 3 clusters
        fetchone_row=(0.1,),
    )
    monkeypatch.setattr("src.metrics.sink.ev_loss", lambda ck, session_id=None, **kw: 0.5)

    n = flush_session_metrics(
        "sess6",
        [0.001, 0.002, 0.003, 0.004, 0.005],  # non-empty timing buffer
        session_started_at=_TS_PIN,
        _tsdb_conn=conn,
        _milvus=_make_milvus(),
    )

    # 3 ev_loss + 1 novel_spot_rate + 2 latency = 6
    assert n == 6, f"Expected 6 rows written, got {n}"


# ---------------------------------------------------------------------------
# Test 8 (METR-04, integration, PC): EXPLAIN confirms idx_strategy_nodes_source_active
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metr_04_active_node_count_query_uses_index(tsdb_dsn):
    """METR-04: EXPLAIN confirms idx_strategy_nodes_source_active is used.

    PC-gated (requires live TimescaleDB). Does NOT assert on execution time
    (PC load varies). Only asserts the planner uses the index — the structural
    prerequisite for the < 100ms guarantee on 50k nodes.
    """
    import json

    import psycopg

    sql = (
        "EXPLAIN (FORMAT JSON, ANALYZE) "
        "SELECT count(*) FROM strategy_nodes WHERE active = TRUE GROUP BY source"
    )

    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        plan = cur.fetchone()[0]

    plan_str = json.dumps(plan)
    assert "idx_strategy_nodes_source_active" in plan_str, (
        f"Expected idx_strategy_nodes_source_active in query plan, got:\n{plan_str}"
    )


# ---------------------------------------------------------------------------
# Re-export test: flush_session_metrics importable from src.metrics too
# ---------------------------------------------------------------------------


def test_flush_session_metrics_importable_from_src_metrics():
    """flush_session_metrics is re-exported from src.metrics package."""
    from src.metrics import flush_session_metrics as f1
    from src.metrics.sink import flush_session_metrics as f2

    assert f1 is f2, (
        "src.metrics.flush_session_metrics must be same object as src.metrics.sink.flush_session_metrics"
    )
