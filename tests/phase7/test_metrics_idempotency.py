"""Phase 7 / WARN 1 / INTG-04 — flush_session_metrics idempotency.

Closes METR-01..03 e2e (currently partial). Cf. .planning/v1.0-MILESTONE-AUDIT.md
WARN 1 + 07-CONTEXT.md D-07-11c (supersedes D-07-6 typo/ts-pin omission).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_cursor():
    cur = MagicMock()
    execute_calls = []

    def record_execute(sql, params=None):
        execute_calls.append((sql, params))

    cur.execute.side_effect = record_execute
    cur._execute_calls = execute_calls
    return cur


def _make_tsdb_conn(fetchall_rows=None, fetchone_row=None):
    """psycopg-shaped mock that records every execute() across cursor()s."""
    conn = MagicMock()
    execute_calls = []
    conn._execute_calls = execute_calls

    def make_cursor_cm():
        cur = MagicMock()
        cur.fetchall.return_value = fetchall_rows or []
        cur.fetchone.return_value = fetchone_row

        def record_execute(sql, params=None):
            execute_calls.append((sql, params))

        cur.execute.side_effect = record_execute
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cur)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    conn.cursor.side_effect = make_cursor_cm
    conn.transaction.return_value.__enter__ = MagicMock(return_value=conn)
    conn.transaction.return_value.__exit__ = MagicMock(return_value=False)
    conn.commit.return_value = None
    return conn


def _make_milvus():
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


def test_on_conflict_in_sql(monkeypatch, session_started_at):
    """WARN 1: every INSERT into metrics carries ON CONFLICT DO NOTHING + correct constraint columns."""
    from src.metrics.sink import flush_session_metrics

    conn = _make_tsdb_conn(fetchall_rows=[("ck1",)], fetchone_row=(0.05,))
    monkeypatch.setattr("src.metrics.sink.ev_loss", lambda *a, **kw: 0.42)

    flush_session_metrics(
        "s1",
        [0.001, 0.002, 0.003],
        session_started_at=session_started_at,
        _tsdb_conn=conn,
        _milvus=_make_milvus(),
    )

    insert_sqls = [sql for sql, _ in conn._execute_calls if "INSERT INTO metrics" in sql]
    assert insert_sqls, "no INSERT INTO metrics calls recorded"
    for sql in insert_sqls:
        assert "ON CONFLICT" in sql, f"INSERT missing ON CONFLICT: {sql}"
        assert "DO NOTHING" in sql, f"INSERT missing DO NOTHING: {sql}"
        assert "(session_id, metric_name, cluster_key, ts)" in sql, (
            f"INSERT missing exact constraint column list: {sql}"
        )


def test_uses_session_started_at(monkeypatch, session_started_at):
    """D-07-11c: every INSERT uses ts == session_started_at (NOT datetime.now())."""
    from src.metrics.sink import flush_session_metrics

    conn = _make_tsdb_conn(fetchall_rows=[("ck1",)], fetchone_row=(0.05,))
    monkeypatch.setattr("src.metrics.sink.ev_loss", lambda *a, **kw: 0.42)

    flush_session_metrics(
        "s1",
        [0.001],
        session_started_at=session_started_at,
        _tsdb_conn=conn,
        _milvus=_make_milvus(),
    )

    insert_params = [params for sql, params in conn._execute_calls if params and "INSERT INTO metrics" in sql]
    # ts is the LAST parameter (per _INSERT_METRIC_SQL column order)
    ts_values = [p[-1] for p in insert_params]
    assert ts_values, "no INSERT params captured"
    assert all(t == session_started_at for t in ts_values), (
        f"Expected all ts pinned to {session_started_at}; got {ts_values}"
    )


@pytest.mark.integration
def test_reflush_no_dup(tsdb_dsn, session_started_at):
    """METR-01..03 (D-07-11c): flush twice minutes apart -> second flush adds zero rows.

    With ts pinned to session_started_at, both flush invocations write rows with the
    SAME ts -> ON CONFLICT fires -> second flush is a no-op.
    """
    import time

    import psycopg

    from src.metrics.sink import flush_session_metrics

    session_id = f"phase7-reflush-{int(time.time())}"

    # Arrange: seed observations so flush has data to AVG over.
    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        for _ in range(20):
            cur.execute(
                "INSERT INTO observations (obs_id, cluster_key, embedding, action_taken, source, session_id, ts) "
                "VALUES (gen_random_uuid(), 'ck_reflush_test', ARRAY[0.0]::REAL[], 'check', 'sim', %s, %s)",
                (session_id, session_started_at),
            )
        conn.commit()

    try:
        # First flush.
        flush_session_metrics(
            session_id,
            [0.001, 0.002, 0.003],
            session_started_at=session_started_at,
        )

        with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM metrics WHERE session_id = %s", (session_id,))
            n_after_first = cur.fetchone()[0]
        assert n_after_first > 0, "first flush wrote nothing — fixture problem?"

        # Pause to ensure datetime.now() WOULD return a different value.
        time.sleep(2)

        # Second flush — same session_started_at, same session_id.
        flush_session_metrics(
            session_id,
            [0.001, 0.002, 0.003],
            session_started_at=session_started_at,
        )

        with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM metrics WHERE session_id = %s", (session_id,))
            n_after_second = cur.fetchone()[0]

        assert n_after_second == n_after_first, (
            f"Re-flush wrote {n_after_second - n_after_first} duplicate rows — "
            "ON CONFLICT DO NOTHING + ts pinning broken"
        )
    finally:
        with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM metrics WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM observations WHERE session_id = %s", (session_id,))
            conn.commit()
