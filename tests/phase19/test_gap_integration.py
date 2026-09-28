from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


def _tsdb_dsn() -> str:
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ.get("TSDB_PASSWORD", "")
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


def test_resolve_round_trip():
    import psycopg
    from src.study.gaps import resolve_gap

    dsn = _tsdb_dsn()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT decision_id FROM observations "
                "WHERE flagged_sparse = TRUE AND decision_id IS NOT NULL "
                "AND hand_id IS NOT NULL LIMIT 1"
            )
            row = cur.fetchone()

    if row is None:
        pytest.skip("no flagged_sparse observations in TSDB — seed the DB first")

    decision_id = row[0]

    with psycopg.connect(dsn) as conn:
        result = resolve_gap(
            decision_id,
            {"check": 1.0},
            _tsdb_conn=conn,
            _milvus=None,
        )
        assert "patch_id" in result
        assert result["patch_id"]

        with conn.cursor() as cur:
            cur.execute(
                "SELECT flagged_sparse FROM observations WHERE decision_id = %s LIMIT 1",
                (decision_id,),
            )
            obs_row = cur.fetchone()
        assert obs_row is not None
        assert obs_row[0] is False

        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM solver_cache WHERE decision_id = %s AND solver_version = 'human_resolved'",
                (decision_id,),
            )
            cache_row = cur.fetchone()
        assert cache_row is not None
