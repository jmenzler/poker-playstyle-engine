"""Phase 9 / solver_cache — unit + integration tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest


def test_persist_solve_issues_parameterized_insert(mock_tsdb_conn):
    """persist_solve issues a single parameterized INSERT with no f-string SQL."""
    from src.eval.solver_cache import SolverCacheEntry, persist_solve

    entry = SolverCacheEntry(
        cluster_key="street_class=postflop|test_cluster",
        action_dist={"check": 0.6, "fold": 0.4},
        exploitability_pct=3.5,
    )
    persist_solve(entry, _tsdb_conn=mock_tsdb_conn)

    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    assert mock_cur.execute.call_count == 1
    sql_str, _params = mock_cur.execute.call_args[0]
    assert "INSERT INTO solver_cache" in sql_str
    assert "%s" in sql_str
    assert "ON CONFLICT" in sql_str
    assert "DO NOTHING" in sql_str
    # No f-string interpolation — cluster_key must be a param, not embedded in SQL
    assert entry.cluster_key not in sql_str


def test_persist_solve_jsonb_columns(mock_tsdb_conn):
    """persist_solve casts action_dist and spot_features with ::jsonb."""
    from src.eval.solver_cache import SolverCacheEntry, persist_solve

    entry = SolverCacheEntry(
        cluster_key="test|key",
        action_dist={"check": 1.0},
        exploitability_pct=0.0,
    )
    persist_solve(entry, _tsdb_conn=mock_tsdb_conn)

    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    sql_str, _ = mock_cur.execute.call_args[0]
    assert "::jsonb" in sql_str


def test_load_solve_returns_entry_when_row_found(mock_tsdb_conn):
    """load_solve returns a SolverCacheEntry when the DB returns a row."""
    from src.eval.solver_cache import SolverCacheEntry, load_solve

    action_dist = {"check": 0.6, "fold": 0.4}
    solved_at = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = (
        "test|cluster",
        action_dist,
        5.2,
        None,
        "fixture",
        solved_at,
    )

    result = load_solve("test|cluster", _tsdb_conn=mock_tsdb_conn)

    assert isinstance(result, SolverCacheEntry)
    assert result.cluster_key == "test|cluster"
    assert result.action_dist == action_dist
    assert result.exploitability_pct == pytest.approx(5.2)
    assert result.solver_version == "fixture"


def test_load_solve_returns_none_when_no_row(mock_tsdb_conn):
    """load_solve returns None when cluster_key not found in DB."""
    from src.eval.solver_cache import load_solve

    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = None

    result = load_solve("nonexistent|key", _tsdb_conn=mock_tsdb_conn)

    assert result is None


def test_load_solve_uses_order_by_solved_at_desc(mock_tsdb_conn):
    """load_solve queries latest row first (ORDER BY solved_at DESC LIMIT 1)."""
    from src.eval.solver_cache import load_solve

    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = None

    load_solve("any|key", _tsdb_conn=mock_tsdb_conn)

    sql_str, _params = mock_cur.execute.call_args[0]
    assert "ORDER BY solved_at DESC" in sql_str
    assert "LIMIT 1" in sql_str
    assert "WHERE cluster_key" in sql_str


def test_solver_cache_entry_struct_fields():
    """SolverCacheEntry has required fields with correct defaults."""
    from src.eval.solver_cache import SolverCacheEntry

    entry = SolverCacheEntry(
        cluster_key="k",
        action_dist={"check": 1.0},
        exploitability_pct=1.0,
    )
    assert entry.spot_features is None
    assert entry.solver_version is None
    assert entry.solved_at is None


@pytest.mark.integration
def test_persist_roundtrip(tsdb_dsn):
    """Solver cache persist+read roundtrip against real TimescaleDB (migration 015 required)."""
    from src.eval.solver_cache import SolverCacheEntry, load_solve, persist_solve

    cluster_key = "integration_test|roundtrip_spot"
    action_dist = {"check": 0.6, "fold": 0.4}
    entry = SolverCacheEntry(
        cluster_key=cluster_key,
        action_dist=action_dist,
        exploitability_pct=2.5,
        solver_version="fixture",
    )

    import psycopg

    conn = psycopg.connect(tsdb_dsn)
    try:
        persist_solve(entry, _tsdb_conn=conn)
        result = load_solve(cluster_key, _tsdb_conn=conn)
        assert result is not None
        assert result.action_dist == action_dist
        assert result.exploitability_pct == pytest.approx(2.5)
    finally:
        # Clean up test row to keep DB idempotent
        with conn.cursor() as cur:
            cur.execute("DELETE FROM solver_cache WHERE cluster_key = %s", (cluster_key,))
        conn.commit()
        conn.close()
