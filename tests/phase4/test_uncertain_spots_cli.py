"""Unit tests for src/cli/uncertain_spots.py.

All tests use a mocked psycopg connection (mock_tsdb_conn from conftest) and
fabricate args via argparse.Namespace directly — no subprocess, no real DB.

Assertions cover:
  (a) Default invocation: WHERE flagged_sparse = TRUE, LIMIT 50
  (b) SELECT list contains all 6 required columns (CONTEXT.md Decision 4)
  (c) ORDER BY clause is exactly "max_neighbor_distance DESC NULLS LAST"
  (d) --session X adds AND session_id = %s with X in params
  (e) --street preflop adds AND cluster_key LIKE %s with '%street_class=preflop%'
  (f) --min-distance 0.3 adds AND max_neighbor_distance >= %s with 0.3 in params
  (g) Output is a valid JSON array containing the six expected column keys per row
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import configure_logging

# Configure logging once so structlog routes to stderr (not stdout),
# keeping stdout clean for JSON-only assertions.
configure_logging(level="WARNING")


def _make_args(
    session: str | None = None,
    limit: int = 50,
    street: str | None = None,
    min_distance: float | None = None,
) -> argparse.Namespace:
    """Fabricate an argparse.Namespace for uncertain-spots without invoking the parser."""
    return argparse.Namespace(
        session=session,
        limit=limit,
        street=street,
        min_distance=min_distance,
    )


def _make_conn(rows: list[tuple] | None = None) -> MagicMock:
    """Build a psycopg-shaped MagicMock connection returning canned rows.

    The mock_cur.description mirrors psycopg's cursor.description: a sequence
    of 2-tuples where index 0 is the column name.
    """
    if rows is None:
        rows = []
    conn = MagicMock()
    mock_cur = MagicMock()
    # Column names matching CONTEXT.md Decision 4 (6 columns)
    mock_cur.description = [
        ("obs_id",),
        ("cluster_key",),
        ("ts",),
        ("session_id",),
        ("action_taken",),
        ("max_neighbor_distance",),
    ]
    mock_cur.fetchall.return_value = rows
    conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


# ---------------------------------------------------------------------------
# (a) Default invocation — WHERE flagged_sparse = TRUE and LIMIT 50
# ---------------------------------------------------------------------------


def test_default_query_filters_flagged_sparse(capsys: pytest.CaptureFixture) -> None:
    """Default call must include WHERE flagged_sparse = TRUE."""
    from src.cli.uncertain_spots import run

    conn = _make_conn()
    run(_make_args(), _conn=conn)

    mock_cur = conn.cursor.return_value.__enter__.return_value
    sql_executed: str = mock_cur.execute.call_args[0][0]

    assert "WHERE flagged_sparse = TRUE" in sql_executed


def test_default_limit_is_50(capsys: pytest.CaptureFixture) -> None:
    """Default LIMIT param must be 50."""
    from src.cli.uncertain_spots import run

    conn = _make_conn()
    run(_make_args(), _conn=conn)

    mock_cur = conn.cursor.return_value.__enter__.return_value
    params: list = list(mock_cur.execute.call_args[0][1])

    assert params[-1] == 50, f"last param (LIMIT) should be 50, got {params}"


# ---------------------------------------------------------------------------
# (b) SELECT list contains all 6 required columns (CONTEXT.md Decision 4)
# ---------------------------------------------------------------------------


def test_select_list_contains_all_six_columns(capsys: pytest.CaptureFixture) -> None:
    """SQL must SELECT obs_id, cluster_key, ts, session_id, action_taken, max_neighbor_distance."""
    from src.cli.uncertain_spots import run

    conn = _make_conn()
    run(_make_args(), _conn=conn)

    mock_cur = conn.cursor.return_value.__enter__.return_value
    sql_executed: str = mock_cur.execute.call_args[0][0]

    assert "obs_id, cluster_key, ts, session_id, action_taken, max_neighbor_distance" in sql_executed


# ---------------------------------------------------------------------------
# (c) ORDER BY clause is exactly "max_neighbor_distance DESC NULLS LAST"
# ---------------------------------------------------------------------------


def test_order_by_max_neighbor_distance_desc_nulls_last(capsys: pytest.CaptureFixture) -> None:
    """ORDER BY must be exactly max_neighbor_distance DESC NULLS LAST (locked by CONTEXT.md)."""
    from src.cli.uncertain_spots import run

    conn = _make_conn()
    run(_make_args(), _conn=conn)

    mock_cur = conn.cursor.return_value.__enter__.return_value
    sql_executed: str = mock_cur.execute.call_args[0][0]

    assert "ORDER BY max_neighbor_distance DESC NULLS LAST" in sql_executed


# ---------------------------------------------------------------------------
# (d) --session X adds AND session_id = %s with X in params
# ---------------------------------------------------------------------------


def test_session_filter_adds_clause_and_param(capsys: pytest.CaptureFixture) -> None:
    """--session X must add AND session_id = %s and place X in params."""
    from src.cli.uncertain_spots import run

    conn = _make_conn()
    run(_make_args(session="sess-abc"), _conn=conn)

    mock_cur = conn.cursor.return_value.__enter__.return_value
    sql_executed: str = mock_cur.execute.call_args[0][0]
    params: list = list(mock_cur.execute.call_args[0][1])

    assert "AND session_id = %s" in sql_executed
    assert "sess-abc" in params


def test_no_session_filter_absent(capsys: pytest.CaptureFixture) -> None:
    """When --session is not supplied, the AND session_id = %s filter must be absent."""
    from src.cli.uncertain_spots import run

    conn = _make_conn()
    run(_make_args(), _conn=conn)

    mock_cur = conn.cursor.return_value.__enter__.return_value
    sql_executed: str = mock_cur.execute.call_args[0][0]

    # session_id appears in the SELECT list — only the WHERE filter clause must be absent
    assert "AND session_id = %s" not in sql_executed


# ---------------------------------------------------------------------------
# (e) --street preflop adds AND cluster_key LIKE %s with '%street_class=preflop%'
# ---------------------------------------------------------------------------


def test_street_preflop_adds_like_clause(capsys: pytest.CaptureFixture) -> None:
    """--street preflop must add AND cluster_key LIKE %s."""
    from src.cli.uncertain_spots import run

    conn = _make_conn()
    run(_make_args(street="preflop"), _conn=conn)

    mock_cur = conn.cursor.return_value.__enter__.return_value
    sql_executed: str = mock_cur.execute.call_args[0][0]
    params: list = list(mock_cur.execute.call_args[0][1])

    assert "AND cluster_key LIKE %s" in sql_executed
    assert "%street_class=preflop%" in params


def test_street_postflop_adds_like_clause(capsys: pytest.CaptureFixture) -> None:
    """--street postflop must add AND cluster_key LIKE %s with postflop token."""
    from src.cli.uncertain_spots import run

    conn = _make_conn()
    run(_make_args(street="postflop"), _conn=conn)

    mock_cur = conn.cursor.return_value.__enter__.return_value
    params: list = list(mock_cur.execute.call_args[0][1])

    assert "%street_class=postflop%" in params


# ---------------------------------------------------------------------------
# (f) --min-distance 0.3 adds AND max_neighbor_distance >= %s with 0.3 in params
# ---------------------------------------------------------------------------


def test_min_distance_adds_clause_and_param(capsys: pytest.CaptureFixture) -> None:
    """--min-distance 0.3 must add AND max_neighbor_distance >= %s with 0.3 in params."""
    from src.cli.uncertain_spots import run

    conn = _make_conn()
    run(_make_args(min_distance=0.3), _conn=conn)

    mock_cur = conn.cursor.return_value.__enter__.return_value
    sql_executed: str = mock_cur.execute.call_args[0][0]
    params: list = list(mock_cur.execute.call_args[0][1])

    assert "AND max_neighbor_distance >= %s" in sql_executed
    assert 0.3 in params


def test_no_min_distance_clause_absent(capsys: pytest.CaptureFixture) -> None:
    """When --min-distance is not supplied, the distance filter clause must be absent."""
    from src.cli.uncertain_spots import run

    conn = _make_conn()
    run(_make_args(), _conn=conn)

    mock_cur = conn.cursor.return_value.__enter__.return_value
    sql_executed: str = mock_cur.execute.call_args[0][0]

    assert "max_neighbor_distance >=" not in sql_executed


# ---------------------------------------------------------------------------
# (g) Output is valid JSON array with the six expected column keys per row
# ---------------------------------------------------------------------------


def test_output_is_valid_json_array(capsys: pytest.CaptureFixture) -> None:
    """stdout must be a parseable JSON array."""
    from src.cli.uncertain_spots import run

    conn = _make_conn()
    run(_make_args(), _conn=conn)

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert isinstance(parsed, list)


def test_output_rows_contain_six_columns(capsys: pytest.CaptureFixture) -> None:
    """Each output row must contain exactly the 6 columns from CONTEXT.md Decision 4."""
    import datetime

    from src.cli.uncertain_spots import run

    canned_row = (
        "obs-001",
        "street_class=preflop|pot_type=srp",
        datetime.datetime(2026, 5, 18, 12, 0, 0),
        "sess-001",
        "call",
        0.7,
    )
    conn = _make_conn(rows=[canned_row])
    run(_make_args(), _conn=conn)

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert len(parsed) == 1
    row = parsed[0]
    expected_keys = {"obs_id", "cluster_key", "ts", "session_id", "action_taken", "max_neighbor_distance"}
    assert set(row.keys()) == expected_keys


def test_output_null_distance_serializes_as_null(capsys: pytest.CaptureFixture) -> None:
    """max_neighbor_distance=None must serialize as JSON null (NULLS LAST support)."""
    import datetime

    from src.cli.uncertain_spots import run

    canned_row = (
        "obs-002",
        "street_class=preflop|pot_type=srp",
        datetime.datetime(2026, 5, 18, 12, 0, 0),
        "sess-001",
        "fold",
        None,
    )
    conn = _make_conn(rows=[canned_row])
    run(_make_args(), _conn=conn)

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed[0]["max_neighbor_distance"] is None


# ---------------------------------------------------------------------------
# Combined filter: all three optional filters applied at once
# ---------------------------------------------------------------------------


def test_all_filters_combined(capsys: pytest.CaptureFixture) -> None:
    """session + street + min-distance combined must produce all three AND clauses."""
    from src.cli.uncertain_spots import run

    conn = _make_conn()
    run(_make_args(session="s1", limit=10, street="postflop", min_distance=0.5), _conn=conn)

    mock_cur = conn.cursor.return_value.__enter__.return_value
    sql_executed: str = mock_cur.execute.call_args[0][0]
    params: list = list(mock_cur.execute.call_args[0][1])

    assert "AND session_id = %s" in sql_executed
    assert "AND cluster_key LIKE %s" in sql_executed
    assert "AND max_neighbor_distance >= %s" in sql_executed
    assert "s1" in params
    assert "%street_class=postflop%" in params
    assert 0.5 in params
    assert params[-1] == 10  # LIMIT
