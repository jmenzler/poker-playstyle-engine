"""Phase 4 Plan 02: unit tests for the extended ObservationWriter.

Tests verify that flagged_sparse + max_neighbor_distance kwargs are included in
the _INSERT_SQL at the correct tuple positions (7 and 8 respectively).
Schema extended to 12 columns by Phase 10 Plan 02 (hand_id, decision_id, felt_snapshot).
"""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_writer(mock_conn: MagicMock):
    """Build an ObservationWriter with an injected mock connection."""
    from src.sim.observation_writer import ObservationWriter

    return ObservationWriter("ignored_dsn", session_id="test-session", _conn=mock_conn)


def _get_executed_rows(mock_conn: MagicMock) -> list[tuple]:
    """Extract the rows list passed to the last executemany call."""
    # cursor is obtained via context manager: conn.cursor().__enter__()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    assert mock_cur.executemany.called, "executemany was never called"
    _sql, rows = mock_cur.executemany.call_args[0]
    return rows


def _get_executed_sql(mock_conn: MagicMock) -> str:
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    assert mock_cur.executemany.called, "executemany was never called"
    sql, _rows = mock_cur.executemany.call_args[0]
    return sql


def test_flagged_sparse_and_max_distance_in_insert():
    """flagged_sparse=True and max_neighbor_distance=0.42 land at tuple positions 7, 8."""
    mock_conn = MagicMock()
    writer = _make_writer(mock_conn)

    writer.record(
        cluster_key="ck1",
        embedding=[0.1, 0.2],
        action_taken="call",
        flagged_sparse=True,
        max_neighbor_distance=0.42,
    )
    writer.close()

    rows = _get_executed_rows(mock_conn)
    assert len(rows) == 1
    row = rows[0]
    assert len(row) == 12, f"expected 12-column tuple, got {len(row)}: {row}"
    assert row[7] is True, f"expected True at index 7, got {row[7]}"
    assert abs(row[8] - 0.42) < 1e-9, f"expected 0.42 at index 8, got {row[8]}"


def test_default_kwargs_preserve_backward_compatibility():
    """record() without new kwargs writes False and None at positions 7, 8."""
    mock_conn = MagicMock()
    writer = _make_writer(mock_conn)

    writer.record(
        cluster_key="ck2",
        embedding=[0.3, 0.4],
        action_taken="fold",
    )
    writer.close()

    rows = _get_executed_rows(mock_conn)
    assert len(rows) == 1
    row = rows[0]
    assert len(row) == 12, f"expected 12-column tuple, got {len(row)}"
    assert row[7] is False, f"default flagged_sparse should be False, got {row[7]}"
    assert row[8] is None, f"default max_neighbor_distance should be None, got {row[8]}"


def test_explicit_none_max_distance_writes_null():
    """Explicit max_neighbor_distance=None passes through as SQL NULL (psycopg auto)."""
    mock_conn = MagicMock()
    writer = _make_writer(mock_conn)

    writer.record(
        cluster_key="ck3",
        embedding=[0.5],
        action_taken="bet_50",
        flagged_sparse=False,
        max_neighbor_distance=None,
    )
    writer.close()

    rows = _get_executed_rows(mock_conn)
    assert len(rows) == 1
    row = rows[0]
    assert row[8] is None, f"explicit None must pass through as NULL, got {row[8]}"


def test_insert_sql_has_12_placeholders():
    """_INSERT_SQL must have 12 %s placeholders and include Phase 4 + Phase 10 columns."""
    mock_conn = MagicMock()
    writer = _make_writer(mock_conn)

    writer.record(
        cluster_key="ck4",
        embedding=[0.1],
        action_taken="check",
        flagged_sparse=True,
        max_neighbor_distance=0.7,
    )
    writer.close()

    sql = _get_executed_sql(mock_conn)
    placeholder_count = sql.count("%s")
    assert placeholder_count == 12, f"expected 12 placeholders in INSERT SQL, got {placeholder_count}"
    assert "flagged_sparse" in sql, "INSERT SQL must include flagged_sparse column"
    assert "max_neighbor_distance" in sql, "INSERT SQL must include max_neighbor_distance column"
