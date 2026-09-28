"""Round-trip tests for harvest_ranges persistence layer."""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_mock_conn(rows: list[tuple] | None = None) -> MagicMock:
    """Return a mock psycopg connection that returns ``rows`` on fetchall."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchall.return_value = rows or []
    mock_conn.cursor.return_value = mock_cursor

    mock_txn = MagicMock()
    mock_txn.__enter__ = MagicMock(return_value=mock_txn)
    mock_txn.__exit__ = MagicMock(return_value=False)
    mock_conn.transaction.return_value = mock_txn

    return mock_conn


def test_round_trip_persist_then_load():
    """persist then load returns the same payload dict."""
    from src.study.harvest_ranges import HarvestRangeRow, load_harvest_ranges_by_hand, persist_harvest_range

    payload = {"combos": ["AhKh", "AsKs"], "weights": [0.5, 0.5], "equity": [0.65, 0.65]}
    row = HarvestRangeRow(hand_id="hand-001", decision_id="hand-001_dp0", seat=0, payload=payload)

    mock_conn = _make_mock_conn(rows=[("hand-001", "hand-001_dp0", 0, payload, None)])

    persist_harvest_range(row, _tsdb_conn=mock_conn)

    cursor = mock_conn.cursor.return_value
    assert cursor.execute.call_count >= 1
    insert_call = cursor.execute.call_args_list[0]
    sql_str = insert_call[0][0]
    assert "INSERT INTO harvest_ranges" in sql_str
    assert "ON CONFLICT (decision_id, seat) DO UPDATE" in sql_str

    mock_conn2 = _make_mock_conn(rows=[("hand-001", "hand-001_dp0", 0, payload, None)])
    results = load_harvest_ranges_by_hand("hand-001", _tsdb_conn=mock_conn2)
    assert len(results) == 1
    loaded = results[0]
    assert loaded.hand_id == "hand-001"
    assert loaded.decision_id == "hand-001_dp0"
    assert loaded.seat == 0
    assert loaded.payload == payload


def test_two_seats_same_decision_id():
    """Two rows with different seats for the same decision_id are independent."""
    from src.study.harvest_ranges import HarvestRangeRow, load_harvest_ranges_by_hand, persist_harvest_range

    payload_oop = {"combos": ["AhKh"], "weights": [1.0], "equity": [0.60]}
    payload_ip = {"combos": ["QcJc"], "weights": [1.0], "equity": [0.40]}

    row_oop = HarvestRangeRow(hand_id="hand-002", decision_id="hand-002_dp1", seat=0, payload=payload_oop)
    row_ip = HarvestRangeRow(hand_id="hand-002", decision_id="hand-002_dp1", seat=1, payload=payload_ip)

    mock_conn_persist = _make_mock_conn()
    persist_harvest_range(row_oop, _tsdb_conn=mock_conn_persist)
    persist_harvest_range(row_ip, _tsdb_conn=mock_conn_persist)

    db_rows = [
        ("hand-002", "hand-002_dp1", 0, payload_oop, None),
        ("hand-002", "hand-002_dp1", 1, payload_ip, None),
    ]
    mock_conn_load = _make_mock_conn(rows=db_rows)
    results = load_harvest_ranges_by_hand("hand-002", _tsdb_conn=mock_conn_load)

    assert len(results) == 2
    seats = {r.seat for r in results}
    assert seats == {0, 1}


def test_upsert_updates_payload():
    """Re-persisting same (decision_id, seat) upserts — SQL uses ON CONFLICT DO UPDATE."""

    from src.study.harvest_ranges import HarvestRangeRow, persist_harvest_range

    row1 = HarvestRangeRow(hand_id="hand-003", decision_id="hand-003_dp0", seat=0, payload={"v": 1})
    row2 = HarvestRangeRow(hand_id="hand-003", decision_id="hand-003_dp0", seat=0, payload={"v": 2})

    mock_conn = _make_mock_conn()

    persist_harvest_range(row1, _tsdb_conn=mock_conn)
    persist_harvest_range(row2, _tsdb_conn=mock_conn)

    cursor = mock_conn.cursor.return_value
    assert cursor.execute.call_count == 2

    for c in cursor.execute.call_args_list:
        sql_str = c[0][0]
        assert "ON CONFLICT (decision_id, seat) DO UPDATE" in sql_str


def test_load_orders_by_dp_index_then_seat():
    """load_harvest_ranges_by_hand issues ORDER BY dp-integer ASC, seat ASC."""
    from src.study.harvest_ranges import load_harvest_ranges_by_hand

    db_rows = [
        ("hand-004", "hand-004_dp0", 0, {}, None),
        ("hand-004", "hand-004_dp0", 1, {}, None),
        ("hand-004", "hand-004_dp2", 0, {}, None),
        ("hand-004", "hand-004_dp10", 0, {}, None),
    ]
    mock_conn = _make_mock_conn(rows=db_rows)
    load_harvest_ranges_by_hand("hand-004", _tsdb_conn=mock_conn)

    cursor = mock_conn.cursor.return_value
    execute_call = cursor.execute.call_args_list[0]
    sql_str = execute_call[0][0]

    assert "split_part(decision_id, '_dp', 2)" in sql_str
    assert "FROM harvest_ranges WHERE hand_id" in sql_str

    params = execute_call[0][1]
    assert "hand-004" in params
