"""Phase 10 Plan 02: unit tests for the 12-column ObservationWriter extension.

Tests verify that hand_id, decision_id, felt_snapshot kwargs are included in
the 12-column _INSERT_SQL at the correct tuple positions (9, 10, 11).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock


def _make_writer(mock_conn: MagicMock):
    """Build an ObservationWriter with an injected mock connection."""
    from src.sim.observation_writer import ObservationWriter

    return ObservationWriter("ignored_dsn", session_id="test-session", _conn=mock_conn)


def _get_executed_rows(mock_conn: MagicMock) -> list[tuple]:
    """Extract the rows list passed to the last executemany call."""
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    assert mock_cur.executemany.called, "executemany was never called"
    _sql, rows = mock_cur.executemany.call_args[0]
    return rows


def _get_executed_sql(mock_conn: MagicMock) -> str:
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    assert mock_cur.executemany.called, "executemany was never called"
    sql, _rows = mock_cur.executemany.call_args[0]
    return sql


_SAMPLE_FELT = {
    "street": "flop",
    "hero_position": "BTN",
    "hero_hole_cards": ["Ah", "Kd"],
    "board_cards": ["2c", "7s", "Jh"],
    "pot_size_bb": 4.5,
    "effective_stack_bb": 97.5,
    "hero_facing_bet_bb": 0.0,
    "hero_bet_size_bb": 0.0,
    "action_sequence": ["fold", "fold", "call"],
    "opponents_remaining": 1,
    "prior_street_aggressor": "CO",
}


def test_felt_kwargs_in_insert_at_correct_positions():
    """hand_id, decision_id, felt_snapshot land at tuple positions 9, 10, 11."""
    mock_conn = MagicMock()
    writer = _make_writer(mock_conn)

    writer.record(
        cluster_key="ck1",
        embedding=[0.1, 0.2],
        action_taken="check",
        hand_id="sess_h0",
        decision_id="sess_h0_dp0",
        felt_snapshot=_SAMPLE_FELT,
    )
    writer.close()

    rows = _get_executed_rows(mock_conn)
    assert len(rows) == 1
    row = rows[0]
    assert len(row) == 12, f"expected 12-column tuple, got {len(row)}: {row}"
    assert row[9] == "sess_h0", f"hand_id at position 9, got {row[9]}"
    assert row[10] == "sess_h0_dp0", f"decision_id at position 10, got {row[10]}"
    # felt_snapshot should be json.dumps'd
    assert isinstance(row[11], str), f"felt_snapshot should be JSON string, got {type(row[11])}"
    decoded = json.loads(row[11])
    assert decoded["street"] == "flop"
    assert decoded["hero_position"] == "BTN"
    assert decoded["hero_hole_cards"] == ["Ah", "Kd"]


def test_felt_snapshot_none_when_not_provided():
    """felt_snapshot=None (default) writes None at position 11."""
    mock_conn = MagicMock()
    writer = _make_writer(mock_conn)

    writer.record(
        cluster_key="ck2",
        embedding=[0.3, 0.4],
        action_taken="fold",
        hand_id="sess_h1",
        decision_id="sess_h1_dp2",
        felt_snapshot=None,
    )
    writer.close()

    rows = _get_executed_rows(mock_conn)
    row = rows[0]
    assert len(row) == 12
    assert row[11] is None, f"None felt_snapshot should write None, got {row[11]}"


def test_backward_compatible_no_new_kwargs():
    """record() without new kwargs still works; positions 9, 10, 11 are all None."""
    mock_conn = MagicMock()
    writer = _make_writer(mock_conn)

    writer.record(
        cluster_key="ck3",
        embedding=[0.5, 0.6],
        action_taken="bet_50",
    )
    writer.close()

    rows = _get_executed_rows(mock_conn)
    assert len(rows) == 1
    row = rows[0]
    assert len(row) == 12, f"expected 12-column tuple even without new kwargs, got {len(row)}"
    assert row[9] is None, f"hand_id default should be None, got {row[9]}"
    assert row[10] is None, f"decision_id default should be None, got {row[10]}"
    assert row[11] is None, f"felt_snapshot default should be None, got {row[11]}"


def test_insert_sql_has_12_placeholders():
    """_INSERT_SQL must have 12 %s placeholders and include the 3 new columns."""
    mock_conn = MagicMock()
    writer = _make_writer(mock_conn)

    writer.record(
        cluster_key="ck4",
        embedding=[0.1],
        action_taken="call",
        hand_id="h",
        decision_id="d",
        felt_snapshot=_SAMPLE_FELT,
    )
    writer.close()

    sql = _get_executed_sql(mock_conn)
    placeholder_count = sql.count("%s")
    assert placeholder_count == 12, f"expected 12 placeholders, got {placeholder_count}"
    assert "hand_id" in sql, "INSERT SQL must include hand_id column"
    assert "decision_id" in sql, "INSERT SQL must include decision_id column"
    assert "felt_snapshot" in sql, "INSERT SQL must include felt_snapshot column"


def test_felt_snapshot_all_11_fields_round_trip():
    """All 11 GameState fields in felt_snapshot survive JSON round-trip."""
    mock_conn = MagicMock()
    writer = _make_writer(mock_conn)

    writer.record(
        cluster_key="ck5",
        embedding=[0.1],
        action_taken="raise",
        felt_snapshot=_SAMPLE_FELT,
    )
    writer.close()

    rows = _get_executed_rows(mock_conn)
    row = rows[0]
    decoded = json.loads(row[11])
    expected_keys = {
        "street",
        "hero_position",
        "hero_hole_cards",
        "board_cards",
        "pot_size_bb",
        "effective_stack_bb",
        "hero_facing_bet_bb",
        "hero_bet_size_bb",
        "action_sequence",
        "opponents_remaining",
        "prior_street_aggressor",
    }
    assert expected_keys <= decoded.keys(), f"missing keys: {expected_keys - decoded.keys()}"
    assert decoded == _SAMPLE_FELT, f"round-trip mismatch: {decoded}"
