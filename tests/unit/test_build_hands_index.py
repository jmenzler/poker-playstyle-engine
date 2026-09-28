"""Unit tests for tools/build_hands_index — hand-level aggregation + upsert."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from tools.build_hands_index import (
    _deepest_street,
    _parse_hm3_ts,
    aggregate_real_hands,
    aggregate_sim_hands,
    upsert_hands_index,
)


def test_deepest_street_picks_max_by_order():
    assert _deepest_street(["preflop", "flop", "turn"]) == "turn"
    assert _deepest_street(["river", "preflop"]) == "river"
    assert _deepest_street(["preflop"]) == "preflop"


def test_deepest_street_ignores_unknown_and_none():
    assert _deepest_street([None, "flop", "bogus"]) == "flop"
    assert _deepest_street([None, "bogus"]) is None
    assert _deepest_street([]) is None


def test_parse_hm3_ts_valid_and_invalid():
    assert _parse_hm3_ts("2023-02-10 04:52:36") == datetime(2023, 2, 10, 4, 52, 36, tzinfo=UTC)
    assert _parse_hm3_ts(None) is None
    assert _parse_hm3_ts("") is None
    assert _parse_hm3_ts("not-a-date") is None


def test_aggregate_real_hands_groups_by_hand_id():
    decisions = [
        {"hand_id": "H1", "decision_idx": 0, "street": "preflop", "hero_pos": "BB", "stake": "NL5"},
        {"hand_id": "H1", "decision_idx": 1, "street": "flop", "hero_pos": "BB", "stake": "NL5"},
        {"hand_id": "H1", "decision_idx": 2, "street": "turn", "hero_pos": "BB", "stake": "NL5"},
        {"hand_id": "H2", "decision_idx": 0, "street": "preflop", "hero_pos": "BTN", "stake": "NL5"},
    ]
    ts = {"H1": datetime(2023, 2, 10, 4, 52, 36, tzinfo=UTC), "H2": None}
    rows = {r["hand_id"]: r for r in aggregate_real_hands(decisions, ts)}

    assert set(rows) == {"H1", "H2"}
    assert rows["H1"]["source"] == "hh"
    assert rows["H1"]["n_decisions"] == 3
    assert rows["H1"]["street_reached"] == "turn"
    assert rows["H1"]["hero_position"] == "BB"
    assert rows["H1"]["stake"] == "NL5"
    assert rows["H1"]["played_ts"] == datetime(2023, 2, 10, 4, 52, 36, tzinfo=UTC)
    assert rows["H2"]["n_decisions"] == 1
    assert rows["H2"]["street_reached"] == "preflop"
    assert rows["H2"]["played_ts"] is None


def test_aggregate_real_hands_skips_missing_hand_id():
    rows = aggregate_real_hands([{"street": "flop"}, {"hand_id": "", "street": "flop"}], {})
    assert rows == []


def test_aggregate_sim_hands_reads_observations():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [
        ("sess_h0", 2, datetime(2026, 1, 1, tzinfo=UTC), "CO", ["preflop", "river"]),
    ]
    rows = aggregate_sim_hands(conn)
    assert len(rows) == 1
    r = rows[0]
    assert r["hand_id"] == "sess_h0"
    assert r["source"] == "sim"
    assert r["n_decisions"] == 2
    assert r["street_reached"] == "river"
    assert r["hero_position"] == "CO"
    assert r["stake"] is None


def test_aggregate_sim_hands_hero_position_agg_is_ordered():
    """hero_position must pick a deterministic element (array_agg ... ORDER BY ts)."""
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []
    aggregate_sim_hands(conn)
    executed_sql = cur.execute.call_args[0][0]
    assert "array_agg(felt_snapshot->>'hero_position' ORDER BY ts)" in executed_sql


def test_upsert_hands_index_batches_and_commits():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    rows = [
        {
            "hand_id": f"H{i}",
            "source": "hh",
            "hero_position": "BB",
            "stake": "NL5",
            "n_decisions": 1,
            "street_reached": "flop",
            "played_ts": None,
        }
        for i in range(12)
    ]
    written = upsert_hands_index(conn, rows, batch_size=5)
    assert written == 12
    # 12 rows / batch 5 -> 3 flushes (5,5,2) -> 3 executemany + 3 commits
    assert cur.executemany.call_count == 3
    assert conn.commit.call_count == 3
