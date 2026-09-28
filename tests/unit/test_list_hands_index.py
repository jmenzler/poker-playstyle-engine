"""Unit tests for the hands_index-backed list_hands (Hands-tab all-source browser)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.study.hands import list_hands


def _mock_conn(rows=None, cols=None):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.description = [(c,) for c in (cols or ["hand_id", "source", "solved"])]
    cur.fetchall.return_value = rows or []
    return conn, cur


def _executed(cur):
    sql, params = cur.execute.call_args.args
    return sql, list(params)


def test_source_all_no_source_filter():
    conn, cur = _mock_conn()
    list_hands(source="all", _tsdb_conn=conn)
    sql, _ = _executed(cur)
    assert "h.source = %s" not in sql
    assert "hands_index" in sql


def test_source_hh_filters():
    conn, cur = _mock_conn()
    list_hands(source="hh", _tsdb_conn=conn)
    sql, params = _executed(cur)
    assert "AND h.source = %s" in sql
    assert "hh" in params


def test_solved_true_adds_exists_filter():
    conn, cur = _mock_conn()
    list_hands(solved=True, _tsdb_conn=conn)
    sql, _ = _executed(cur)
    assert "AND EXISTS (SELECT 1 FROM harvest_ranges" in sql
    assert "NOT EXISTS" not in sql


def test_solved_false_adds_not_exists_filter():
    conn, cur = _mock_conn()
    list_hands(solved=False, _tsdb_conn=conn)
    sql, _ = _executed(cur)
    assert "AND NOT EXISTS (SELECT 1 FROM harvest_ranges" in sql


def test_solved_none_no_where_exists():
    conn, cur = _mock_conn()
    list_hands(solved=None, _tsdb_conn=conn)
    sql, _ = _executed(cur)
    # EXISTS still appears once in the SELECT (derived `solved` column), never in WHERE.
    assert sql.count("EXISTS (SELECT 1 FROM harvest_ranges") == 1


def test_solved_keys_on_hand_id():
    conn, cur = _mock_conn()
    list_hands(_tsdb_conn=conn)
    sql, _ = _executed(cur)
    assert "hr.hand_id = h.hand_id" in sql


def test_corpus_solved_column_in_select():
    conn, cur = _mock_conn()
    list_hands(_tsdb_conn=conn)
    sql, _ = _executed(cur)
    assert "AS corpus_solved" in sql
    assert "JOIN solver_cache sc ON sc.decision_id = o.decision_id" in sql
    assert "sc.exploitability_pct > 0" in sql
    assert "o.hand_id = h.hand_id" in sql


def test_solved_param_never_filters_corpus():
    conn, cur = _mock_conn()
    list_hands(solved=True, _tsdb_conn=conn)
    sql, _ = _executed(cur)
    # solver_cache EXISTS appears once (the SELECT column); the WHERE filter is harvest_ranges.
    assert sql.count("JOIN solver_cache sc") == 1


def test_corpus_solved_true_adds_exists_filter():
    conn, cur = _mock_conn()
    list_hands(corpus_solved=True, _tsdb_conn=conn)
    sql, _ = _executed(cur)
    # SELECT column + WHERE filter.
    assert sql.count("JOIN solver_cache sc") == 2
    assert "NOT EXISTS (SELECT 1 FROM observations o" not in sql


def test_corpus_solved_false_adds_not_exists_filter():
    conn, cur = _mock_conn()
    list_hands(corpus_solved=False, _tsdb_conn=conn)
    sql, _ = _executed(cur)
    assert "AND NOT EXISTS (SELECT 1 FROM observations o" in sql


def test_pagination_and_order():
    conn, cur = _mock_conn()
    list_hands(limit=25, offset=50, _tsdb_conn=conn)
    sql, params = _executed(cur)
    assert "ORDER BY h.played_ts DESC NULLS LAST" in sql
    assert "LIMIT %s OFFSET %s" in sql
    assert params[-2:] == [25, 50]


def test_stake_filter():
    conn, cur = _mock_conn()
    list_hands(stake="NL5", _tsdb_conn=conn)
    sql, params = _executed(cur)
    assert "AND h.stake = %s" in sql
    assert "NL5" in params


def test_invalid_source_raises():
    conn, _ = _mock_conn()
    with pytest.raises(ValueError, match="source"):
        list_hands(source="bogus", _tsdb_conn=conn)


def test_returns_dict_rows():
    conn, _ = _mock_conn(rows=[("H1", "hh", True)], cols=["hand_id", "source", "solved"])
    out = list_hands(_tsdb_conn=conn)
    assert out == [{"hand_id": "H1", "source": "hh", "solved": True}]
