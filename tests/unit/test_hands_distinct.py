"""Unit tests for list_distinct() whitelist enforcement (D-14).

No DB required — whitelist rejection is pure Python logic.
"""

from __future__ import annotations

import pytest

from src.study.hands import _DISTINCT_WHITELIST, list_distinct


def test_whitelist_contains_exactly_three_fields() -> None:
    """The whitelist must be exactly {session_id, cluster_key, obs_id}."""
    assert _DISTINCT_WHITELIST == frozenset({"session_id", "cluster_key", "obs_id"})


def test_non_whitelisted_field_raises_value_error() -> None:
    """list_distinct with an unlisted field raises ValueError — no DB needed."""
    with pytest.raises(ValueError, match="not in whitelist"):
        list_distinct("evil; DROP TABLE observations; --")


def test_sql_injection_attempt_raises_value_error() -> None:
    """Classic SQL injection pattern is rejected before any DB access."""
    with pytest.raises(ValueError):
        list_distinct("1 UNION SELECT password FROM users --")


def test_empty_string_raises_value_error() -> None:
    """Empty string is not in the whitelist."""
    with pytest.raises(ValueError):
        list_distinct("")


def test_valid_field_calls_db(monkeypatch) -> None:
    """list_distinct with a whitelisted field calls the DB and returns results."""
    rows = [("sess_a",), ("sess_b",)]
    cursor_mock = type(
        "Cur",
        (),
        {
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: False,
            "execute": lambda s, *a: None,
            "fetchall": lambda s: rows,
        },
    )()
    conn_mock = type(
        "Conn",
        (),
        {
            "cursor": lambda s: cursor_mock,
            "close": lambda s: None,
        },
    )()

    list_distinct("session_id", _tsdb_conn=conn_mock)


def test_valid_field_returns_string_list(monkeypatch) -> None:
    """list_distinct returns a list[str] extracted from the first column."""
    rows = [("alpha",), ("beta",), ("gamma",)]
    cursor_mock = type(
        "Cur",
        (),
        {
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: False,
            "execute": lambda s, *a: None,
            "fetchall": lambda s: rows,
        },
    )()
    conn_mock = type(
        "Conn",
        (),
        {
            "cursor": lambda s: cursor_mock,
            "close": lambda s: None,
        },
    )()

    result = list_distinct("cluster_key", _tsdb_conn=conn_mock)
    assert result == ["alpha", "beta", "gamma"]
