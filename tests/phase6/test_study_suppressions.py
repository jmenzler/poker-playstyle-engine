"""Unit tests for src/study/suppressions.py.

Mocked psycopg cursors — no live DB. Validates the suppress/unsuppress/is_suppressed
contract that the Strategy Leaks panel's `[accept as intentional]` flow depends on.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.study.suppressions import is_suppressed, suppress, unsuppress


def _mock_conn(fetchone_returns=None, rowcount=0):
    """Single-cursor mock that returns ``fetchone_returns`` and reports ``rowcount``."""
    conn = MagicMock()
    cm = MagicMock()
    cm.__enter__ = lambda s: cm
    cm.__exit__ = lambda *a: False
    cm.fetchone.return_value = fetchone_returns
    cm.rowcount = rowcount
    conn.cursor.return_value = cm
    conn.transaction.return_value.__enter__ = lambda s: s
    conn.transaction.return_value.__exit__ = lambda *a: False
    return conn


def test_suppress_returns_uuid_string():
    conn = _mock_conn()
    sid = suppress("cluster_xyz", reason="intentional", _tsdb_conn=conn)
    # UUID4 string canonical form length = 36 (8-4-4-4-12 + 4 dashes)
    assert isinstance(sid, str)
    assert len(sid) == 36
    conn.cursor.return_value.execute.assert_called_once()


def test_suppress_invokes_insert_with_params():
    conn = _mock_conn()
    suppress("cluster_xyz", reason="intentional", _tsdb_conn=conn)
    call_args = conn.cursor.return_value.execute.call_args
    sql_text = call_args.args[0]
    params = call_args.args[1]
    assert "INSERT INTO leak_suppressions" in sql_text
    # Parameterized — no f-string user input
    assert "%s" in sql_text
    # cluster_key + reason both passed as params
    assert "cluster_xyz" in params
    assert "intentional" in params


def test_unsuppress_returns_rowcount():
    conn = _mock_conn(rowcount=2)
    n = unsuppress("cluster_xyz", _tsdb_conn=conn)
    assert n == 2


def test_unsuppress_invokes_update_with_param():
    conn = _mock_conn(rowcount=1)
    unsuppress("cluster_xyz", _tsdb_conn=conn)
    call_args = conn.cursor.return_value.execute.call_args
    sql_text = call_args.args[0]
    params = call_args.args[1]
    assert "UPDATE leak_suppressions" in sql_text
    assert "active = FALSE" in sql_text
    assert "cluster_xyz" in params


def test_is_suppressed_true_when_row_exists():
    conn = _mock_conn(fetchone_returns=(1,))
    assert is_suppressed("cluster_xyz", _tsdb_conn=conn) is True


def test_is_suppressed_false_when_no_row():
    conn = _mock_conn(fetchone_returns=None)
    assert is_suppressed("cluster_xyz", _tsdb_conn=conn) is False
