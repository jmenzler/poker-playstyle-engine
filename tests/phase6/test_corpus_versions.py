"""Unit tests for src/study/corpus_versions.py.

Mocks psycopg cursors (SQL-substring dispatch, like test_study_patches.py)
covering snapshot, list_versions ordering, and resolve_cutoff_ts fail-loud.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.study.corpus_versions import list_versions, resolve_cutoff_ts, snapshot

_SNAPSHOT_DESC = [("version",), ("cutoff_ts",)]
_LIST_DESC = [("version",), ("cutoff_ts",), ("source",), ("label",), ("created_at",)]
_RESOLVE_DESC = [("cutoff_ts",)]


def _mock_conn(*, insert_row=None, list_rows=(), resolve_row=None, captured=None):
    """Mock psycopg conn dispatching INSERT/list/resolve by SQL substring.

    Records whether ``conn.transaction()`` was entered so write paths can be
    asserted to commit (non-autocommit conns roll back otherwise).
    """
    conn = MagicMock()
    conn.tx_entered = False

    tx_cm = MagicMock()

    def _tx_enter(_self):
        conn.tx_entered = True
        return tx_cm

    tx_cm.__enter__ = _tx_enter
    tx_cm.__exit__ = lambda *a: False
    conn.transaction.return_value = tx_cm

    def cursor_factory():
        cm = MagicMock()
        cm.__enter__ = lambda self: cm
        cm.__exit__ = lambda *a: False
        cm.fetchall.return_value = []
        cm.fetchone.return_value = None

        def execute(sql, params=None):
            s = " ".join(sql.split()).lower()
            if captured is not None:
                captured["sql"] = s
                captured["params"] = params
            if "insert into corpus_versions" in s:
                cm.description = _SNAPSHOT_DESC
                cm.fetchone.return_value = insert_row
            elif "from corpus_versions" in s and "where version = %s" in s:
                cm.description = _RESOLVE_DESC
                cm.fetchone.return_value = resolve_row
            elif "from corpus_versions" in s:
                cm.description = _LIST_DESC
                cm.fetchall.return_value = list(list_rows)
            return None

        cm.execute.side_effect = execute
        return cm

    conn.cursor.side_effect = cursor_factory
    return conn


def test_snapshot_issues_insert_with_three_params_and_returns_version():
    captured: dict = {}
    conn = _mock_conn(insert_row=(7, 1_700_000_000), captured=captured)
    result = snapshot("v1", source="autoloop", _tsdb_conn=conn)

    assert result["version"] == 7
    assert result["cutoff_ts"] == 1_700_000_000

    assert "insert into corpus_versions" in captured["sql"]
    assert "returning" in captured["sql"]
    params = captured["params"]
    assert len(params) == 3
    # cutoff_ts is an int epoch, source + label echo the args.
    cutoff_ts, source, label = params
    assert isinstance(cutoff_ts, int)
    assert source == "autoloop"
    assert label == "v1"


def test_snapshot_wraps_insert_in_transaction():
    """The INSERT runs inside conn.transaction() so an injected non-autocommit
    conn commits instead of silently rolling back on close."""
    conn = _mock_conn(insert_row=(7, 1_700_000_000))
    snapshot("v1", source="autoloop", _tsdb_conn=conn)
    assert conn.tx_entered, "snapshot must wrap its INSERT in conn.transaction()"


def test_snapshot_defaults_source_to_manual():
    captured: dict = {}
    conn = _mock_conn(insert_row=(1, 42), captured=captured)
    snapshot("genesis", _tsdb_conn=conn)
    _cutoff_ts, source, _label = captured["params"]
    assert source == "manual"


def test_resolve_cutoff_ts_returns_int():
    conn = _mock_conn(resolve_row=(1_700_000_500,))
    assert resolve_cutoff_ts(3, _tsdb_conn=conn) == 1_700_000_500


def test_resolve_cutoff_ts_unknown_version_raises():
    conn = _mock_conn(resolve_row=None)
    with pytest.raises(ValueError):
        resolve_cutoff_ts(999, _tsdb_conn=conn)


def test_list_versions_orders_by_version_desc():
    captured: dict = {}
    rows = [
        (2, 1_700_000_100, "autoloop", "v2", "2026-06-24T10:00:00Z"),
        (1, 1_700_000_000, "manual", "v1", "2026-06-24T09:00:00Z"),
    ]
    conn = _mock_conn(list_rows=rows, captured=captured)
    result = list_versions(_tsdb_conn=conn)

    assert "order by version desc" in captured["sql"]
    assert len(result) == 2
    assert result[0]["version"] == 2
    assert {"version", "cutoff_ts", "source", "label", "created_at"} <= set(result[0].keys())
