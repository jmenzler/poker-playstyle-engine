"""SIM-06 guard: an explicit, reused --session-id must not silently double-write."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tools.run_sim_session import (
    SessionIdCollisionError,
    _guard_explicit_session_id,
)


def _conn_with_existing(exists: bool, *, deleted: int = 0) -> tuple[MagicMock, MagicMock]:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    cur.fetchone.return_value = (1,) if exists else None
    cur.rowcount = deleted
    return conn, cur


def test_guard_raises_on_existing_session_without_force():
    conn, cur = _conn_with_existing(True)
    with pytest.raises(SessionIdCollisionError):
        _guard_explicit_session_id(conn, "bench-001", force=False)
    # The SELECT ran; no DELETE was issued.
    sqls = [c.args[0] for c in cur.execute.call_args_list]
    assert any("SELECT 1 FROM observations" in s for s in sqls)
    assert not any("DELETE" in s for s in sqls)
    conn.commit.assert_not_called()


def test_guard_noop_when_session_absent():
    conn, cur = _conn_with_existing(False)
    _guard_explicit_session_id(conn, "fresh-id", force=False)
    sqls = [c.args[0] for c in cur.execute.call_args_list]
    assert any("SELECT 1 FROM observations" in s for s in sqls)
    assert not any("DELETE" in s for s in sqls)
    conn.commit.assert_not_called()


def test_guard_force_deletes_then_commits():
    conn, cur = _conn_with_existing(True, deleted=42)
    _guard_explicit_session_id(conn, "bench-001", force=True)
    sqls = [c.args[0] for c in cur.execute.call_args_list]
    assert any("SELECT 1 FROM observations" in s for s in sqls)
    delete_calls = [c for c in cur.execute.call_args_list if "DELETE" in c.args[0]]
    assert len(delete_calls) == 1
    assert delete_calls[0].args[1] == ("bench-001",)
    conn.commit.assert_called_once()
