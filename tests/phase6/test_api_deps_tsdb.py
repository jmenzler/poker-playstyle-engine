"""src/api/deps.get_tsdb — per-request teardown rolls back before closing."""

from __future__ import annotations

from unittest.mock import MagicMock

import src.api.deps as deps


def test_get_tsdb_rolls_back_then_closes(monkeypatch):
    """Uncommitted/aborted txn state must be rolled back deterministically on teardown."""
    conn = MagicMock()
    monkeypatch.setattr(deps.timescale, "connect", lambda dsn: conn)
    monkeypatch.setenv("TSDB_PASSWORD", "x")

    gen = deps.get_tsdb()
    yielded = next(gen)
    assert yielded is conn

    # Exhaust the generator to trigger the finally block.
    try:
        next(gen)
    except StopIteration:
        pass

    conn.rollback.assert_called_once()
    conn.close.assert_called_once()
