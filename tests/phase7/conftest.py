# rot-allow-file
"""Shared fixtures for Phase 7 (Audit Closure) tests.

Provides tsdb_dsn / milvus_uri / milvus_token for integration tests (PC-gated),
plus Phase-7-specific fixtures:
  - mock_tsdb_conn: psycopg-shaped MagicMock with BOTH cursor() AND transaction()
    context-manager protocols. Phase 4's mock does NOT mock transaction(); the
    metrics-sink (src/metrics/sink.py:148) uses `with conn.transaction(),
    conn.cursor() as cur:` and would crash on the phase4 shape.
  - mock_milvus_client: MagicMock with .search.return_value=[[]] and
    .upsert.return_value=None
  - session_started_at: deterministic session start timestamp for
    metrics-idempotency tests (per D-07-11c — `ts` pinning makes ON CONFLICT
    actually idempotent across re-flushes).

Do NOT import from tests/phase4/conftest.py — Phase 7 mock_tsdb_conn is
self-contained to keep the transaction() mock explicit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# --- Integration fixtures (PC-gated) -----------------------------------------


@pytest.fixture(scope="session")
def tsdb_dsn() -> str:
    """Build TimescaleDB DSN from env vars (prod DB, not test DB)."""
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud if missing
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


@pytest.fixture(scope="session")
def milvus_uri() -> str:
    host = os.environ.get("MILVUS_HOST", "127.0.0.1")
    port = os.environ.get("MILVUS_PORT", "51530")
    return f"http://{host}:{port}"


@pytest.fixture(scope="session")
def milvus_token() -> str | None:
    """Milvus auth in ``root:password`` form (prepended if .env stores raw password)."""
    raw = os.environ.get("MILVUS_TOKEN")
    if raw is None:
        return None
    return raw if ":" in raw else f"root:{raw}"


# --- Phase 7 unit test fixtures ----------------------------------------------


@pytest.fixture
def mock_tsdb_conn() -> MagicMock:
    """psycopg-shaped MagicMock connection for unit tests.

    Supports BOTH context-manager protocols:

    1. Cursor protocol:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cur.fetchall()
        conn.commit()
        conn.close()

    2. Transaction protocol (src/metrics/sink.py:148 uses this):
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(...)

    Phase 4's mock_tsdb_conn does NOT mock transaction(); Phase 7 keeps the
    Phase 5 self-contained shape so Plans 07-05 / 07-06 unit tests work.
    """
    conn = MagicMock()
    mock_cur = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.transaction.return_value.__enter__ = MagicMock(return_value=conn)
    conn.transaction.return_value.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.fixture
def mock_milvus_client() -> MagicMock:
    """MagicMock pymilvus MilvusClient for unit tests.

    Defaults:
    - .search.return_value = [[]] — empty results (no neighbors); tests override per-case.
    - .upsert.return_value = None — pymilvus 3.0 upsert returns None on success.
    """
    client = MagicMock()
    client.search.return_value = [[]]
    client.upsert.return_value = None
    return client


@pytest.fixture
def session_started_at():
    """Deterministic session start timestamp for metrics-idempotency tests.

    Per D-07-11c — pinning `ts = session_started_at` (instead of
    `datetime.now(UTC)` on each flush) is what makes the migration-009
    composite UNIQUE catch re-flushes minutes/hours later, not just the
    millisecond-window double-flush.
    """
    from datetime import UTC, datetime

    return datetime(2026, 5, 19, 22, 0, 0, tzinfo=UTC)
