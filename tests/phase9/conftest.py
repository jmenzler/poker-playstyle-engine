# rot-allow-file
"""Shared fixtures for Phase 9 (Eval Harness & Live Play) tests.

Provides:
  - tsdb_dsn: session-scoped DSN from env vars for integration tests
  - mock_tsdb_conn: psycopg-shaped MagicMock with cursor() AND transaction()
  - mock_milvus_client: MagicMock with .search.return_value=[[]]
  - known_action_dist: uniform dict over 15 CANONICAL_ACTIONS for unit tests
  - solved_spot_fixture: hand-verified solver result for cache roundtrip tests
    without the solver binary (RESEARCH A4/Open-Q-1 fallback)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@pytest.fixture(scope="session")
def tsdb_dsn() -> str:
    """Build TimescaleDB DSN from env vars (prod DB, not test DB)."""
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


@pytest.fixture
def mock_tsdb_conn() -> MagicMock:
    """psycopg-shaped MagicMock connection for unit tests.

    Supports BOTH context-manager protocols:
    1. Cursor: with conn.cursor() as cur: cur.execute(...)
    2. Transaction: with conn.transaction(): with conn.cursor() as cur: ...
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
    - .search.return_value = [[]] — empty results (no neighbors)
    """
    client = MagicMock()
    client.search.return_value = [[]]
    return client


@pytest.fixture
def known_action_dist() -> dict[str, float]:
    """Deterministic uniform 15-action probability dict for unit tests."""
    from src.decision_engine.blending import CANONICAL_ACTIONS

    n = len(CANONICAL_ACTIONS)
    return {a: 1.0 / n for a in CANONICAL_ACTIONS}


@pytest.fixture
def solved_spot_fixture() -> dict:
    """Fabricated solver-shaped result for cache round-trip tests."""
    from src.decision_engine.blending import CANONICAL_ACTIONS

    n = len(CANONICAL_ACTIONS)
    action_dist = {a: 1.0 / n for a in CANONICAL_ACTIONS}
    return {
        "cluster_key": "preflop_BTN_vs_BB_3bet_100bb",
        "action_dist": action_dist,
        "exploitability_pct": 2.3,
    }
