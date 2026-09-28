"""Shared fixtures for Phase 5 (Auto-Loop) tests.

Provides tsdb_dsn / milvus_uri / milvus_token for integration tests (PC-gated),
plus Phase-5-specific fixtures:
  - mock_tsdb_conn: psycopg-shaped MagicMock with BOTH cursor() AND transaction()
    context-manager protocols (Phase 4's mock does NOT mock transaction()).
  - mock_milvus_client: MagicMock with .search.return_value=[[]] and .upsert.return_value=None
  - autoloop_config: AutoLoopConfig with FAST settings for unit tests (low n_hands/resamples)
  - synthetic_cluster_key: fixed BTN postflop cluster key for E2E and unit tests

Do NOT import from tests/phase4/conftest.py — Phase 5 mock_tsdb_conn is self-contained
to avoid coupling and to make the transaction() mock explicit.
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


# --- Phase 5 unit test fixtures -----------------------------------------------


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

    2. Transaction protocol (Phase 5 PatchEngine uses this):
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(...)

    Phase 4's mock_tsdb_conn does NOT mock transaction(); Phase 5 extends it.
    Self-contained: do not import from tests/phase4/conftest.py.
    """
    conn = MagicMock()
    mock_cur = MagicMock()
    # cursor() context manager: __enter__ returns the cursor mock
    conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    # transaction() context manager: __enter__ returns conn itself (psycopg3 semantics)
    conn.transaction.return_value.__enter__ = MagicMock(return_value=conn)
    conn.transaction.return_value.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.fixture
def mock_milvus_client() -> MagicMock:
    """MagicMock pymilvus MilvusClient for unit tests.

    Defaults:
    - .search.return_value = [[]] — empty results (no neighbors); tests override per-case.
    - .upsert.return_value = None — pymilvus 3.0 upsert returns None on success.

    Tests that need specific neighbor results override: mock.search.return_value = [[...]].
    """
    client = MagicMock()
    client.search.return_value = [[]]
    client.upsert.return_value = None
    return client


@pytest.fixture
def autoloop_config():
    """AutoLoopConfig with FAST settings for unit tests.

    Keeps algorithm semantics intact but compresses timing-sensitive parameters:
    - min_observations=5 (vs 20): allows small synthetic clusters to qualify
    - max_patches_per_run=1 (vs 5): single-patch tests stay deterministic
    - n_hands_validation=10 (vs 5000): A/B runs complete instantly
    - bootstrap_resamples=10 (vs 1000): CI computation stays fast
    All other values match CONTEXT.md Decision 5 defaults.
    """
    from src._config import AutoLoopConfig

    return AutoLoopConfig(
        enabled=True,
        tau_leak=0.3,
        min_observations=5,
        max_patches_per_run=1,
        n_hands_validation=10,
        bootstrap_resamples=10,
        ci_alpha=0.05,
    )


@pytest.fixture
def synthetic_cluster_key() -> str:
    """Fixed cluster key for E2E and unit tests requiring a stable BTN postflop cluster.

    Matches CONTEXT.md Decision 7 synthetic known-bad cluster fixture.
    Format: sorted hard_filter fields joined by '|' (cluster_key canonical form).
    """
    return "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
