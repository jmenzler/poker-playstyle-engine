"""Shared fixtures for Phase 6 (Study Tool) tests.

Mirrors tests/phase5/conftest.py structure. Provides:

- Integration fixtures (PC-gated, scope=session):
    tsdb_dsn, milvus_uri, milvus_token — env-var-driven DSN/URI/token builders
- Live-DB fixtures (skip if env missing):
    tsdb_conn        — psycopg connection wrapped in a transaction that rolls back
    milvus_client    — pymilvus MilvusClient against the configured Milvus instance
- App fixtures:
    api_client       — FastAPI TestClient; uses pytest.importorskip("src.api.main")
                       so tests skip cleanly during early Phase 6 waves before
                       src/api/main.py exists.
- Test-data fixtures:
    sample_cluster_keys — list[str] of realistic cluster_key strings
    study_config        — StudyConfig loaded from config/study.toml

Conventions:
- All live-DB fixtures `yield` and clean up in teardown.
- DB fixtures use transactions that rollback after the test (no test data persists
  across tests; safe to run against the prod TimescaleDB instance).
- The `integration` marker (registered in pyproject.toml) is required on tests
  that hit the real TimescaleDB / Milvus stack.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# --- Integration fixtures (PC-gated) -----------------------------------------


@pytest.fixture(scope="session")
def tsdb_dsn() -> str:
    """Build TimescaleDB DSN from env vars (prod DB, not test DB).

    Mirrors tests/phase5/conftest.py — fail loud if TSDB_PASSWORD is missing so
    integration tests never silently fall back to a local DB with a wrong password.
    """
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud if missing
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


@pytest.fixture(scope="session")
def milvus_uri() -> str:
    """Build Milvus URI from env vars."""
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


# --- Live-DB fixtures (skip if env missing) ----------------------------------


@pytest.fixture
def tsdb_conn():
    """Live TimescaleDB connection wrapped in a transaction that always rolls back.

    Skips the test if TSDB_HOST or TSDB_PASSWORD env vars are unset. Uses
    src.db.timescale.connect (the only sanctioned psycopg connect path; raises
    DBConnectError with redacted DSN on failure).

    Teardown rolls back any mutations + closes the connection — tests can INSERT
    freely without polluting the prod database.
    """
    if "TSDB_HOST" not in os.environ or "TSDB_PASSWORD" not in os.environ:
        pytest.skip("TSDB env vars not set (TSDB_HOST / TSDB_PASSWORD)")

    from src.db.timescale import connect

    host = os.environ["TSDB_HOST"]
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]
    dsn = f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"

    conn = connect(dsn)
    try:
        # psycopg3 transaction() context manager: rolls back on exit unless commit() called.
        with conn.transaction() as txn:
            yield conn
            # Explicit rollback after the test completes — even on success — so
            # no test data persists. The transaction() manager would commit on
            # clean exit by default.
            txn.connection.rollback()
    finally:
        conn.close()


@pytest.fixture
def milvus_client(milvus_uri: str, milvus_token: str | None):
    """Live pymilvus MilvusClient.

    Skips when MILVUS_HOST is unset. Uses src.db.milvus.connect (forces a
    reachability probe; raises DBConnectError on transport failure). Closed
    deterministically in teardown.
    """
    if "MILVUS_HOST" not in os.environ:
        pytest.skip("MILVUS_HOST env var not set")

    from src.db.milvus import connect

    client = connect(milvus_uri, token=milvus_token)
    try:
        yield client
    finally:
        # MilvusClient.close() is idempotent on pymilvus 3.x; safe to call.
        try:
            client.close()
        except Exception:
            pass


# --- App fixtures ------------------------------------------------------------


@pytest.fixture
def api_client():
    """FastAPI TestClient wrapping src.api.main.app.

    Uses pytest.importorskip so tests calling this fixture skip cleanly during
    early Phase 6 waves before src/api/main.py exists. Once Plan 06-03
    (FastAPI scaffolding) lands, all api_client-using tests will run.
    """
    pytest.importorskip("src.api.main")
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    from src.api.main import app  # type: ignore[import-not-found]

    return TestClient(app)


# --- Test-data fixtures ------------------------------------------------------


@pytest.fixture
def sample_cluster_keys() -> list[str]:
    """Realistic cluster_key strings for parameterized tests.

    Format matches what is stored in `strategy_nodes.cluster_key` and what the
    Canonicalizer.encode() returns (sorted `key=value` segments joined by `|`).
    Covers preflop, flop, and turn streets so downstream tests can exercise
    street-specific logic without re-deriving keys.
    """
    return [
        "action_history=cc|hero_pos_rel=ip|n_players=2|pot_type=srp|street_class=flop|texture=dry_rainbow",
        "action_history=open|hero_pos_rel=btn|n_players=6|pot_type=open|street_class=preflop",
        "action_history=ccb50c|hero_pos_rel=oop|n_players=2|pot_type=3bp|street_class=turn|texture=wet_two_tone",
    ]


@pytest.fixture
def study_config():
    """StudyConfig loaded from config/study.toml in the repo root.

    Tests that need to override defaults (e.g., a small ascii_sparkline_width)
    construct a fresh StudyConfig(**overrides) directly rather than mutating
    this fixture (msgspec.Struct is frozen).
    """
    from src._config import load_study_config

    return load_study_config(Path("config/study.toml"))
