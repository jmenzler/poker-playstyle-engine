"""Shared fixtures for Phase 4 (Labeling) tests.

Provides tsdb_dsn / milvus_uri / milvus_token for integration tests (PC-gated),
plus Phase-4-specific fixtures: sparse/dense Milvus mock clients and a
psycopg-shaped TimescaleDB mock connection.
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


# --- Mock Milvus clients ------------------------------------------------------


@pytest.fixture
def mock_milvus_client_sparse():
    """MagicMock Milvus client with neighbor distances > SPARSE_DIST_TAU.

    Simulates a sparse kNN neighborhood: 3 neighbors with distances [0.6, 0.7, 0.8].
    max_neighbor_distance=0.8, which is > SPARSE_DIST_TAU=0.5 -> flagged_sparse=True.
    n=3 >= SPARSE_N_MIN=3 so the flag is driven by distance alone.
    """
    client = MagicMock()
    client.search.return_value = [
        [
            {
                "distance": 0.6,
                "entity": {
                    "decision_id": "sp1",
                    "hero_action_type": "fold",
                    "confidence": 1.0,
                    "gto_score": 1.0,
                },
            },
            {
                "distance": 0.7,
                "entity": {
                    "decision_id": "sp2",
                    "hero_action_type": "call",
                    "confidence": 1.0,
                    "gto_score": 1.0,
                },
            },
            {
                "distance": 0.8,
                "entity": {
                    "decision_id": "sp3",
                    "hero_action_type": "fold",
                    "confidence": 1.0,
                    "gto_score": 1.0,
                },
            },
        ]
    ]
    return client


@pytest.fixture
def mock_milvus_client_dense():
    """MagicMock Milvus client with neighbor distances < SPARSE_DIST_TAU.

    Simulates a dense kNN neighborhood: 3 neighbors with distances [0.1, 0.2, 0.4].
    max_neighbor_distance=0.4, which is <= SPARSE_DIST_TAU=0.5 -> flagged_sparse=False.
    n=3 >= SPARSE_N_MIN=3 so no flag is set.
    """
    client = MagicMock()
    client.search.return_value = [
        [
            {
                "distance": 0.1,
                "entity": {
                    "decision_id": "dn1",
                    "hero_action_type": "call",
                    "confidence": 1.0,
                    "gto_score": 1.0,
                },
            },
            {
                "distance": 0.2,
                "entity": {
                    "decision_id": "dn2",
                    "hero_action_type": "fold",
                    "confidence": 1.0,
                    "gto_score": 1.0,
                },
            },
            {
                "distance": 0.4,
                "entity": {
                    "decision_id": "dn3",
                    "hero_action_type": "call",
                    "confidence": 1.0,
                    "gto_score": 1.0,
                },
            },
        ]
    ]
    return client


@pytest.fixture
def mock_tsdb_conn() -> MagicMock:
    """psycopg-shaped MagicMock connection for unit testing ObservationWriter.

    Supports the context-manager cursor protocol:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()
        conn.close()
    """
    conn = MagicMock()
    # cursor().__enter__ returns the cursor mock; __exit__ is a no-op
    mock_cur = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn
