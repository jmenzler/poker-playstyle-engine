"""External-service fixtures configured through TSDB_* and MILVUS_* variables."""

import os

import pytest


@pytest.fixture(scope="session")
def tsdb_dsn() -> str:
    """Build TimescaleDB DSN from env vars."""
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
    """Return Milvus auth token in ``root:password`` format.

    Milvus built-in auth requires ``user:password``. The .env stores only the
    raw password (MILVUS_TOKEN=<password>). We prepend ``root:`` here unless
    the value already contains ``:``.
    """
    raw = os.environ.get("MILVUS_TOKEN")
    if raw is None:
        return None
    return raw if ":" in raw else f"root:{raw}"
