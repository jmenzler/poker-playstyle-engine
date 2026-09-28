"""FastAPI dependencies — DB connection providers + StudyConfig.

Wraps existing src/db/timescale.connect + src/db/milvus.connect with proper
lifespan management. uvicorn ``--workers 1`` is mandatory (D-02 + Pitfall 6)
so module-level singletons (the cached config + the Milvus client) are safe.

Per-request connections:
    * get_tsdb yields a fresh psycopg connection per request, closing on
      cleanup. Matches the ``_tsdb_conn`` injection contract that every
      ``src/study/*`` and ``src/eval/*`` backend function honors.

Module-level singletons:
    * get_config caches the StudyConfig load (frozen msgspec.Struct).
    * get_milvus lazily constructs a single MilvusClient; pymilvus 3.x is
      designed for long-lived clients (gRPC channel reuse).

DSN/URI sourcing:
    Both ``_tsdb_dsn_from_env`` and ``_milvus_uri_from_env`` mirror the
    pattern in src/study/*.py (TSDB_HOST / TSDB_PASSWORD / MILVUS_HOST env
    vars). Fail-loud on missing TSDB_PASSWORD (KeyError); silent fallback
    is forbidden per ERR-01.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src._config import StudyConfig, load_study_config
from src.db import milvus as milvus_db
from src.db import timescale

_STUDY_CONFIG_PATH = Path("config/study.toml")

# --- Cached singletons -------------------------------------------------------

_cached_config: StudyConfig | None = None
_milvus_client: Any = None


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from env vars (mirrors src/study/*.py)."""
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud per ERR-01
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


def _milvus_uri_from_env() -> str:
    """Build a Milvus URI from env vars."""
    host = os.environ.get("MILVUS_HOST", "127.0.0.1")
    port = os.environ.get("MILVUS_PORT", "51530")
    return f"http://{host}:{port}"


def _milvus_token_from_env() -> str | None:
    raw = os.environ.get("MILVUS_TOKEN")
    if raw is None:
        return None
    return raw if ":" in raw else f"root:{raw}"


# --- FastAPI dependency providers --------------------------------------------


def get_config() -> StudyConfig:
    """Return the StudyConfig (loaded once, cached for process lifetime).

    Falls back to ``StudyConfig()`` defaults only when ``config/study.toml`` is
    absent — tests and dev environments without a deployed TOML still get a
    valid config (loopback-bound, sequential solver). A *present* but malformed
    TOML must fail loud: ConfigError propagates rather than masking the bad file
    behind defaults (ERR-01).
    """
    global _cached_config
    if _cached_config is None:
        if not _STUDY_CONFIG_PATH.exists():
            _cached_config = StudyConfig()
        else:
            _cached_config = load_study_config(_STUDY_CONFIG_PATH)
    return _cached_config


def get_tsdb() -> Iterator[Any]:
    """Yield a TSDB connection per request. Closes on cleanup.

    This is the FastAPI Depends-friendly generator form; the underlying
    ``timescale.connect`` is the only sanctioned psycopg connect path (ERR-01).
    """
    conn = timescale.connect(_tsdb_dsn_from_env())
    try:
        yield conn
    finally:
        # autocommit=False: roll back any uncommitted/aborted txn so an
        # uncommitted direct write is discarded deterministically rather than
        # silently lost at close. Never raise from a yield-fixture teardown.
        with contextlib.suppress(Exception):
            conn.rollback()
        with contextlib.suppress(Exception):
            conn.close()


def get_milvus() -> Any:
    """Return a process-wide pymilvus MilvusClient (lazy).

    pymilvus 3.x clients are designed for long-lived reuse (gRPC channel),
    and D-02 mandates a single uvicorn worker — so a module-level singleton
    is correct here. Tests inject their own client via Depends overrides.
    """
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = milvus_db.connect(_milvus_uri_from_env(), token=_milvus_token_from_env())
    return _milvus_client
