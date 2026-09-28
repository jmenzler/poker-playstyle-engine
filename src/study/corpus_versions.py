"""Corpus version registry — maps discrete versions to epoch cutoff timestamps.

A version is the user-facing handle; ``cutoff_ts`` (INT epoch) is what the engine
pins on (``added_at <= cutoff_ts``). Mirrors the ``study/patches.py`` idiom.
"""

from __future__ import annotations

import contextlib
import os
from datetime import UTC, datetime
from typing import Any

from src._log import get_logger
from src.db import timescale

log = get_logger("study.corpus_versions")


def snapshot(label: str, source: str = "manual", *, _tsdb_conn: Any = None) -> dict:
    """Stamp the current moment as a new corpus version.

    Returns a dict with the new ``version`` and its epoch ``cutoff_ts``.
    ``source`` must satisfy the migration CHECK (manual/autoloop/solver).
    """
    cutoff_ts = int(datetime.now(tz=UTC).timestamp())
    log.info("study.corpus_versions.snapshot.started", label=label, source=source, cutoff_ts=cutoff_ts)
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO corpus_versions (cutoff_ts, source, label) "
                "VALUES (%s, %s, %s) RETURNING version, cutoff_ts",
                (cutoff_ts, source, label),
            )
            col = [d[0] for d in cur.description]
            row = cur.fetchone()
        result = dict(zip(col, row, strict=True))
        log.info("study.corpus_versions.snapshot.complete", version=result["version"])
        return result
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def list_versions(*, _tsdb_conn: Any = None) -> list[dict]:
    """Return all registered versions, newest first."""
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT version, cutoff_ts, source, label, created_at "
                "FROM corpus_versions ORDER BY version DESC"
            )
            col = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [dict(zip(col, r, strict=True)) for r in rows]
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def resolve_cutoff_ts(version: int, *, _tsdb_conn: Any = None) -> int:
    """Resolve a version to its epoch cutoff; raise ``ValueError`` if unknown."""
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT cutoff_ts FROM corpus_versions WHERE version = %s", (version,))
            row = cur.fetchone()
        if row is None:
            raise ValueError(f"unknown corpus version: {version}")
        return int(row[0])
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def _tsdb_dsn_from_env() -> str:
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
