"""Leak-suppression writer.  D-NEW-28.

Suppressions hide ``cluster_key`` rows from future Strategy Leaks ranks.
Append-only; reversible via ``UPDATE active=FALSE`` (``unsuppress``).

NOTE: This is a NEW write path that does NOT go through PatchEngine because
it doesn't touch ``strategy_nodes``.  The pre-commit hook (which forbids
direct strategy_nodes writes outside ``src/patch_engine.py``) is therefore
not violated.  Documented here explicitly per 06-PATTERNS.md landmines.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Any

from src._log import get_logger
from src.db import timescale

log = get_logger("study.suppressions")


def suppress(cluster_key: str, reason: str = "", *, _tsdb_conn: Any = None) -> str:
    """Insert a new active suppression row.

    Args:
        cluster_key:  Cluster to suppress.
        reason:       Free-text operator rationale (logged + persisted).
        _tsdb_conn:   Test-injection hook.

    Returns:
        ``suppression_id`` (UUID4 string canonical form, length 36).
    """
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    sid = str(uuid.uuid4())
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO leak_suppressions (suppression_id, cluster_key, reason, active) "
                "VALUES (%s, %s, %s, TRUE)",
                (sid, cluster_key, reason),
            )
        log.info("study.suppressions.suppress", cluster_key=cluster_key, suppression_id=sid)
        return sid
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def unsuppress(cluster_key: str, *, _tsdb_conn: Any = None) -> int:
    """Deactivate ALL active suppressions for ``cluster_key``.

    Returns:
        Number of rows updated (``cur.rowcount``).
    """
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "UPDATE leak_suppressions SET active = FALSE WHERE cluster_key = %s AND active = TRUE",
                (cluster_key,),
            )
            n = cur.rowcount
        log.info("study.suppressions.unsuppress", cluster_key=cluster_key, n_updated=n)
        return n
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def is_suppressed(cluster_key: str, *, _tsdb_conn: Any = None) -> bool:
    """Return ``True`` iff an active suppression exists for ``cluster_key``."""
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM leak_suppressions WHERE cluster_key = %s AND active = TRUE LIMIT 1",
                (cluster_key,),
            )
            return cur.fetchone() is not None
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from env vars (mirrors leak_detector.py)."""
    import os

    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
