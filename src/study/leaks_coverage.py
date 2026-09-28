"""Type 1 leaks — coverage gaps (sparse kNN).  D-04.

Mirrors ``src/cli/uncertain_spots.py`` SQL pattern but exposed as a backend
function callable from both CLI (``leaks --type coverage``) and FastAPI
(``/api/leaks?type=coverage``).

Notes on schema (Phase 1 migrations + 008):
    The ``observations`` hypertable has columns ``flagged_sparse`` (BOOL) and
    ``max_neighbor_distance`` (REAL, nullable) per migration 008.  There is NO
    ``street`` column on this table — the planner's example SQL referenced one,
    but the on-disk schema does not.  This module exposes ``min_distance``
    and ``session_id`` filters only; ``street`` is dropped (Rule 3 deviation).
"""

from __future__ import annotations

import contextlib
from typing import Any

from src._log import get_logger
from src.db import timescale

log = get_logger("study.leaks_coverage")


def rank_coverage_gaps(
    *,
    limit: int = 20,
    session_id: str | None = None,
    min_distance: float | None = None,
    _tsdb_conn: Any = None,
) -> list[dict]:
    """Sparse-flagged observations ranked by ``max_neighbor_distance DESC NULLS LAST``.

    Args:
        limit:         Cap on result rows (CLI default = 20).
        session_id:    Optional session filter.
        min_distance:  Optional floor on ``max_neighbor_distance``.
        _tsdb_conn:    Test-injection hook; if ``None`` open via
                       ``timescale.connect(_tsdb_dsn_from_env())``.

    Returns:
        ``list[dict]`` with keys
        ``obs_id, cluster_key, ts, session_id, action_taken, max_neighbor_distance``.
    """
    log.info("study.leaks_coverage.started", limit=limit, session_id=session_id)
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        sql = (
            "SELECT obs_id, cluster_key, ts, session_id, action_taken, max_neighbor_distance "
            "FROM observations WHERE flagged_sparse = TRUE"
        )
        params: list[object] = []
        if session_id is not None:
            sql += " AND session_id = %s"
            params.append(session_id)
        if min_distance is not None:
            sql += " AND max_neighbor_distance > %s"
            params.append(min_distance)
        sql += " ORDER BY max_neighbor_distance DESC NULLS LAST LIMIT %s"
        params.append(limit)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            col = [d[0] for d in cur.description]
            rows = cur.fetchall()
        result = [dict(zip(col, r, strict=True)) for r in rows]
        log.info("study.leaks_coverage.complete", n_rows=len(result))
        return result
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from env vars (mirrors leak_detector.py)."""
    import os

    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
