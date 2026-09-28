"""Solver cache: persist and load solve results keyed by cluster_key.

Provides shared substrate for both eval tiers (D-09-6):
- Tier-1 (LOO action-accuracy) reads action_dist from the cache.
- Tier-2 (GTO exploitability) reads exploitability_pct from the cache.

One solve per cluster produces both values. This module persists/loads that
pair via the solver_cache hypertable (migration 015).

Schema (migration 018 adds decision_id):
  solver_cache(cluster_key, decision_id, action_dist JSONB, exploitability_pct DOUBLE,
    spot_features JSONB, solver_version TEXT, solved_at TIMESTAMPTZ);
  PK (cluster_key, solved_at).

Security: T-9-07 — all SQL uses parameterized %s placeholders. cluster_key and
action_dist are bound parameters, never interpolated into the query string.

Test injection: _tsdb_conn accepts a pre-built psycopg connection so unit tests
can pass a MagicMock without needing a real DB.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import msgspec

from src._log import get_logger
from src.db import timescale

log = get_logger("eval.solver_cache")


class SolverCacheEntry(msgspec.Struct, frozen=True, kw_only=True):
    """One row of the solver_cache hypertable.

    Attributes:
        cluster_key: Cluster identifier (same key used by the decision engine).
        decision_id: Per-DP observation identifier (migration 018); None for legacy rows.
        action_dist: 15-action canonical vocab distribution summing to 1.0±0.001.
        exploitability_pct: GTO exploitability in percent (Tier-2 metric).
        spot_features: Optional raw spot features used for the solve.
        solver_version: Tag for the solver binary or 'fixture' for hand-verified entries.
        solved_at: UTC timestamp of the solve (None when not yet persisted).
    """

    cluster_key: str
    action_dist: dict[str, float]
    exploitability_pct: float
    decision_id: str | None = None
    spot_features: dict[str, Any] | None = None
    solver_version: str | None = None
    solved_at: datetime | None = None


def persist_solve(entry: SolverCacheEntry, *, _tsdb_conn: Any = None) -> None:
    """Insert a SolverCacheEntry into the solver_cache hypertable.

    Uses ON CONFLICT (cluster_key, solved_at) DO NOTHING — idempotent for same
    cluster+timestamp pair. Production callers should set solver_version to track
    provenance (T-9-09).

    Args:
        entry: The SolverCacheEntry to persist.
        _tsdb_conn: Optional injected psycopg connection (tests). When None,
            opens a fresh connection via timescale.connect using the env-var DSN
            and closes it on exit.

    Raises:
        psycopg.Error: any underlying DB failure propagates.
    """
    own_conn = False
    conn = _tsdb_conn
    if conn is None:
        conn = timescale.connect(_tsdb_dsn_from_env())
        own_conn = True
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO solver_cache "
                "    (cluster_key, decision_id, action_dist, exploitability_pct, spot_features, solver_version, solved_at) "
                "VALUES (%s, %s, %s::jsonb, %s, %s::jsonb, %s, COALESCE(%s, now())) "
                "ON CONFLICT (cluster_key, solved_at) DO NOTHING",
                (
                    entry.cluster_key,
                    entry.decision_id,
                    json.dumps(entry.action_dist),
                    entry.exploitability_pct,
                    json.dumps(entry.spot_features) if entry.spot_features is not None else None,
                    entry.solver_version,
                    entry.solved_at,
                ),
            )
        log.info("eval.solver_cache.persisted", cluster_key=entry.cluster_key, decision_id=entry.decision_id)
    finally:
        if own_conn:
            conn.close()


def load_solve(cluster_key: str, *, _tsdb_conn: Any = None) -> SolverCacheEntry | None:
    """Load the most-recent cached solve for a cluster_key.

    Args:
        cluster_key: Cluster identifier to look up.
        _tsdb_conn: Optional injected psycopg connection (tests).

    Returns:
        SolverCacheEntry with the latest solve, or None if no row exists.
    """
    own_conn = False
    conn = _tsdb_conn
    if conn is None:
        conn = timescale.connect(_tsdb_dsn_from_env())
        own_conn = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT cluster_key, action_dist, exploitability_pct, spot_features, solver_version, solved_at "
                "FROM solver_cache WHERE cluster_key = %s ORDER BY solved_at DESC LIMIT 1",
                (cluster_key,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return SolverCacheEntry(
            cluster_key=str(row[0]),
            action_dist=row[1] if isinstance(row[1], dict) else json.loads(row[1]),
            exploitability_pct=float(row[2]),
            spot_features=row[3] if row[3] is not None else None,
            solver_version=row[4],
            solved_at=row[5],
        )
    finally:
        if own_conn:
            conn.close()


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from env vars (TSDB_PASSWORD required)."""
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
