"""metrics.sink: session-level metrics persistence (METR-01..03).

Writes ev_loss / novel_spot_rate / latency_p50 / latency_p99 rows into the
``metrics`` hypertable after each RECORD session ends.

Contract (from CONTEXT.md Decision 6):
    - METR-01: one ev_loss row per touched cluster_key in the session.
    - METR-02: one novel_spot_rate row per session (cluster_key=NULL).
    - METR-03: one latency_p50 + one latency_p99 row per session (cluster_key=NULL),
        derived from the harness per-decision timing buffer.

Reuse:
    Reuses ``src.metrics.ev_loss.ev_loss()`` pure function unchanged (Phase 4).
    Clusters with < k=10 observations raise NoStrategyError — their ev_loss row
    is skipped with a warning; the rest of the flush continues.

Idempotency (Phase 7 / WARN 1 / D-07-11c):
    Idempotent across re-flushes via ON CONFLICT (session_id, metric_name,
    cluster_key, ts) DO NOTHING + ts pinned to ``session_started_at`` (NOT
    ``datetime.now(UTC)``). Re-running for the same session_id with the same
    session_started_at is a no-op — duplicate rows are silently dropped by the
    metrics_unique_per_session UNIQUE constraint (migration 009). Callers MUST
    pass the same ``session_started_at`` on every re-flush of the same session.

Test injection:
    _tsdb_conn and _milvus kwargs accept pre-built connections for unit tests.
    Production callers never pass these; connections are opened from env vars.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any

import numpy as np

from src._errors import NoStrategyError
from src._log import get_logger
from src.db import milvus as milvus_db
from src.db import timescale
from src.metrics.ev_loss import ev_loss

log = get_logger("metrics.sink")

# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_INSERT_METRIC_SQL = (
    "INSERT INTO metrics (metric_id, session_id, cluster_key, metric_name, value, ts) "
    "VALUES (%s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (session_id, metric_name, cluster_key, ts) DO NOTHING"
)

_TOUCHED_CLUSTERS_SQL = "SELECT DISTINCT cluster_key FROM observations WHERE session_id = %s"

_NOVEL_SPOT_RATE_SQL = (
    "SELECT "
    "    COUNT(*) FILTER (WHERE flagged_sparse = TRUE)::float / NULLIF(COUNT(*), 0) "
    "    AS novel_rate "
    "FROM observations WHERE session_id = %s"
)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def flush_session_metrics(
    session_id: str,
    timing_buffer: list[float],
    *,
    session_started_at: datetime,
    _tsdb_conn: Any = None,
    _milvus: Any = None,
) -> int:
    """Write all per-session metrics rows to the metrics hypertable.

    Called by tools/run_sim_session.py and tests/phase5/test_autoloop_e2e.py
    after a RECORD session ends. Returns the count of metric rows written.

    Idempotent across re-flushes via ON CONFLICT (session_id, metric_name,
    cluster_key, ts) DO NOTHING + ts pinned to ``session_started_at``
    (D-07-11c — supersedes D-07-6). Re-running for the same session_id with
    the same ``session_started_at`` produces zero new rows; the metrics_unique_per_session
    UNIQUE constraint (migration 009) silently absorbs the conflicts.

    Args:
        session_id: the session whose observations are being summarized.
        timing_buffer: list of per-decision elapsed-seconds floats from
            run_record_session()'s summary['latency_buffer']. May be empty
            (e.g., if the session was aborted before any decisions). When
            empty, latency_p50/p99 are skipped (METR-03 unsatisfied for
            empty sessions — caller must ensure a real session ran).
        session_started_at: deterministic session start timestamp (UTC).
            REQUIRED kw-only argument (no default — a default of datetime.now()
            would defeat ON CONFLICT across re-flushes; see D-07-11c). Every
            metric row inserted by this call uses this exact value for ``ts``.
        _tsdb_conn: test-injection hook. Pre-built psycopg connection.
            Production callers never pass this.
        _milvus: test-injection hook. Pre-built MilvusClient.
            Production callers never pass this.

    Returns:
        Number of metric rows written. Lower bound = 1 (novel_spot_rate);
        typical = 1 + N_clusters + 2 (novel_spot_rate + N ev_loss + 2 latency).
        On a re-flush, rows_written reflects local INSERT attempts; ON CONFLICT
        silently drops them at the DB layer (re-flush is a no-op).
    """
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    client = _milvus if _milvus is not None else milvus_db.connect_from_env()

    # Pin ts for every INSERT in this flush so re-runs at later wall-clock
    # times hit ON CONFLICT and become no-ops (D-07-11c).
    ts_pin = session_started_at
    novel_rate = 0.0  # initialize for use in log.info at end

    # --- Read phase (outside transaction — read-only, no rollback needed) ---

    # 1. Compute ev_loss per touched cluster_key (METR-01)
    touched = _touched_clusters(conn, session_id)
    ev_loss_rows: list[tuple[str, float]] = []
    for cluster_key in touched:
        try:
            kl = ev_loss(
                cluster_key,
                session_id=session_id,
                _tsdb_conn=conn,
                _milvus=client,
            )
        except NoStrategyError as exc:
            log.warning(
                "metrics.sink.skip_cluster",
                cluster_key=cluster_key,
                session_id=session_id,
                detail=str(exc),
            )
            continue
        ev_loss_rows.append((cluster_key, kl))

    # 2. Compute novel_spot_rate (METR-02)
    with conn.cursor() as cur:
        cur.execute(_NOVEL_SPOT_RATE_SQL, (session_id,))
        row = cur.fetchone()
        novel_rate = float(row[0]) if row and row[0] is not None else 0.0

    # 3. Compute latency percentiles (METR-03) — CPU only, no DB read
    latency_rows: list[tuple[str, float]] = []
    if timing_buffer:
        arr = np.array(timing_buffer, dtype=np.float64)
        latency_rows = [
            ("latency_p50", float(np.percentile(arr, 50))),
            ("latency_p99", float(np.percentile(arr, 99))),
        ]
    else:
        log.warning("metrics.sink.empty_timing_buffer", session_id=session_id)

    # --- Write phase (single atomic transaction — WR-05: prevents partial rows) ---
    rows_written = 0
    with conn.transaction(), conn.cursor() as cur:
        for cluster_key, kl in ev_loss_rows:
            cur.execute(
                _INSERT_METRIC_SQL,
                (str(uuid.uuid4()), session_id, cluster_key, "ev_loss", kl, ts_pin),
            )
            rows_written += 1

        cur.execute(
            _INSERT_METRIC_SQL,
            (str(uuid.uuid4()), session_id, None, "novel_spot_rate", novel_rate, ts_pin),
        )
        rows_written += 1

        for metric_name, value in latency_rows:
            cur.execute(
                _INSERT_METRIC_SQL,
                (str(uuid.uuid4()), session_id, None, metric_name, value, ts_pin),
            )
            rows_written += 1

    log.info(
        "metrics.sink.flush_done",
        session_id=session_id,
        rows_written=rows_written,
        clusters_touched=len(touched),
        novel_rate=round(novel_rate, 4),
    )
    return rows_written


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _touched_clusters(conn: Any, session_id: str) -> list[str]:
    """Return list of distinct cluster_keys touched in a session."""
    with conn.cursor() as cur:
        cur.execute(_TOUCHED_CLUSTERS_SQL, (session_id,))
        return [row[0] for row in cur.fetchall()]


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from environment variables.

    TSDB_PASSWORD is required; all other vars have sensible defaults.
    Mirrors pattern from src/metrics/ev_loss.py.
    """
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


def _milvus_uri_from_env() -> str:
    """Build a Milvus URI from environment variables."""
    host = os.environ["MILVUS_HOST"]
    port = os.environ.get("MILVUS_PORT", "51530")
    return f"http://{host}:{port}"
