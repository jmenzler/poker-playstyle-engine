"""CLI: run a deterministic RECORD-mode sim session.

Usage:
    python tools/run_sim_session.py --session-seed 42 --n-hands 1000
    python tools/run_sim_session.py --session-seed 42 --n-hands 30000 \\
        --session-id bench-001

An explicit --session-id that already has observations is rejected (SIM-06: a
re-run would double-write every row). Pass --force to delete the prior rows
first. Omit --session-id for a fresh UUID4 (no collision possible).

Environment:
    TSDB_HOST/TSDB_PORT/TSDB_DB/TSDB_USER/TSDB_PASSWORD  TimescaleDB connection
    MILVUS_HOST/MILVUS_PORT/MILVUS_TOKEN                 Milvus connection
    PREFLOP_MANIFEST/POSTFLOP_MANIFEST (optional)        z-score manifest paths
    LOG_LEVEL (optional)                                 default INFO
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src._log import configure_logging, get_logger
from src.db import timescale
from src.decision_engine.engine import engine_from_env
from src.sim import ObservationWriter, SimAdapter, run_record_session

log = get_logger("tools.run_sim_session")


class SessionIdCollisionError(RuntimeError):
    """Raised when an explicit --session-id already has observations and --force is absent."""


def _guard_explicit_session_id(conn, session_id: str, *, force: bool) -> None:
    """Fail loudly (SIM-06) if ``session_id`` already has observation rows.

    Re-running an explicit --session-id double-writes every observation (the
    INSERT has no ON CONFLICT and decision_ids are namespaced by session_id, not
    seed). With ``force`` the prior rows are DELETEd first; otherwise raise.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM observations WHERE session_id=%s LIMIT 1", (session_id,))
        exists = cur.fetchone() is not None
    if not exists:
        return
    if not force:
        raise SessionIdCollisionError(
            f"session_id {session_id!r} already has observations; re-running would "
            f"double-write every row. Pass --force to delete the prior rows first, "
            f"or use a fresh --session-id."
        )
    with conn.cursor() as cur:
        cur.execute("DELETE FROM observations WHERE session_id=%s", (session_id,))
        deleted = cur.rowcount
    conn.commit()
    log.warning(
        "tools.run_sim_session.force_deleted_prior_observations",
        session_id=session_id,
        deleted_rows=deleted,
    )


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from env vars (TSDB_PASSWORD is required).

    Mirrors the shape used by tests/integration/conftest.py.
    """
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud if missing
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a deterministic RECORD-mode sim session.",
    )
    ap.add_argument("--session-seed", type=int, required=True, help="top-level RNG seed")
    ap.add_argument("--n-hands", type=int, required=True, help="number of hands to play")
    ap.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="explicit session_id (defaults to a fresh UUID4)",
    )
    ap.add_argument(
        "--tsdb-dsn",
        type=str,
        default=None,
        help="TimescaleDB DSN; defaults to env vars (TSDB_HOST/PORT/DB/USER/PASSWORD)",
    )
    ap.add_argument(
        "--preflop-manifest",
        type=Path,
        default=Path("tools/zscore_preflop.json"),
        help="z-score manifest for preflop_decisions",
    )
    ap.add_argument(
        "--postflop-manifest",
        type=Path,
        default=Path("tools/zscore_postflop.json"),
        help="z-score manifest for postflop_decisions",
    )
    ap.add_argument("--batch-size", type=int, default=500, help="ObservationWriter batch size")
    ap.add_argument("--engine-k", type=int, default=10, help="kNN limit (default 10)")
    ap.add_argument(
        "--force",
        action="store_true",
        help="with an explicit --session-id, DELETE existing observations for it before writing",
    )
    args = ap.parse_args()

    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

    # Resolve session_id BEFORE constructing the writer so that the writer's
    # session_id matches the value the harness uses for every observation row.
    # Without this, --session-id None would cause the writer to be tagged
    # "pending" while observations are written under a different uuid generated
    # internally by run_record_session.
    session_id = args.session_id
    explicit_session_id = session_id is not None
    if session_id is None:
        session_id = str(uuid.uuid4())
        log.info("tools.run_sim_session.assigned_session_id", session_id=session_id)

    dsn = args.tsdb_dsn or _tsdb_dsn_from_env()

    # SIM-06: an explicit, reused --session-id would silently double-write every
    # observation. Guard before any write. Auto-uuid sessions are unique by
    # construction and skip the check.
    if explicit_session_id:
        guard_conn = timescale.connect(dsn)
        try:
            _guard_explicit_session_id(guard_conn, session_id, force=args.force)
        except SessionIdCollisionError as exc:
            log.error("tools.run_sim_session.session_id_collision", session_id=session_id, error=str(exc))
            print(str(exc), file=sys.stderr)
            return 1
        finally:
            guard_conn.close()

    # Capture session start clock BEFORE the sim loop. Threaded into
    # flush_session_metrics as the pinned ts so re-flushes of the same session
    # are idempotent at the metrics_unique_per_session UNIQUE level (D-07-11c).
    session_started_at = datetime.now(UTC)

    engine = engine_from_env(
        preflop_manifest=args.preflop_manifest,
        postflop_manifest=args.postflop_manifest,
        k=args.engine_k,
        rng_seed=args.session_seed,
    )
    adapter = SimAdapter()

    with ObservationWriter(dsn, session_id, batch_size=args.batch_size) as writer:
        summary = run_record_session(
            adapter,
            engine,
            writer,
            session_seed=args.session_seed,
            n_hands=args.n_hands,
            session_id=session_id,
        )
    # writer.close() is called on context manager exit above.
    # Phase 5: persist session-level metrics (METR-01..03) after observations are written.
    from src.metrics.sink import flush_session_metrics

    try:
        n_metrics = flush_session_metrics(
            summary["session_id"],
            summary.get("latency_buffer", []),
            session_started_at=session_started_at,
        )
        log.info(
            "autoloop.session_metrics_flushed",
            session_id=summary["session_id"],
            metrics_rows=n_metrics,
        )
    except Exception as exc:
        # Metrics flush failure is non-fatal — observations are already written.
        log.warning(
            "autoloop.metrics_flush_failed",
            session_id=summary["session_id"],
            error=str(exc),
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
