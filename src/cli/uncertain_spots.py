"""uncertain-spots subcommand: list flagged-sparse observations.

Query: read-only SELECT against observations WHERE flagged_sparse = TRUE,
sorted by max_neighbor_distance DESC NULLS LAST. All filters are optional
and applied as parameterized psycopg v3 placeholders (no string interpolation).

Output: JSON array of objects to stdout, one element per row. Datetime fields
serialize to ISO 8601 via ``json.dumps(..., default=str)``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger
from src.db import timescale

log = get_logger("cli.uncertain_spots")


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from env vars (TSDB_PASSWORD is required)."""
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud if missing
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


def run(args: argparse.Namespace, *, _conn=None) -> int:
    """Execute the uncertain-spots query and print JSON to stdout.

    Args:
        args: parsed argparse.Namespace with .session, .limit, .street,
              .min_distance attributes.
        _conn: optional pre-built psycopg connection for test injection.
               Production callers never pass this.

    Returns:
        0 on success, non-zero on error.
    """
    conn = _conn if _conn is not None else timescale.connect(_tsdb_dsn_from_env())

    # Build parameterized SQL with optional filter clauses.
    # Base query — 6 columns per CONTEXT.md Decision 4 (LOCKED).
    sql = (
        "SELECT obs_id, cluster_key, ts, session_id, action_taken, max_neighbor_distance "
        "FROM observations "
        "WHERE flagged_sparse = TRUE"
    )
    params: list[object] = []

    if args.session is not None:
        sql += " AND session_id = %s"
        params.append(args.session)

    if args.street is not None:
        # cluster_key carries street_class as a token, e.g. "street_class=preflop|pot_type=srp"
        sql += " AND cluster_key LIKE %s"
        params.append(f"%street_class={args.street}%")

    if args.min_distance is not None:
        # NULL rows are excluded automatically: NULL >= X evaluates to UNKNOWN (false)
        sql += " AND max_neighbor_distance >= %s"
        params.append(args.min_distance)

    # Sort: max_neighbor_distance DESC NULLS LAST (per CONTEXT.md Decision 4 — LOCKED)
    sql += " ORDER BY max_neighbor_distance DESC NULLS LAST"
    sql += " LIMIT %s"
    params.append(args.limit)

    log.info(
        "cli.uncertain_spots.query",
        session=args.session,
        limit=args.limit,
        street=args.street,
        min_distance=args.min_distance,
    )

    with conn.cursor() as cur:
        cur.execute(sql, params)
        col_names = [desc[0] for desc in cur.description]
        raw_rows = cur.fetchall()

    rows = [dict(zip(col_names, row, strict=True)) for row in raw_rows]
    print(json.dumps(rows, indent=2, default=str))

    log.info("cli.uncertain_spots.done", row_count=len(rows))
    return 0


if __name__ == "__main__":
    import argparse as _ap

    _parser = _ap.ArgumentParser(description="List flagged-sparse observations.")
    _parser.add_argument("--session", type=str, default=None)
    _parser.add_argument("--limit", type=int, default=50)
    _parser.add_argument("--street", type=str, default=None, choices=["preflop", "postflop"])
    _parser.add_argument("--min-distance", type=float, default=None, dest="min_distance")
    sys.exit(run(_parser.parse_args()))
