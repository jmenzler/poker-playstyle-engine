# rot-allow-file
"""CLI-01 / FastAPI POST /api/ingest — HH ingest pipeline orchestrator (D-10).

- Incremental by default: report watermark = SELECT MAX(ts) FROM observations
  BEFORE and AFTER invoking the pipeline.
- --rebuild path: full pipeline (`tools/phase2_pipeline.py --all --rebuild`).
- Reuses frozen z-score manifests (do NOT refit).
- Milvus upsert idempotent by observation_id PK (Phase 2 lock) — operative
  idempotency layer for re-ingest (D-07-11a). TSDB observations has no
  UNIQUE on obs_id (would be illegal on a hypertable without ts; the PK
  (obs_id, ts) already covers the legal shape).

NOTE: tools/phase2_pipeline.py writes to Milvus only, NOT observations. On a
fresh PC, MAX(ts) returns NULL and watermark_before == watermark_after == None
is the expected steady state. Watermark is decorative for the HH-ingest path
(D-07-2, D-07-11e); true re-ingest dedup is the Milvus PK on deterministic
decision_id = f"{hand_id}_dp{decision_idx}" (tools/extract_decisions.py:266).

tools/phase2_pipeline.py does NOT currently accept a --watermark flag.
ingest_incremental therefore reports the watermark for operator visibility but
relies on Milvus upsert idempotency to skip already-processed rows. Adding
`--watermark N` to the pipeline is v2 scope (DEF-06-05-01).
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any

from src._log import get_logger
from src.db import timescale

log = get_logger("study.ingest")


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from environment variables (mirrors patch_engine)."""
    import os

    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess:
    """Default pipeline runner — subprocess.run with a 1-hour wall clock cap."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=3600)


def _parse_count(stdout: str, key: str) -> int | None:
    """Parse `key=N` (or JSON `"key": N`) from structlog or plain-text pipeline output.

    Returns None when no parseable count is found (caller substitutes 0).
    """
    for line in stdout.splitlines():
        if key not in line:
            continue
        # Try structlog JSON first
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and key in obj:
                return int(obj[key])
        except (ValueError, TypeError):
            pass
        # Fall back to substring parse: `key=N` anywhere in the line
        marker = key + "="
        if marker in line:
            tail = line.split(marker, 1)[1]
            num = tail.split()[0].rstrip(",")
            try:
                return int(num)
            except ValueError:
                continue
    return None


def _parse_skipped_reasons(stdout: str) -> dict[str, int]:
    """Parse skip reasons (straddle, run_it_twice) from pipeline log output."""
    reasons: dict[str, int] = {}
    for r in ("straddle", "run_it_twice"):
        n = _parse_count(stdout, f"skipped_{r}")
        if n is not None:
            reasons[r] = n
    return reasons


def ingest_incremental(
    *,
    _tsdb_conn: Any = None,
    _runner: Any = None,
) -> dict:
    """Run incremental ingest from current watermark to end of HM source.

    Returns:
        {hands_processed, hands_skipped, elapsed_s, watermark_before, watermark_after}.
    """
    start = time.monotonic()
    log.info("study.ingest.incremental.started")
    own_conn = False
    conn = _tsdb_conn
    if conn is None:
        conn = timescale.connect(_tsdb_dsn_from_env())
        own_conn = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(ts) FROM observations")
            wm_before = cur.fetchone()[0]

        # tools/phase2_pipeline.py does NOT accept --watermark today; relying on
        # Milvus upsert idempotency by decision_id PK. See module docstring.
        result = (_runner or _default_runner)(["python", "-m", "tools.phase2_pipeline", "--all"])

        with conn.cursor() as cur:
            cur.execute("SELECT MAX(ts) FROM observations")
            wm_after = cur.fetchone()[0]

        elapsed = time.monotonic() - start
        out = {
            "hands_processed": _parse_count(result.stdout, "hands_processed") or 0,
            "hands_skipped": _parse_skipped_reasons(result.stdout),
            "elapsed_s": elapsed,
            "watermark_before": wm_before,
            "watermark_after": wm_after,
        }
        log.info(
            "study.ingest.incremental.complete",
            hands_processed=out["hands_processed"],
            watermark_before=wm_before,
            watermark_after=wm_after,
            elapsed_s=elapsed,
        )
        return out
    finally:
        if own_conn:
            conn.close()


def ingest_rebuild(
    *,
    _tsdb_conn: Any = None,
    _runner: Any = None,
) -> dict:
    """Run full rebuild pipeline. Caller-confirmed destructive action.

    Returns:
        {hands_processed, hands_skipped, elapsed_s, watermark_before=None,
         watermark_after=None, rebuild=True}.
    """
    start = time.monotonic()
    log.info("study.ingest.rebuild.started")
    result = (_runner or _default_runner)(["python", "-m", "tools.phase2_pipeline", "--all", "--rebuild"])
    elapsed = time.monotonic() - start
    out = {
        "hands_processed": _parse_count(result.stdout, "hands_processed") or 0,
        "hands_skipped": _parse_skipped_reasons(result.stdout),
        "elapsed_s": elapsed,
        "watermark_before": None,
        "watermark_after": None,
        "rebuild": True,
    }
    log.info(
        "study.ingest.rebuild.complete",
        hands_processed=out["hands_processed"],
        elapsed_s=elapsed,
    )
    return out
