"""CLI-01 — `poker-engine ingest` shim. D-10.

Default mode is incremental upsert (delegates to
``src.study.ingest.ingest_incremental``); ``--rebuild`` triggers
``ingest_rebuild`` (full pipeline replay).

Exit codes:
    0 — success
    2 — unexpected error

Help text:
    CLI-01: Run HH ingest pipeline (incremental by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger

log = get_logger("cli.ingest")


def run(args: argparse.Namespace, *, _tsdb_conn: Any = None, _milvus: Any = None) -> int:
    """Run incremental or rebuild ingest; render summary or JSON.

    Args:
        args: parsed argparse.Namespace with .rebuild (bool) and .format.
        _tsdb_conn: test-injection psycopg connection.
        _milvus: unused; accepted for shim symmetry.

    Returns:
        0 on success, 2 on unexpected error.
    """
    log.info("command_started", command="ingest", rebuild=args.rebuild)
    try:
        from src.study.ingest import ingest_incremental, ingest_rebuild

        fn = ingest_rebuild if args.rebuild else ingest_incremental
        result = fn(_tsdb_conn=_tsdb_conn)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.ingest.unexpected", error=str(exc))
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_render_text(result))
    log.info(
        "command_complete",
        command="ingest",
        hands_processed=result.get("hands_processed"),
        elapsed_s=result.get("elapsed_s"),
    )
    return 0


def _render_text(r: dict) -> str:
    """Multi-line ingest summary."""
    skipped = r.get("hands_skipped") or {}
    if isinstance(skipped, dict):
        skip_str = " · ".join(f"{k}={v}" for k, v in skipped.items()) or "0"
    else:
        skip_str = str(skipped)
    return (
        "ingest complete\n"
        f"  hands_processed: {r.get('hands_processed', 0)}\n"
        f"  hands_skipped:   {skip_str}\n"
        f"  elapsed:         {float(r.get('elapsed_s') or 0):.1f}s\n"
        f"  watermark:       {r.get('watermark_before')} → {r.get('watermark_after')}"
    )
