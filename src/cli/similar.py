"""CLI-02 — `poker-engine similar <cluster_key>` shim. D-13.

Wraps ``src.study.similar.find_similar``. JSON form sets
``include_action_dist=True`` so the FastAPI mirror returns identical payload.
Text form truncates ``cluster_key`` at 60 chars with a Unicode ellipsis.

Help text:
    CLI-02: Return k nearest clusters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger

log = get_logger("cli.similar")


def run(args: argparse.Namespace, *, _tsdb_conn: Any = None, _milvus: Any = None) -> int:
    """Return top-k similar clusters; render text or JSON.

    Args:
        args: parsed argparse.Namespace with .cluster_key, .k, .format.
        _tsdb_conn: test-injection psycopg connection.
        _milvus: test-injection MilvusClient.

    Returns:
        0 on success, 2 on unexpected error.
    """
    log.info(
        "command_started",
        command="similar",
        cluster_key=args.cluster_key,
        k=args.k,
    )
    try:
        from src.study.similar import find_similar

        result = find_similar(
            args.cluster_key,
            k=args.k,
            include_action_dist=(args.format == "json"),  # D-13 — JSON adds action_dist
            _tsdb_conn=_tsdb_conn,
            _milvus=_milvus,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.similar.unexpected", error=str(exc))
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_render_text(result))
    log.info("command_complete", command="similar", n_hits=len(result))
    return 0


def _truncate_ck(ck: str, max_len: int = 60) -> str:
    """Truncate a long cluster_key with a Unicode ellipsis at the end."""
    return ck if len(ck) <= max_len else f"{ck[: max_len - 1]}…"


def _render_text(rows: list[dict]) -> str:
    """Right-aligned tabular text renderer per D-13."""
    if not rows:
        return "(no similar clusters found)"
    lines = [f"{'rank':>4} {'cluster_key':<62} {'distance':>10} {'source':<10} {'n_obs':>6}"]
    for r in rows:
        lines.append(
            f"{int(r.get('rank', 0)):>4} "
            f"{_truncate_ck(str(r.get('cluster_key', ''))):<62} "
            f"{float(r.get('distance', 0)):>10.4f} "
            f"{r.get('source', '')!s:<10} "
            f"{int(r.get('n_obs', 0)):>6}"
        )
    return "\n".join(lines)
