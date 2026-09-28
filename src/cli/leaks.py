"""CLI-04 — `poker-engine leaks` shim. D-12.

Wraps ``src.study.leaks.rank_leaks``. Text and JSON formats supported.

Output on success:
    JSON to stdout (--format json) or sectioned text (--format text).
    Exit code 0.

Output on unexpected error:
    JSON to stderr: {"error": "..."}
    Exit code 2.

Help text (per CONTEXT D-12):
    CLI-04: List clusters ranked by ev_loss with Bayesian credible intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger

log = get_logger("cli.leaks")


def run(args: argparse.Namespace, *, _tsdb_conn: Any = None, _milvus: Any = None) -> int:
    """Rank coverage gaps + strategy leaks and render to stdout.

    Args:
        args: parsed argparse.Namespace with .type, .min_n, .limit, .format.
        _tsdb_conn: test-injection psycopg connection.
        _milvus: unused here; accepted for the shared shim signature.

    Returns:
        0 on success, 2 on unexpected error.
    """
    log.info(
        "command_started",
        command="leaks",
        type=args.type,
        min_n=args.min_n,
        limit=args.limit,
    )
    try:
        from src.study.leaks import rank_leaks

        result = rank_leaks(
            leak_type=args.type,
            min_n=args.min_n,
            limit=args.limit,
            _tsdb_conn=_tsdb_conn,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.leaks.unexpected", error=str(exc))
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_render_text(result, args.type))
    log.info(
        "command_complete",
        command="leaks",
        n_coverage=len(result.get("coverage", [])),
        n_strategy=len(result.get("strategy", [])),
    )
    return 0


def _truncate_ck(ck: str, max_len: int = 60) -> str:
    """Truncate a long cluster_key with a Unicode ellipsis in the middle."""
    if len(ck) <= max_len:
        return ck
    half = (max_len - 1) // 2
    return f"{ck[:half]}…{ck[-half:]}"


def _render_text(result: dict, leak_type: str) -> str:
    """Sectioned text renderer per UI-SPEC v2.1.

    Renders ## COVERAGE GAPS and ## STRATEGY LEAKS sections gated by leak_type.
    Low-N rows in the strategy section get a `(n=N, shrunk)` suffix per D-06.
    """
    lines: list[str] = []
    if leak_type in ("coverage", "both") and result.get("coverage"):
        lines.append("## COVERAGE GAPS")
        lines.append(f"{'obs_id':<10} {'cluster_key':<62} {'max_dist':>8} {'session':<12}")
        for r in result["coverage"]:
            lines.append(
                f"{str(r.get('obs_id', ''))[:10]:<10} "
                f"{_truncate_ck(str(r.get('cluster_key', ''))):<62} "
                f"{float(r.get('max_neighbor_distance') or 0):>8.3f} "
                f"{str(r.get('session_id', ''))[:12]:<12}"
            )
        lines.append("")
    if leak_type in ("strategy", "both") and result.get("strategy"):
        lines.append("## STRATEGY LEAKS")
        lines.append(f"{'cluster_key':<62} {'ev_loss [CI]':<28} {'n_obs':>6} {'source':<10} {'stage':<5}")
        for r in result["strategy"]:
            ci_str = (
                f"{float(r.get('ev_loss', 0)):.3f} "
                f"[{float(r.get('ci_low', 0)):.3f}-{float(r.get('ci_high', 0)):.3f}]"
            )
            n_str = f"(n={int(r.get('n_obs', 0))}{', shrunk' if r.get('shrunk') else ''})"
            lines.append(
                f"{_truncate_ck(str(r.get('cluster_key', ''))):<62} "
                f"{ci_str:<20}{n_str:<8} "
                f"{int(r.get('n_obs', 0)):>6} "
                f"{r.get('source', '')!s:<10} "
                f"{r.get('stage', '')!s:<5}"
            )
    return "\n".join(lines) if lines else "(no leaks)"
