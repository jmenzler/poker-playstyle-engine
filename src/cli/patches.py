"""CLI-07 — `poker-engine patches list` + `poker-engine patches rollback <id>` shim.

D-14 columns: patch_id, ts, cluster_key, source, pre_ev_loss, post_ev_loss, status.

Two-level subparser:
    poker-engine patches list      [--limit N] [--cluster-key X] [--source X] [--format text|json]
    poker-engine patches rollback  <patch_id>

The ``rollback`` subcommand delegates to the existing ``src.cli.rollback.run``
shim shipped in Phase 5 (no modification to that file).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger

log = get_logger("cli.patches")


def run(args: argparse.Namespace, *, _tsdb_conn: Any = None, _milvus: Any = None) -> int:
    """Two-level dispatcher: list | rollback.

    Args:
        args: parsed argparse.Namespace with .patches_command and either
              list-side args (.limit, .cluster_key, .source, .format) or
              rollback-side args (.patch_id).
        _tsdb_conn: test-injection psycopg connection.
        _milvus: test-injection MilvusClient (passed through to rollback).

    Returns:
        0 on success, 1 on rollback validation failure, 2 on unexpected error.
    """
    if args.patches_command == "list":
        return _run_list(args, _tsdb_conn=_tsdb_conn)
    if args.patches_command == "rollback":
        # Lazy import keeps argparse --help fast and prevents PatchEngine
        # construction at module-load time.
        from src.cli.rollback import run as rollback_run

        return rollback_run(args, _tsdb_conn=_tsdb_conn, _milvus=_milvus)
    print(json.dumps({"error": "subcommand required: list | rollback"}), file=sys.stderr)
    return 2


def _run_list(args: argparse.Namespace, *, _tsdb_conn: Any) -> int:
    """`patches list` — wraps src.study.patches.list_patches."""
    log.info(
        "command_started",
        command="patches.list",
        limit=args.limit,
        cluster_key=args.cluster_key,
        source=args.source,
    )
    try:
        from src.study.patches import list_patches

        result = list_patches(
            limit=args.limit,
            cluster_key=args.cluster_key,
            source=args.source,
            _tsdb_conn=_tsdb_conn,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.patches.list.unexpected", error=str(exc))
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_render_text(result))
    log.info("command_complete", command="patches.list", n_rows=len(result))
    return 0


def _truncate(s: str, n: int = 60) -> str:
    """Truncate long cluster_keys with a Unicode ellipsis."""
    return s if len(s) <= n else f"{s[: n - 1]}…"


def _render_text(rows: list[dict]) -> str:
    """Reverse-chronological table per D-14."""
    if not rows:
        return "(no patches)"
    lines = [
        f"{'patch_id':<10} {'ts':<22} {'cluster_key':<62} "
        f"{'source':<10} {'pre_ev':>8} {'post_ev':>8} {'status':<10}"
    ]
    for r in rows:
        lines.append(
            f"{str(r.get('patch_id', ''))[:10]:<10} "
            f"{str(r.get('ts', ''))[:22]:<22} "
            f"{_truncate(str(r.get('cluster_key', ''))):<62} "
            f"{r.get('source', '')!s:<10} "
            f"{float(r.get('pre_ev_loss') or 0):>8.3f} "
            f"{float(r.get('post_ev_loss') or 0):>8.3f} "
            f"{r.get('status', '')!s:<10}"
        )
    return "\n".join(lines)
