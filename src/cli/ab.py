"""CLI-05 — `poker-engine ab <cluster_key> <patch_id>` shim.

Wraps ``src.study.ab.run_ab``. Prints ev_loss delta + 95% CI + n_hands.

Exit codes:
    0 — success
    1 — expected error (ValueError, e.g. cluster_key/patch_id mismatch)
    2 — unexpected error

Help text:
    CLI-05: Run sim A/B for a candidate patch; print ev_loss delta + CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger

log = get_logger("cli.ab")


def run(args: argparse.Namespace, *, _tsdb_conn: Any = None, _milvus: Any = None) -> int:
    """Run sim A/B and render result.

    Args:
        args: parsed argparse.Namespace with .cluster_key, .patch_id,
              .n_hands, .seed, .format.
        _tsdb_conn: test-injection psycopg connection.
        _milvus: test-injection MilvusClient.

    Returns:
        0 on success, 1 on ValueError, 2 on unexpected error.
    """
    log.info(
        "command_started",
        command="ab",
        cluster_key=args.cluster_key,
        patch_id=args.patch_id,
        n_hands=args.n_hands,
        seed=args.seed,
    )
    try:
        from src.study.ab import run_ab

        result = run_ab(
            args.cluster_key,
            args.patch_id,
            n_hands=args.n_hands,
            seed=args.seed,
            _tsdb_conn=_tsdb_conn,
            _milvus=_milvus,
        )
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.ab.failed", error=str(exc))
        return 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.ab.unexpected", error=str(exc))
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_render_text(result))
    log.info("command_complete", command="ab", patch_id=args.patch_id)
    return 0


def _render_text(r: dict) -> str:
    """Formatted block render with signed delta + CI."""
    return (
        f"sim A/B for patch_id={r.get('patch_id')} cluster_key={r.get('cluster_key')}\n"
        f"  ev_loss_delta:  {float(r.get('ev_loss_delta', 0)):+.4f}\n"
        f"  95% CI:         [{float(r.get('ci_low', 0)):+.4f}, "
        f"{float(r.get('ci_high', 0)):+.4f}]\n"
        f"  n_hands:        {int(r.get('n_hands', 0))}\n"
        f"  seed:           {int(r.get('seed', 0))}"
    )
