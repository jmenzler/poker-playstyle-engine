"""D-NEW-30 — `poker-engine eval` shim.

Runs ``src.eval.run_match.run_match`` head-to-head against a baseline
strategy from ``src.eval.baselines.REGISTRY``. Prints ``MatchResult``
(bb/100, CI bounds, per-street + per-texture, top-5 cluster lists).

Exit codes:
    0 — success
    1 — expected error (KeyError on unknown opponent — caught defensively
        even though argparse ``choices=REGISTRY`` rejects them first)
    2 — unexpected error

Help text:
    D-NEW-30: Run head-to-head match vs a baseline strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger

log = get_logger("cli.eval")


def run(args: argparse.Namespace, *, _tsdb_conn: Any = None) -> int:
    """Run a single match and render result.

    Args:
        args: parsed argparse.Namespace with .opponent, .hands, .seed,
              .no_persist, .format.
        _tsdb_conn: test-injection psycopg connection.

    Returns:
        0 on success, 1 on KeyError (unknown opponent), 2 on unexpected error.
    """
    log.info(
        "command_started",
        command="eval",
        opponent=args.opponent,
        hands=args.hands,
        seed=args.seed,
    )
    try:
        from src.eval.run_match import run_match

        result = run_match(
            args.opponent,
            hands=args.hands,
            seed=args.seed,
            persist=not args.no_persist,
            _tsdb_conn=_tsdb_conn,
        )
    except KeyError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.eval.unknown_opponent", error=str(exc))
        return 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.eval.unexpected", error=str(exc))
        return 2

    # Use msgspec to serialise the MatchResult struct uniformly for both
    # text and JSON renderers.
    import msgspec

    out_dict = msgspec.to_builtins(result)
    if args.format == "json":
        print(json.dumps(out_dict, indent=2, default=str))
    else:
        print(_render_text(out_dict))
    log.info(
        "command_complete",
        command="eval",
        match_id=result.match_id,
        status=result.status,
    )
    return 0


def _render_text(r: dict) -> str:
    """Multi-line match summary."""
    return (
        f"match {r['match_id']}\n"
        f"  engine vs {r['opponent']}\n"
        f"  status:  {str(r['status']).upper()}\n"
        f"  bb/100:  {float(r['bb_per_100']):+.2f}  "
        f"CI: [{float(r['ci_low']):+.2f}, {float(r['ci_high']):+.2f}]\n"
        f"  hands:   {r['hands']}  seed: {r['seed']}\n"
        f"  per_street: {r.get('per_street')}\n"
        f"  per_texture: {r.get('per_texture')}"
    )
