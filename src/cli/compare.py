"""`poker-engine compare` shim — head-to-head corpus-version A/B.

Pins ``--old`` / ``--new`` corpus versions and plays the new engine against the
old engine at one table; prints v_new's bb/100 edge. Exit 0 ok, 2 on error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger

log = get_logger("cli.compare")


def run(args: argparse.Namespace) -> int:
    """Run a paired-seed version A/B and render the delta. Returns exit code."""
    log.info(
        "command_started", command="compare", old=args.old, new=args.new, hands=args.hands, seed=args.seed
    )
    try:
        from src.study.compare import compare

        result = compare(args.old, args.new, hands=args.hands, seed=args.seed)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.compare.unexpected", error=str(exc))
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_render_text(result))
    log.info("command_complete", command="compare", bb_per_100=result["bb_per_100"])
    return 0


def _render_text(r: dict) -> str:
    """One-block summary of the head-to-head version A/B."""
    ci = r["ci"]
    return (
        f"compare v{r['v_new']} vs v{r['v_old']} (head-to-head)\n"
        f"  v{r['v_new']} bb/100 vs v{r['v_old']}: {float(r['bb_per_100']):+.2f}  "
        f"CI: [{float(ci[0]):+.2f}, {float(ci[1]):+.2f}]\n"
        f"  hands: {r['n_hands']}"
    )
