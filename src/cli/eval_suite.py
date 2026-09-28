"""poker-engine eval-suite CLI subcommand.

Exit codes (D-09-15/16 contract — human/CI reads the code):
    0 — suite passed (tier1_pass AND match_pass)
    1 — suite failed
    2 — unexpected error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger

log = get_logger("cli.eval_suite")


def run(args: argparse.Namespace, *, _tsdb_conn: Any = None) -> int:
    """Run the full eval suite and render the result.

    Args:
        args: Parsed Namespace with .hands, .seed, .format.
        _tsdb_conn: Test-injection psycopg connection.

    Returns:
        0 if suite_pass, 1 if suite failed, 2 on unexpected error.
    """
    log.info("command_started", command="eval-suite", hands=args.hands, seed=args.seed)
    try:
        from src.eval.suite import run_suite

        result = run_suite(
            hands=args.hands,
            seed=args.seed,
            _tsdb_conn=_tsdb_conn,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.eval_suite.failed", error=str(exc))
        return 2

    import msgspec

    out_dict = msgspec.to_builtins(result)
    if args.format == "json":
        print(json.dumps(out_dict, indent=2, default=str))
    else:
        print(_render_text(out_dict))
    log.info("command_complete", command="eval-suite", suite_pass=result.suite_pass)
    return 0 if result.suite_pass else 1


def _render_text(r: dict) -> str:
    """Multi-section suite summary."""
    return (
        f"Eval Suite Report\n"
        f"  Tier-1 LOO TVD: {float(r['tier1_tvd']):.4f}  "
        f"(floor={float(r['tier1_threshold_floor']):.4f})  "
        f"{'PASS' if r['tier1_pass'] else 'FAIL'}\n"
        f"  Tier-2 exploitability: {float(r['tier2_exploitability']):.4f}  (report-only)\n"
        f"  Match-play: {'PASS' if r['match_pass'] else 'FAIL'}\n"
        f"  Suite: {'PASS' if r['suite_pass'] else 'FAIL'}"
    )
