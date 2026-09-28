"""autoloop step subcommand: run one auto-loop cycle.

Output on success:
    JSON to stdout: {"patches_applied": <N>}
    Exit code 0.

Output on driver error:
    JSON to stderr: {"error": "..."}
    Exit code 2.

Usage:
    poker-engine autoloop step
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger
from src.autoloop.driver import AutoLoopDriver
from src.autoloop.orchestrator import AutonomousRunner

log = get_logger("cli.autoloop")


def run(args: argparse.Namespace) -> int:
    """Dispatch the autoloop subcommand: `step` (one cycle) or `run` (autonomous loop).

    A legacy Namespace with no autoloop_cmd attribute defaults to the step path.
    Returns 0 on success, 2 on error.
    """
    cmd = getattr(args, "autoloop_cmd", "step")
    if cmd == "run":
        return _run_autonomous(args)

    try:
        driver = AutoLoopDriver()
        applied = driver.step()
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.autoloop.error", error=str(exc))
        return 2

    print(json.dumps({"patches_applied": applied}))
    log.info("cli.autoloop.done", patches_applied=applied)
    return 0


def _run_autonomous(args: argparse.Namespace) -> int:
    """Run the autonomous multi-cycle loop. Returns 0 on success, 2 on startup error."""
    try:
        run_id = args.resume or args.run_id
        if run_id is None:
            print(
                json.dumps({"error": "--run-id is required unless --resume is given"}),
                file=sys.stderr,
            )
            return 2
        runner = AutonomousRunner(run_id=run_id, base_seed=args.seed, max_cycles=args.max_cycles)
        summary = runner.run()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.autoloop.run.error", error=str(exc))
        return 2

    print(json.dumps(summary))
    log.info(
        "cli.autoloop.run.done",
        cycles=summary.get("cycles"),
        patches_accepted=summary.get("patches_accepted"),
        patches_rejected=summary.get("patches_rejected"),
    )
    return 0
