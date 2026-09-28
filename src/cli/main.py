"""poker-engine CLI dispatcher.

Phase 4-5 surface (back-compat):
    poker-engine uncertain-spots [--session ID] [--limit N] [--street S] [--min-distance F]
    poker-engine ev-loss <cluster_key> [--session ID]
    poker-engine autoloop step
    poker-engine rollback <patch_id>

Phase 6 additions (CLI-01..07 + D-NEW-30 Eval):
    poker-engine ingest    [--rebuild]                                    [--format text|json]
    poker-engine similar   <cluster_key> [--k N]                          [--format text|json]
    poker-engine edit-node <cluster_key> [--set 'k:v,...' | --editor]
                                         [--reason TEXT]                  [--format text|json]
    poker-engine leaks     [--type coverage|strategy|both] [--min-n N]
                           [--limit N]                                    [--format text|json]
    poker-engine ab        <cluster_key> <patch_id> [--n-hands N] [--seed N]
                                                                          [--format text|json]
    poker-engine dashboard                                                [--format text|json]
    poker-engine patches   list [--limit N] [--cluster-key X] [--source X][--format text|json]
    poker-engine patches   rollback <patch_id>
    poker-engine eval      --opponent NAME [--hands N] [--seed N] [--no-persist]
                                                                          [--format text|json]

All Phase-6 subcommands accept ``--format text|json`` (default text) per D-09.
JSON form is the same shape the FastAPI endpoints (Plan 07) return.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser (extracted for testability)."""
    ap = argparse.ArgumentParser(
        prog="poker-engine",
        description="poker-engine operator CLI.",
    )
    subparsers = ap.add_subparsers(dest="command", required=True)

    # ---------- Phase 4-5 (back-compat) -------------------------------------

    # uncertain-spots
    us_sp = subparsers.add_parser(
        "uncertain-spots",
        help="List flagged-sparse observations sorted by max neighbor distance DESC.",
    )
    us_sp.add_argument("--session", type=str, default=None, metavar="ID")
    us_sp.add_argument("--limit", type=int, default=50, metavar="N")
    us_sp.add_argument(
        "--street",
        type=str,
        default=None,
        choices=["preflop", "postflop"],
        metavar="S",
    )
    us_sp.add_argument(
        "--min-distance",
        type=float,
        default=None,
        metavar="F",
        dest="min_distance",
    )

    # ev-loss
    el_sp = subparsers.add_parser(
        "ev-loss",
        help="Compute KL divergence (ev_loss) for a cluster_key.",
    )
    el_sp.add_argument("cluster_key", type=str)
    el_sp.add_argument("--session", type=str, default=None, metavar="ID")

    # autoloop
    al_sp = subparsers.add_parser(
        "autoloop",
        help="Auto-loop: `step` runs one cycle; `run` drives the autonomous multi-cycle loop.",
    )
    al_sub = al_sp.add_subparsers(dest="autoloop_cmd", required=True)
    al_sub.add_parser("step", help="Run one auto-loop cycle.")

    run_sp = al_sub.add_parser("run", help="Run the autonomous multi-cycle auto-loop.")
    run_sp.add_argument(
        "--autonomous",
        action="store_true",
        help="Drive repeated sim->step cycles unattended.",
    )
    run_sp.add_argument(
        "--run-id",
        type=str,
        default=None,
        dest="run_id",
        help="Run identity; keys the checkpoint file. Required unless --resume.",
    )
    run_sp.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Run-level base seed; per-cycle seed = base + cycle_idx.",
    )
    run_sp.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        dest="max_cycles",
        help="Stop after N cycles (default: run until gracefully stopped).",
    )
    run_sp.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume an existing run by run_id from its checkpoint.",
    )

    # rollback (top-level back-compat — same code path is also wired under
    # `patches rollback` per CLI-07)
    rb_sp = subparsers.add_parser(
        "rollback",
        help="Roll back a patch by patch_id.",
    )
    rb_sp.add_argument("patch_id", type=str, help="UUID of the patch to roll back.")

    # ---------- Phase 6 (CLI-01..07 + D-NEW-30) -----------------------------

    # CLI-01: ingest
    ingest_sp = subparsers.add_parser(
        "ingest",
        help="CLI-01: Run HH ingest pipeline (incremental by default).",
    )
    ingest_sp.add_argument(
        "--rebuild",
        action="store_true",
        help="Trigger full pipeline rebuild instead of incremental.",
    )
    ingest_sp.add_argument("--format", choices=["text", "json"], default="text")

    # CLI-02: similar
    similar_sp = subparsers.add_parser(
        "similar",
        help="CLI-02: Return k nearest clusters.",
    )
    similar_sp.add_argument("cluster_key", type=str)
    similar_sp.add_argument("--k", type=int, default=10)
    similar_sp.add_argument("--format", choices=["text", "json"], default="text")

    # CLI-03: edit-node
    edit_sp = subparsers.add_parser(
        "edit-node",
        help="CLI-03: Edit a cluster's action_dist via PatchEngine (source=manual).",
    )
    edit_sp.add_argument("cluster_key", type=str)
    edit_group = edit_sp.add_mutually_exclusive_group()
    edit_group.add_argument(
        "--set",
        type=str,
        default=None,
        help="comma-separated freq pairs: 'fold:0.3,call:0.5,bet_50:0.2'",
    )
    edit_group.add_argument(
        "--editor",
        action="store_true",
        help="Open $EDITOR with the action_dist JSON (default if neither flag is given).",
    )
    edit_sp.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Optional reason text recorded with the patch.",
    )
    edit_sp.add_argument("--format", choices=["text", "json"], default="text")

    # CLI-04: leaks
    leaks_sp = subparsers.add_parser(
        "leaks",
        help="CLI-04: List clusters ranked by ev_loss with Bayesian credible intervals.",
    )
    leaks_sp.add_argument(
        "--type",
        choices=["coverage", "strategy", "both"],
        default="both",
    )
    leaks_sp.add_argument("--min-n", type=int, default=20, dest="min_n")
    leaks_sp.add_argument("--limit", type=int, default=20)
    leaks_sp.add_argument("--format", choices=["text", "json"], default="text")

    # CLI-05: ab
    ab_sp = subparsers.add_parser(
        "ab",
        help="CLI-05: Run sim A/B for a candidate patch; print ev_loss delta + CI.",
    )
    ab_sp.add_argument("cluster_key", type=str)
    ab_sp.add_argument("patch_id", type=str)
    ab_sp.add_argument("--n-hands", type=int, default=10000, dest="n_hands")
    ab_sp.add_argument("--seed", type=int, default=42)
    ab_sp.add_argument("--format", choices=["text", "json"], default="text")

    # CLI-06: dashboard
    dashboard_sp = subparsers.add_parser(
        "dashboard",
        help="CLI-06: Print verdict-led dashboard (loop_health, ev_loss_trend, ...).",
    )
    dashboard_sp.add_argument("--format", choices=["text", "json"], default="text")

    # CLI-07: patches (two-level)
    patches_sp = subparsers.add_parser(
        "patches",
        help="CLI-07: List patches reverse-chrono; rollback <patch_id>.",
    )
    patches_sub = patches_sp.add_subparsers(dest="patches_command", required=True)

    patches_list_sp = patches_sub.add_parser("list", help="List patches.")
    patches_list_sp.add_argument("--limit", type=int, default=20)
    patches_list_sp.add_argument("--cluster-key", type=str, default=None, dest="cluster_key")
    patches_list_sp.add_argument("--source", type=str, default=None)
    patches_list_sp.add_argument("--format", choices=["text", "json"], default="text")

    patches_rollback_sp = patches_sub.add_parser("rollback", help="Roll back a patch by patch_id.")
    patches_rollback_sp.add_argument("patch_id", type=str)

    # D-NEW-30: eval (REGISTRY-driven choices keep names typo-safe)
    from src.eval.baselines import REGISTRY as _BASELINE_REGISTRY

    eval_sp = subparsers.add_parser(
        "eval",
        help="D-NEW-30: Run head-to-head match vs a baseline strategy.",
    )
    eval_sp.add_argument(
        "--opponent",
        choices=sorted(_BASELINE_REGISTRY),
        required=True,
    )
    eval_sp.add_argument("--hands", type=int, default=10000)
    eval_sp.add_argument("--seed", type=int, default=42)
    eval_sp.add_argument(
        "--no-persist",
        action="store_true",
        dest="no_persist",
        help="Skip writing MatchResult to matches table (dry-run).",
    )
    eval_sp.add_argument("--format", choices=["text", "json"], default="text")

    # Corpus-version A/B: pin two versions, paired-seed match, report EV delta.
    compare_sp = subparsers.add_parser(
        "compare",
        help="Paired-seed A/B between two corpus versions (reports bb/100 EV delta).",
    )
    compare_sp.add_argument("--old", type=int, required=True, metavar="V")
    compare_sp.add_argument("--new", type=int, required=True, metavar="V")
    compare_sp.add_argument("--hands", type=int, default=2000)
    compare_sp.add_argument("--seed", type=int, default=42)
    compare_sp.add_argument("--format", choices=["text", "json"], default="text")

    # Phase 9: eval-suite
    eval_suite_sp = subparsers.add_parser(
        "eval-suite",
        help="Phase 9: Run full eval suite (LOO gate + exploitability + match-play).",
    )
    eval_suite_sp.add_argument("--hands", type=int, default=10000)
    eval_suite_sp.add_argument("--seed", type=int, default=42)
    eval_suite_sp.add_argument("--format", choices=["text", "json"], default="text")

    return ap


def main() -> int:
    """Dispatch to the selected subcommand. Returns exit code (0 = success)."""
    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

    ap = _build_parser()
    args = ap.parse_args()

    # ---------- Dispatch (lazy imports per established pattern) -------------

    if args.command == "uncertain-spots":
        from src.cli.uncertain_spots import run

        return run(args)

    if args.command == "ev-loss":
        from src.cli.ev_loss import run

        return run(args)

    if args.command == "autoloop":
        from src.cli.autoloop import run

        return run(args)

    if args.command == "rollback":
        from src.cli.rollback import run

        return run(args)

    if args.command == "ingest":
        from src.cli.ingest import run

        return run(args)

    if args.command == "similar":
        from src.cli.similar import run

        return run(args)

    if args.command == "edit-node":
        from src.cli.edit_node import run

        return run(args)

    if args.command == "leaks":
        from src.cli.leaks import run

        return run(args)

    if args.command == "ab":
        from src.cli.ab import run

        return run(args)

    if args.command == "dashboard":
        from src.cli.dashboard import run

        return run(args)

    if args.command == "patches":
        from src.cli.patches import run

        return run(args)

    if args.command == "eval":
        from src.cli.eval import run

        return run(args)

    if args.command == "compare":
        from src.cli.compare import run

        return run(args)

    if args.command == "eval-suite":
        from src.cli.eval_suite import run

        return run(args)

    # Unreachable: argparse required=True ensures a subcommand is always chosen.
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
