"""Phase 2 — Bootstrap kNN Knowledge Base: end-to-end pipeline orchestrator.

Strings together 6 stages with SHA256 sidecar-based incremental resume (BOOT-05)
and OBS-03 unified progress logging to both stderr and outputs/phase2_build.log.

Stages (in order):
  1. replay    -> tools/hm-replayer (Rust binary, reads hm3.sqlite, writes hm_events.jsonl)
  2. extract   -> tools/extract_decisions.py (jsonl -> jsonl)
  3. embed     -> tools/build_embedding.py (jsonl -> chunked parquet)
  4. zscore    -> tools/zscore_fit.py (parquet -> zscore_preflop.json + zscore_postflop.json)
  5. upsert    -> tools/upsert_milvus.py (parquet + manifests -> Milvus; sentinel .upsert.done)
  6. validate  -> tools/validation_experiments.py (manifests + Milvus -> EMBEDDING-VALIDATION-RESULTS.md)

Usage:
    python tools/phase2_pipeline.py --all
    python tools/phase2_pipeline.py --stage embed
    python tools/phase2_pipeline.py --rebuild-from embed
    python tools/phase2_pipeline.py --rebuild                   # same as --rebuild-from all
    python tools/phase2_pipeline.py --rebuild-milvus --rebuild  # full drop+recreate Milvus + all stages
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src._log import configure_logging, get_logger

log = get_logger("tools.phase2_pipeline")

# ─── Constants ──────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "research" / "preflop-ranges" / "outputs"
BUILD_LOG = OUTPUT_DIR / "phase2_build.log"


# ─── SHA256 sidecar helpers (BOOT-05 incremental) ──────────────────────────


def _input_sha256(paths: list[Path]) -> str:
    """Compute a deterministic SHA256 over the content of all given paths.

    Supports glob patterns (paths containing *, ?, or [). Sorted for determinism.
    Only existing files contribute to the hash; missing paths are silently skipped
    so that a stage whose inputs haven't been produced yet gets a fresh hash on
    each call (preventing spurious skip).
    """
    h = hashlib.sha256()
    expanded: list[Path] = []
    for p in paths:
        p_str = str(p)
        if any(c in p_str for c in ("*", "?", "[")):
            # Glob-expand; sort to ensure determinism across OS
            matches = sorted(Path(m) for m in glob.glob(p_str))
            expanded.extend(matches)
        else:
            expanded.append(p)

    for p in sorted(expanded):
        if not p.exists():
            continue
        with p.open("rb") as fh:
            while chunk := fh.read(1 << 20):  # 1 MiB chunks
                h.update(chunk)
    return h.hexdigest()


def _stage_done(output: Path, current_input_hash: str) -> bool:
    """Return True iff the output file exists AND its sidecar matches *current_input_hash*."""
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if not output.exists() or not sidecar.exists():
        return False
    return sidecar.read_text().strip() == current_input_hash


def _write_sidecar(output: Path, input_hash: str) -> None:
    """Write *input_hash* next to *output* as ``<output>.sha256``."""
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(input_hash + "\n")


# ─── Stage definitions ──────────────────────────────────────────────────────


@dataclass
class StageDef:
    """Descriptor for a single pipeline stage."""

    name: str
    cmd_builder: Callable[[argparse.Namespace], list[str]]
    inputs: Callable[[argparse.Namespace], list[Path]]
    primary_output: Callable[[argparse.Namespace], Path]


def _cmd_replay(args: argparse.Namespace) -> list[str]:
    """Build the hm-replayer Rust binary invocation."""
    binary = REPO_ROOT / "tools" / "hm-replayer" / "target" / "release" / "hm-replayer"
    return [
        str(binary),
        "--db",
        args.hm3_sqlite,
        "--out",
        str(OUTPUT_DIR / "hm_events.jsonl"),
    ]


def _cmd_extract(args: argparse.Namespace) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "tools" / "extract_decisions.py"),
        "--in",
        str(OUTPUT_DIR / "hm_events.jsonl"),
        "--out",
        str(OUTPUT_DIR / "hm_decisions.jsonl"),
    ]


def _cmd_preflop_equity(args: argparse.Namespace) -> list[str]:
    """Build the preflop equity table via the exact 169x169 matrix method.

    Replaces the legacy Monte Carlo path: deterministic, exact deciles, fast.
    Depends on tools/preflop_hand_vs_hand_169.parquet (committed).
    """
    return [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "tools" / "build_preflop_equity_table.py"),
        "--decisions",
        str(OUTPUT_DIR / "hm_decisions.jsonl"),
        "--out",
        str(REPO_ROOT / "tools" / "preflop_equity_table.parquet"),
        "--method",
        "matrix",
        "--matrix",
        str(REPO_ROOT / "tools" / "preflop_hand_vs_hand_169.parquet"),
        "--log-file",
        str(BUILD_LOG),
    ]


def _cmd_embed(args: argparse.Namespace) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "tools" / "build_embedding.py"),
        "--decisions",
        str(OUTPUT_DIR / "hm_decisions.jsonl"),
        "--equity-table",
        str(REPO_ROOT / "tools" / "equity_table.parquet"),
        "--preflop-equity-table",
        str(REPO_ROOT / "tools" / "preflop_equity_table.parquet"),
        "--out-dir",
        str(OUTPUT_DIR),
        "--chunk-size",
        str(args.chunk_size),
        "--log-file",
        str(BUILD_LOG),
    ]
    return cmd


def _cmd_zscore(args: argparse.Namespace) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "tools" / "zscore_fit.py"),
        "--chunks-dir",
        str(OUTPUT_DIR),
        "--out-preflop",
        str(REPO_ROOT / "tools" / "zscore_preflop.json"),
        "--out-postflop",
        str(REPO_ROOT / "tools" / "zscore_postflop.json"),
        "--log-file",
        str(BUILD_LOG),
    ]


def _cmd_upsert(args: argparse.Namespace) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "tools" / "upsert_milvus.py"),
        "--chunks-dir",
        str(OUTPUT_DIR),
        "--preflop-manifest",
        str(REPO_ROOT / "tools" / "zscore_preflop.json"),
        "--postflop-manifest",
        str(REPO_ROOT / "tools" / "zscore_postflop.json"),
        "--log-file",
        str(BUILD_LOG),
    ]
    if getattr(args, "rebuild_milvus", False):
        cmd.append("--rebuild")
    return cmd


def _cmd_validate(args: argparse.Namespace) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "tools" / "validation_experiments.py"),
        "--preflop-manifest",
        str(REPO_ROOT / "tools" / "zscore_preflop.json"),
        "--postflop-manifest",
        str(REPO_ROOT / "tools" / "zscore_postflop.json"),
        "--out",
        str(REPO_ROOT / "docs" / "research" / "EMBEDDING-VALIDATION-RESULTS.md"),
    ]


STAGES: list[StageDef] = [
    StageDef(
        name="replay",
        cmd_builder=_cmd_replay,
        inputs=lambda args: [Path(args.hm3_sqlite)],
        primary_output=lambda args: OUTPUT_DIR / "hm_events.jsonl",
    ),
    StageDef(
        name="extract",
        cmd_builder=_cmd_extract,
        inputs=lambda args: [OUTPUT_DIR / "hm_events.jsonl"],
        primary_output=lambda args: OUTPUT_DIR / "hm_decisions.jsonl",
    ),
    StageDef(
        name="preflop_equity",
        cmd_builder=_cmd_preflop_equity,
        inputs=lambda args: [
            OUTPUT_DIR / "hm_decisions.jsonl",
            REPO_ROOT / "tools" / "preflop_hand_vs_hand_169.parquet",
        ],
        primary_output=lambda args: REPO_ROOT / "tools" / "preflop_equity_table.parquet",
    ),
    StageDef(
        name="embed",
        cmd_builder=_cmd_embed,
        inputs=lambda args: [
            OUTPUT_DIR / "hm_decisions.jsonl",
            REPO_ROOT / "tools" / "equity_table.parquet",
            REPO_ROOT / "tools" / "preflop_equity_table.parquet",
        ],
        primary_output=lambda args: OUTPUT_DIR / "hm_embeddings_chunk_0001.parquet",
    ),
    StageDef(
        name="zscore",
        cmd_builder=_cmd_zscore,
        inputs=lambda args: [Path(str(OUTPUT_DIR / "hm_embeddings_chunk_*.parquet"))],
        primary_output=lambda args: REPO_ROOT / "tools" / "zscore_postflop.json",
    ),
    StageDef(
        name="upsert",
        cmd_builder=_cmd_upsert,
        inputs=lambda args: [
            REPO_ROOT / "tools" / "zscore_preflop.json",
            REPO_ROOT / "tools" / "zscore_postflop.json",
            Path(str(OUTPUT_DIR / "hm_embeddings_chunk_*.parquet")),
        ],
        primary_output=lambda args: OUTPUT_DIR / ".upsert.done",
    ),
    StageDef(
        name="validate",
        cmd_builder=_cmd_validate,
        inputs=lambda args: [
            REPO_ROOT / "tools" / "zscore_preflop.json",
            REPO_ROOT / "tools" / "zscore_postflop.json",
        ],
        primary_output=lambda args: REPO_ROOT / "docs" / "research" / "EMBEDDING-VALIDATION-RESULTS.md",
    ),
]

STAGE_NAMES: list[str] = [s.name for s in STAGES]


# ─── Cascade helper ─────────────────────────────────────────────────────────


def _rebuild_from_cascade(name: str) -> set[str]:
    """Return the set of stages that must be re-run when *name* is forced.

    ``"all"`` returns all stage names. Raises ``ValueError`` for unknown names.
    """
    if name == "all":
        return set(STAGE_NAMES)
    if name not in STAGE_NAMES:
        raise ValueError(f"Unknown stage: {name!r}. Valid: {STAGE_NAMES}")
    idx = STAGE_NAMES.index(name)
    return set(STAGE_NAMES[idx:])


# ─── Progress logging ────────────────────────────────────────────────────────


def _log_progress(stage_name: str, msg: str) -> None:
    """Emit a progress line to stderr AND append to BUILD_LOG (OBS-03)."""
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{timestamp}] [{stage_name}] {msg}"
    print(line, file=sys.stderr)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with BUILD_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ─── Stage runner ────────────────────────────────────────────────────────────


def _run_stage(stage: StageDef, args: argparse.Namespace, force: bool = False) -> bool:
    """Run *stage*; skip if hash unchanged and not *force*.

    Returns True if the stage was actually executed, False if skipped.
    Raises RuntimeError on non-zero subprocess exit.
    """
    input_hash = _input_sha256(stage.inputs(args))
    output_path = stage.primary_output(args)

    if not force and _stage_done(output_path, input_hash):
        _log_progress(stage.name, f"SKIPPED (input hash unchanged: {input_hash[:12]}…)")
        log.info("pipeline.stage.skipped", stage=stage.name, hash_prefix=input_hash[:12])
        return False

    _log_progress(stage.name, f"STARTING (hash: {input_hash[:12]}…)")
    log.info("pipeline.stage.start", stage=stage.name)
    t0 = time.time()

    cmd = stage.cmd_builder(args)
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        msg = f"FAILED (returncode={result.returncode}, elapsed={elapsed:.1f}s)"
        _log_progress(stage.name, msg)
        log.error("pipeline.stage.failed", stage=stage.name, returncode=result.returncode, elapsed_s=elapsed)
        raise RuntimeError(f"Stage '{stage.name}' failed with returncode {result.returncode}")

    _log_progress(stage.name, f"OK (elapsed={elapsed:.1f}s)")
    log.info("pipeline.stage.ok", stage=stage.name, elapsed_s=elapsed)

    # For the upsert sentinel, write the sentinel file before the sidecar
    if stage.name == "upsert" and not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("done\n")

    _write_sidecar(output_path, input_hash)
    return True


# ─── Main entry point ────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and orchestrate pipeline stages."""
    ap = argparse.ArgumentParser(
        description="Phase 2 pipeline orchestrator — replay→extract→embed→zscore→upsert→validate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--all",
        action="store_true",
        help="Run all 6 stages (with sidecar-based skip for unchanged inputs)",
    )
    mode.add_argument(
        "--stage",
        choices=STAGE_NAMES,
        help="Run a single stage unconditionally (force=True)",
    )
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-run of all stages (equivalent to --rebuild-from all)",
    )
    ap.add_argument(
        "--rebuild-from",
        choices=["all", *STAGE_NAMES],
        dest="rebuild_from",
        help="Force re-run from the named stage forward (downstream cascade)",
    )
    ap.add_argument(
        "--rebuild-milvus",
        action="store_true",
        help="Pass --rebuild to upsert_milvus.py (drops both collections + recreates schema)",
    )
    ap.add_argument(
        "--hm3-sqlite",
        default="data/hands.sqlite",
        help="Path to HoldemManager 3 SQLite database (replay stage only)",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=25_000,
        help="Parquet chunk row count for build_embedding.py",
    )

    args = ap.parse_args(argv)
    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

    # --rebuild is sugar for --rebuild-from all
    if args.rebuild:
        args.rebuild_from = "all"

    # Compute forced set from --rebuild-from
    forced: set[str] = _rebuild_from_cascade(args.rebuild_from) if args.rebuild_from else set()

    if args.stage:
        # Single-stage unconditional run
        stage = next(s for s in STAGES if s.name == args.stage)
        _run_stage(stage, args, force=True)
        _log_progress("pipeline", f"STAGE '{args.stage}' COMPLETE")
        return 0

    # Full pipeline (--all or default when --rebuild-from is used without --stage)
    _log_progress("pipeline", f"START — forced stages: {sorted(forced) or 'none'}")
    for stage in STAGES:
        _run_stage(stage, args, force=(stage.name in forced))

    _log_progress("pipeline", "ALL STAGES COMPLETE")
    log.info("pipeline.complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
