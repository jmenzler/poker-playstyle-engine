"""Build the exact 169x169 preflop all-in equity matrix as a parquet.

Wraps the Rust `preflop-matrix` binary (tools/preflop-matrix/, rs_poker exact
enumeration — every C(48,5) board, no Monte Carlo). The Rust step emits a JSON
{classes, matrix}; this wrapper validates it and writes a long-form parquet
(hero_class, villain_class, equity) that downstream range-weighted preflop
equity can consume via a simple weighted sum over villain combo_counts.

Run:
    cargo build --release -p preflop-matrix
    ./target/release/preflop-matrix --out tools/preflop_hand_vs_hand_169.json --threads 12
    uv run python tools/build_preflop_hand_vs_hand.py \\
        --matrix-json tools/preflop_hand_vs_hand_169.json \\
        --out tools/preflop_hand_vs_hand_169.parquet

Or let this script invoke the binary itself with --build.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = REPO_ROOT / "target" / "release" / "preflop-matrix"

# Canonical 169 class ordering — must match the Rust binary's all_classes().
RANKS = "23456789TJQKA"
_PAIRS = [r + r for r in RANKS]
_SUITED = [RANKS[i] + RANKS[j] + "s" for i in range(13) for j in range(i)]
_OFFSUIT = [RANKS[i] + RANKS[j] + "o" for i in range(13) for j in range(i)]
EXPECTED_CLASSES: set[str] = set(_PAIRS + _SUITED + _OFFSUIT)


def _run_binary(out_json: Path, threads: int) -> None:
    if not BINARY.exists():
        raise FileNotFoundError(f"{BINARY} not found — run `cargo build --release -p preflop-matrix` first")
    cmd = [str(BINARY), "--out", str(out_json), "--threads", str(threads)]
    log.info("running preflop-matrix: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_and_validate(matrix_json: Path) -> tuple[list[str], list[list[float]]]:
    data = json.loads(matrix_json.read_text())
    classes: list[str] = data["classes"]
    matrix: list[list[float]] = data["matrix"]

    if len(classes) != 169:
        raise ValueError(f"expected 169 classes, got {len(classes)}")
    if set(classes) != EXPECTED_CLASSES:
        missing = EXPECTED_CLASSES - set(classes)
        extra = set(classes) - EXPECTED_CLASSES
        raise ValueError(f"class set mismatch; missing={missing} extra={extra}")
    if len(matrix) != 169 or any(len(row) != 169 for row in matrix):
        raise ValueError("matrix is not 169x169")

    # Symmetry + range sanity (exact: m[i][j] + m[j][i] == 1, diagonal ~0.5).
    for i in range(169):
        if not (0.0 <= matrix[i][i] <= 1.0):
            raise ValueError(f"diagonal {classes[i]} out of range: {matrix[i][i]}")
        for j in range(i + 1, 169):
            s = matrix[i][j] + matrix[j][i]
            if abs(s - 1.0) > 1e-9:
                raise ValueError(f"asymmetry {classes[i]}/{classes[j]}: {matrix[i][j]}+{matrix[j][i]}={s}")
    return classes, matrix


def _write_parquet(classes: list[str], matrix: list[list[float]], out: Path) -> int:
    hero_col: list[str] = []
    vill_col: list[str] = []
    eq_col: list[float] = []
    for i, hero in enumerate(classes):
        for j, vill in enumerate(classes):
            hero_col.append(hero)
            vill_col.append(vill)
            eq_col.append(matrix[i][j])
    table = pa.table(
        {
            "hero_class": pa.array(hero_col, pa.string()),
            "villain_class": pa.array(vill_col, pa.string()),
            "equity": pa.array(eq_col, pa.float64()),
        }
    )
    pq.write_table(table, out, compression="zstd")
    return len(eq_col)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build exact 169x169 preflop equity parquet")
    ap.add_argument(
        "--matrix-json",
        type=Path,
        default=REPO_ROOT / "tools" / "preflop_hand_vs_hand_169.json",
        help="JSON emitted by the preflop-matrix binary",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tools" / "preflop_hand_vs_hand_169.parquet",
    )
    ap.add_argument("--build", action="store_true", help="invoke the Rust binary first")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--log-file", type=Path, default=None)
    args = ap.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)

    if args.build:
        _run_binary(args.matrix_json, args.threads)

    classes, matrix = _load_and_validate(args.matrix_json)
    n = _write_parquet(classes, matrix, args.out)
    log.info("wrote %s (%d rows, 169x169 exact, symmetry verified)", args.out, n)


if __name__ == "__main__":
    main()
