"""Fit per-dim z-score manifests from chunked embeddings (Phase 2 Plan 05).

Reads chunked Parquet via pyarrow.dataset (no full-RAM materialization).
Two-pass numpy fit per collection. Writes per-collection JSON manifests
to be loaded by tools/upsert_milvus.py at upsert time.

Usage:
    python tools/zscore_fit.py \
        --chunks-dir research/preflop-ranges/outputs/ \
        --out-preflop tools/zscore_preflop.json \
        --out-postflop tools/zscore_postflop.json \
        [--log-file research/preflop-ranges/outputs/phase2_build.log]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src._log import configure_logging, get_logger

log = get_logger("tools.zscore_fit")

FEATURE_SPEC_VERSION = 3
PREFLOP_DIM = 34
POSTFLOP_DIM = 80
BATCH_SIZE = 10_000  # pyarrow.dataset batch


def fit_collection(chunk_dir: Path, dim: int, street_class: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Two-pass mean/std over chunked Parquet for a given street_class.

    Pass 1: sum per dim -> mean
    Pass 2: sum of (x - mean)^2 per dim -> variance -> std

    std<1e-9 dims are flagged + replaced with 1.0 in the returned std array.

    Returns:
        (mean, std, n_total) — all float64 arrays of shape (dim,).

    Raises:
        RuntimeError: if no rows are found for the given street_class.
    """
    # Production chunks are split per street: hm_embeddings_chunk_NNNN_{preflop,postflop}.parquet.
    # Test fixtures use a single file per chunk: hm_embeddings_chunk_NNNN.parquet.
    # Filter discovery so we never read sibling docs (e.g. preflop-range JSONs in OUTPUT_DIR).
    chunk_files = sorted(chunk_dir.glob(f"hm_embeddings_chunk_*_{street_class}.parquet"))
    if not chunk_files:
        chunk_files = sorted(chunk_dir.glob("hm_embeddings_chunk_*.parquet"))
    if not chunk_files:
        raise RuntimeError(
            f"No chunk parquets found in {chunk_dir} matching "
            f"hm_embeddings_chunk_*_{street_class}.parquet or hm_embeddings_chunk_*.parquet"
        )
    dataset = ds.dataset([str(p) for p in chunk_files], format="parquet")
    row_filter = ds.field("street_class") == street_class
    # Pass 1 — mean
    sum_per_dim = np.zeros(dim, dtype=np.float64)
    n_total = 0
    for batch in dataset.to_batches(batch_size=BATCH_SIZE, filter=row_filter):
        if batch.num_rows == 0:
            continue
        arr = np.stack(batch.column("embedding").to_numpy(zero_copy_only=False))
        sum_per_dim += arr.sum(axis=0)
        n_total += arr.shape[0]
    if n_total == 0:
        raise RuntimeError(f"No rows found in {chunk_dir} for street_class={street_class!r}")
    mean = sum_per_dim / n_total

    # Pass 2 — variance using true mean (avoids catastrophic cancellation)
    sumsq_per_dim = np.zeros(dim, dtype=np.float64)
    for batch in dataset.to_batches(batch_size=BATCH_SIZE, filter=row_filter):
        if batch.num_rows == 0:
            continue
        arr = np.stack(batch.column("embedding").to_numpy(zero_copy_only=False))
        sumsq_per_dim += ((arr - mean) ** 2).sum(axis=0)
    std_raw = np.sqrt(sumsq_per_dim / n_total)

    # Zero-std safeguard.
    # Threshold 1e-6 rather than 1e-9: float32 embeddings converted to float64 for
    # the two-pass sum can produce tiny residuals (≈ 4e-8) for truly constant dims
    # due to float32→float64 precision loss in the mean. Real non-constant dims have
    # std ≫ 1e-4 in practice, so 1e-6 is safely above float32 noise but well below
    # any meaningful variance.
    zero_std_dims = np.where(std_raw < 1e-6)[0]
    if len(zero_std_dims) > 0:
        log.warning(
            "zscore_fit.zero_std_dims",
            collection=street_class,
            dims=zero_std_dims.tolist(),
            n=int(n_total),
        )
    std = np.where(std_raw < 1e-6, 1.0, std_raw)
    return mean, std, n_total


def write_manifest(
    path: Path,
    *,
    collection: str,
    mean: np.ndarray,
    std: np.ndarray,
    n_total: int,
    dim_names: list[str] | None = None,
) -> None:
    """Write the JSON manifest per CONTEXT.md Decision 1 format.

    Args:
        path: destination file path (parent dirs created if needed).
        collection: collection identifier, e.g. "preflop_decisions".
        mean: float64 array of per-dim means, shape (dim,).
        std: float64 array of per-dim stds (zero-guarded), shape (dim,).
        n_total: number of rows used in the fit.
        dim_names: optional list of dim labels; defaults to "dim_000"…"dim_NNN".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if dim_names is None:
        dim_names = [f"dim_{i:03d}" for i in range(len(mean))]
    manifest = {
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "fit_population_n": int(n_total),
        "fit_date": datetime.date.today().isoformat(),
        "collection": collection,
        "dim_stats": [
            {
                "dim": i,
                "name": dim_names[i],
                "mean": float(mean[i]),
                "std": float(std[i]),
            }
            for i in range(len(mean))
        ],
    }
    path.write_text(json.dumps(manifest, indent=2))
    log.info(
        "zscore_fit.manifest_written",
        path=str(path),
        collection=collection,
        n=int(n_total),
        dims=len(mean),
    )


def load_manifest(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of write_manifest — returns (mean, std) for Plan 06 upsert.

    Raises:
        RuntimeError: if feature_spec_version != FEATURE_SPEC_VERSION.
    """
    m = json.loads(path.read_text())
    if m.get("feature_spec_version") != FEATURE_SPEC_VERSION:
        raise RuntimeError(
            f"Manifest {path} has feature_spec_version={m.get('feature_spec_version')}; "
            f"expected {FEATURE_SPEC_VERSION}"
        )
    mean = np.array([d["mean"] for d in m["dim_stats"]], dtype=np.float64)
    std = np.array([d["std"] for d in m["dim_stats"]], dtype=np.float64)
    return mean, std


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit per-dim z-score manifests from chunked embeddings")
    ap.add_argument("--chunks-dir", required=True, type=Path)
    ap.add_argument("--out-preflop", required=True, type=Path)
    ap.add_argument("--out-postflop", required=True, type=Path)
    ap.add_argument("--log-file", type=Path, default=None)
    args = ap.parse_args()

    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
    if not args.chunks_dir.exists():
        log.error("zscore_fit.chunks_dir_missing", path=str(args.chunks_dir))
        return 1

    for collection, dim, out_path in (
        ("preflop_decisions", PREFLOP_DIM, args.out_preflop),
        ("postflop_decisions", POSTFLOP_DIM, args.out_postflop),
    ):
        street_class = collection.replace("_decisions", "")  # "preflop" or "postflop"
        log.info("zscore_fit.start", collection=collection)
        mean, std, n_total = fit_collection(args.chunks_dir, dim, street_class)
        write_manifest(out_path, collection=collection, mean=mean, std=std, n_total=n_total)
        log.info("zscore_fit.complete", collection=collection, n=n_total, dims=dim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
