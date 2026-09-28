"""Batched idempotent Milvus upsert with z-score -> weights apply order (Phase 2 Plan 06).

Pipeline per row:
  raw_vec (from chunk parquet, post-min-max)
    -> z-score: (vec - mean) / std        # per CONTEXT.md Decision 1
    -> weighted: z * group_weights        # group weights last
    -> upsert with PK=decision_id, scalar fields, feature_spec_version=2

Modes:
  --rebuild   : drop both collections + setup_milvus_collection.setup(); first-load uses client.insert()
  (default)   : idempotent re-run via client.upsert() by decision_id PK

Usage:
    python tools/upsert_milvus.py \\
        --chunks-dir research/preflop-ranges/outputs/ \\
        --preflop-manifest tools/zscore_preflop.json \\
        --postflop-manifest tools/zscore_postflop.json \\
        [--rebuild] [--batch-size 5000] [--log-file research/preflop-ranges/outputs/phase2_build.log]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src._log import configure_logging, get_logger
from src.db.milvus import connect
from tools.feature_extractors.postflop import _GROUP_WEIGHTS as POSTFLOP_GROUP_WEIGHTS
from tools.zscore_fit import load_manifest

log = get_logger("tools.upsert_milvus")

FEATURE_SPEC_VERSION = 3
DEFAULT_BATCH_SIZE = 5000  # CONTEXT.md Decision 2B
PROGRESS_EVERY = 1_000  # OBS-03

PHASE1_FILTER_TEST_IDS = [
    "filter-test-hu-preflop_decisions",
    "filter-test-mw-preflop_decisions",
    "filter-test-hu-postflop_decisions",
    "filter-test-mw-postflop_decisions",
]


# --- Weight vector construction -----------------------------------------------

# Postflop group dim counts per FEATURES-v2.md §Feature Group Weights:
# A=12, B=8, C=6, D=8, E=7, F=16, G=6, H=7, J=10  -> sum = 80
_POSTFLOP_GROUP_DIM_COUNTS = [
    ("A", 12),
    ("B", 8),
    ("C", 6),
    ("D", 8),
    ("E", 7),
    ("F", 16),
    ("G", 6),
    ("H", 7),
    ("J", 10),
]


def build_postflop_weight_vector() -> np.ndarray:
    """80-dim weight vector; group dim counts per FEATURES-v2.md; weights from _GROUP_WEIGHTS."""
    parts = []
    for grp, n in _POSTFLOP_GROUP_DIM_COUNTS:
        parts.extend([POSTFLOP_GROUP_WEIGHTS[grp]] * n)
    w = np.array(parts, dtype=np.float64)
    assert w.shape == (80,), f"postflop weight vector wrong shape: {w.shape}"
    return w


def build_preflop_weight_vector() -> np.ndarray:
    """32-dim weight vector derived from preflop extractor group structure.

    Tries to import _GROUP_WEIGHTS and _PREFLOP_DIM_LAYOUT from the preflop extractor.
    Falls back to identity (all-ones) with a WARN log if the extractor exposes no group layout.
    """
    try:
        from tools.feature_extractors.preflop import _GROUP_WEIGHTS as PREFLOP_GW  # type: ignore
        from tools.feature_extractors.preflop import _PREFLOP_DIM_LAYOUT  # type: ignore

        parts = []
        for grp, n in _PREFLOP_DIM_LAYOUT:
            parts.append(np.full(n, PREFLOP_GW.get(grp, 1.0)))
        w = np.concatenate(parts)
    except (ImportError, AttributeError):
        log.warning(
            "upsert_milvus.preflop_weights_default_ones",
            reason="preflop extractor exposes no group layout; using identity weighting",
        )
        w = np.ones(34, dtype=np.float64)
    assert w.shape == (34,), f"preflop weight vector wrong shape: {w.shape}"
    return w


# --- Normalization -------------------------------------------------------------


def normalize(vec: np.ndarray, mean: np.ndarray, std: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply CONTEXT.md Decision 1 order: (x - mean) / std THEN * weights."""
    z = (vec - mean) / std
    return z * weights


# --- Milvus helpers -----------------------------------------------------------


def _milvus_client_from_env():
    """Build MilvusClient from env vars (MILVUS_HOST, MILVUS_PORT, MILVUS_TOKEN)."""
    host = os.environ["MILVUS_HOST"]
    port = os.environ["MILVUS_PORT"]
    raw = os.environ.get("MILVUS_TOKEN")
    token = raw if (raw and ":" in raw) else (f"root:{raw}" if raw else None)
    uri = f"http://{host}:{port}"
    return connect(uri, token=token), uri, token


def _cleanup_filter_test_residue(client) -> int:
    """Delete Phase 1 filter-test-* rows from both collections. Returns total rows deleted.

    Uses explicit decision_id list form (more robust than `like` per PATTERNS.md Analog C).
    """
    total = 0
    for collection in ("preflop_decisions", "postflop_decisions"):
        if not client.has_collection(collection):
            continue
        id_list = [f'"{i}"' for i in PHASE1_FILTER_TEST_IDS]
        filt = f"decision_id in [{','.join(id_list)}]"
        try:
            res = client.delete(collection_name=collection, filter=filt)
            client.flush(collection)
            n = getattr(res, "delete_count", 0) if not isinstance(res, dict) else res.get("delete_count", 0)
            total += int(n)
            log.info("upsert_milvus.filter_test_cleanup", collection=collection, deleted=int(n))
        except Exception as e:
            # A swallowed delete leaves filter-test-* rows live in the served
            # index, polluting kNN neighbors — fail loud so the upsert aborts.
            log.error(
                "upsert_milvus.filter_test_cleanup_failed",
                collection=collection,
                error=str(e),
            )
            raise
    return total


def rebuild(client, uri: str | None = None, token: str | None = None) -> None:
    """Drop both collections + re-run setup_milvus_collection.setup().

    Args:
        client: active MilvusClient (used for drop).
        uri: Milvus server URI for setup(); if None, read from MILVUS_HOST/MILVUS_PORT env vars.
        token: auth token for setup(); if None, read from MILVUS_TOKEN env var.
    """
    for coll in ("preflop_decisions", "postflop_decisions"):
        if client.has_collection(coll):
            client.drop_collection(coll)
            log.info("upsert_milvus.collection_dropped", collection=coll)

    # Resolve uri/token for setup() if not provided
    if uri is None:
        host = os.environ["MILVUS_HOST"]
        port = os.environ["MILVUS_PORT"]
        uri = f"http://{host}:{port}"
    if token is None:
        raw = os.environ.get("MILVUS_TOKEN")
        token = raw if (raw and ":" in raw) else (f"root:{raw}" if raw else None)

    from scripts.setup_milvus_collection import setup

    setup(uri, token=token)
    log.info("upsert_milvus.rebuild_complete")


# --- Upsert pipeline ----------------------------------------------------------


def _upsert_one_collection(
    client,
    collection_name: str,
    chunks_dir: Path,
    manifest_path: Path,
    weights: np.ndarray,
    dim: int,
    mode: str,
    batch_size: int,
) -> int:
    """Upsert or insert rows for one collection. Returns total rows processed."""
    mean, std = load_manifest(manifest_path)
    assert mean.shape == (dim,), f"manifest dim mismatch: {mean.shape} vs {dim}"
    street_class = collection_name.replace("_decisions", "")
    chunk_files = sorted(chunks_dir.glob(f"hm_embeddings_chunk_*_{street_class}.parquet"))
    if not chunk_files:
        chunk_files = sorted(chunks_dir.glob("hm_embeddings_chunk_*.parquet"))
    if not chunk_files:
        raise RuntimeError(
            f"No chunk parquets found in {chunks_dir} matching "
            f"hm_embeddings_chunk_*_{street_class}.parquet or hm_embeddings_chunk_*.parquet"
        )
    dataset = ds.dataset([str(p) for p in chunk_files], format="parquet")
    op = client.insert if mode == "insert" else client.upsert
    batch: list[dict] = []
    total = 0
    t0 = time.time()
    for record_batch in dataset.to_batches(
        batch_size=batch_size,
        filter=ds.field("street_class") == street_class,
    ):
        ids = record_batch.column("id").to_pylist()
        embs = record_batch.column("embedding").to_pylist()
        street_classes = record_batch.column("street_class").to_pylist()
        pot_types = record_batch.column("pot_type").to_pylist()
        hero_pos_rels = record_batch.column("hero_pos_rel").to_pylist()
        n_players = record_batch.column("n_players_active").to_pylist()
        # Optional columns — older test fixtures may not include them.
        col_names = set(record_batch.schema.names)
        if "street" in col_names:
            streets = record_batch.column("street").to_pylist()
        else:
            streets = [street_class for street_class in street_classes]
        if "spr_x100" in col_names:
            spr_x100s = record_batch.column("spr_x100").to_pylist()
        else:
            spr_x100s = [-1] * record_batch.num_rows
        # hero_action_type is REQUIRED — fail loud if absent (no silent corruption
        # of blended distributions in DecisionEngine).
        if "hero_action_type" in col_names:
            hero_action_types = record_batch.column("hero_action_type").to_pylist()
        else:
            raise RuntimeError(
                f"Chunk parquet missing required column 'hero_action_type' for collection "
                f"{collection_name!r}. Re-run tools/build_embedding.py with the Phase 3 schema."
            )
        # active / confidence / gto_score are OPTIONAL with safe defaults — Phase 2
        # chunk parquets predate these fields; back-fill them on subsequent rebuilds.
        # Defaults: active=True (DPs are live), confidence=1.0 (no measured
        # confidence), gto_score=1.0 (unsolved). Uniform defaults preserve the
        # pre-Phase-3 blend behaviour bit-for-bit (weight collapses to similarity).
        n_rows = record_batch.num_rows
        actives = record_batch.column("active").to_pylist() if "active" in col_names else [True] * n_rows
        confidences = (
            record_batch.column("confidence").to_pylist() if "confidence" in col_names else [1.0] * n_rows
        )
        gto_scores = (
            record_batch.column("gto_score").to_pylist() if "gto_score" in col_names else [1.0] * n_rows
        )
        # Raw postflop-snap inputs — optional with defaults for pre-snap parquets.
        size_pot_fracs = (
            record_batch.column("hero_action_size_pot_frac").to_pylist()
            if "hero_action_size_pot_frac" in col_names
            else [0.0] * n_rows
        )
        raise_ratios = (
            record_batch.column("raise_ratio").to_pylist() if "raise_ratio" in col_names else [-1.0] * n_rows
        )
        allins = (
            record_batch.column("hero_action_allin").to_pylist()
            if "hero_action_allin" in col_names
            else [False] * n_rows
        )
        for i in range(record_batch.num_rows):
            raw = np.asarray(embs[i], dtype=np.float64)
            weighted = normalize(raw, mean, std, weights)
            batch.append(
                {
                    "decision_id": ids[i],
                    "embedding": weighted.tolist(),
                    "street_class": street_classes[i],
                    "street": streets[i],
                    "pot_type": pot_types[i],
                    "hero_pos_rel": hero_pos_rels[i],
                    "n_players_active": int(n_players[i]),
                    "spr_x100": int(spr_x100s[i]),
                    "hero_action_type": hero_action_types[i] or "",
                    "hero_action_size_pot_frac": float(
                        size_pot_fracs[i] if size_pot_fracs[i] is not None else 0.0
                    ),
                    "raise_ratio": float(raise_ratios[i] if raise_ratios[i] is not None else -1.0),
                    "hero_action_allin": bool(allins[i]) if allins[i] is not None else False,
                    "active": bool(actives[i]) if actives[i] is not None else True,
                    "confidence": float(confidences[i]) if confidences[i] is not None else 1.0,
                    "gto_score": float(gto_scores[i]) if gto_scores[i] is not None else 1.0,
                    "feature_spec_version": FEATURE_SPEC_VERSION,
                    "action_dist": "",
                    "added_at": 0,
                    "removed_at": 0,
                }
            )
            if len(batch) >= batch_size:
                op(collection_name=collection_name, data=batch)
                total += len(batch)
                batch = []
                if total % PROGRESS_EVERY == 0:
                    log.info(
                        "upsert_milvus.progress",
                        collection=collection_name,
                        n=total,
                        elapsed_s=round(time.time() - t0, 1),
                    )
    if batch:
        op(collection_name=collection_name, data=batch)
        total += len(batch)
    client.flush(collection_name)
    log.info("upsert_milvus.complete", collection=collection_name, n=total, mode=mode)
    return total


def upsert_all(
    chunks_dir: Path,
    preflop_manifest: Path,
    postflop_manifest: Path,
    *,
    mode: str = "upsert",
    batch_size: int = DEFAULT_BATCH_SIZE,
    client=None,
) -> dict[str, int]:
    """Upsert both collections. Returns {collection_name: row_count}.

    Args:
        chunks_dir: directory containing chunked parquet files.
        preflop_manifest: path to preflop z-score manifest JSON (from Plan 05).
        postflop_manifest: path to postflop z-score manifest JSON (from Plan 05).
        mode: "upsert" (idempotent default) or "insert" (first-load rebuild path).
        batch_size: rows per upsert/insert call (default 5000).
        client: optional pre-built MilvusClient; created from env vars if None.
    """
    if client is None:
        client, _, _ = _milvus_client_from_env()
    weights_pre = build_preflop_weight_vector()
    weights_post = build_postflop_weight_vector()
    pre_n = _upsert_one_collection(
        client,
        "preflop_decisions",
        chunks_dir,
        preflop_manifest,
        weights_pre,
        dim=34,
        mode=mode,
        batch_size=batch_size,
    )
    post_n = _upsert_one_collection(
        client,
        "postflop_decisions",
        chunks_dir,
        postflop_manifest,
        weights_post,
        dim=80,
        mode=mode,
        batch_size=batch_size,
    )
    return {"preflop_decisions": pre_n, "postflop_decisions": post_n}


# --- CLI ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Batched Milvus upsert with z-score -> weights apply order")
    ap.add_argument("--chunks-dir", required=True, type=Path)
    ap.add_argument("--preflop-manifest", required=True, type=Path)
    ap.add_argument("--postflop-manifest", required=True, type=Path)
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop + recreate both collections; uses client.insert() (no tombstones)",
    )
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--log-file", type=Path, default=None)
    args = ap.parse_args()

    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
    if not args.chunks_dir.exists():
        log.error("upsert_milvus.chunks_dir_missing", path=str(args.chunks_dir))
        return 1
    for p in (args.preflop_manifest, args.postflop_manifest):
        if not p.exists():
            log.error("upsert_milvus.manifest_missing", path=str(p))
            return 1

    client, uri, token = _milvus_client_from_env()
    if args.rebuild:
        rebuild(client, uri=uri, token=token)
        mode = "insert"
    else:
        _cleanup_filter_test_residue(client)
        mode = "upsert"

    result = upsert_all(
        args.chunks_dir,
        args.preflop_manifest,
        args.postflop_manifest,
        mode=mode,
        batch_size=args.batch_size,
        client=client,
    )
    log.info("upsert_milvus.summary", **result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
