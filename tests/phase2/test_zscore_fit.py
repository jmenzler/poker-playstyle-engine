"""Tests for tools/zscore_fit.py — numeric stability + manifest schema tests.

Activated by Phase 2 Plan 05. Each test encodes a locked numeric contract for
the two-pass z-score fitter. Implementation must NOT be changed to pass these
tests; the tests define the contract.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Repo-root path injection so we can import tools.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# pytestmark — uncomment when this file's tests need the live Milvus stack
# pytestmark = pytest.mark.integration


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_chunked_parquet(
    tmp_dir: Path,
    n_rows_per_chunk: list[int],
    dim: int,
    street_class: str,
    *,
    seed: int = 99,
    constant_dim: int | None = None,
    constant_value: float = 0.5,
) -> Path:
    """Write synthetic chunked parquet files to tmp_dir.

    Args:
        tmp_dir: directory to write parquet files into.
        n_rows_per_chunk: list of row counts; one file per entry.
        dim: embedding dimensionality.
        street_class: value for the street_class column.
        seed: RNG seed (deterministic).
        constant_dim: if set, that dimension is forced to constant_value for all rows.
        constant_value: value used when constant_dim is set.
    """
    rng = np.random.default_rng(seed=seed)
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("hand_id", pa.string()),
            pa.field("street_class", pa.string()),
            pa.field("pot_type", pa.string()),
            pa.field("hero_pos_rel", pa.string()),
            pa.field("hero_pos", pa.string()),
            pa.field("n_players_active", pa.int32()),
            pa.field("embedding", pa.list_(pa.float32(), dim)),
            pa.field("schema_version", pa.int32()),
        ]
    )
    for chunk_idx, n in enumerate(n_rows_per_chunk, start=1):
        embeddings = rng.random((n, dim)).astype(np.float32)
        if constant_dim is not None:
            embeddings[:, constant_dim] = constant_value
        table = pa.table(
            {
                "id": [f"c{chunk_idx}_dp{i}" for i in range(n)],
                "hand_id": [f"c{chunk_idx}_h{i}" for i in range(n)],
                "street_class": [street_class] * n,
                "pot_type": ["srp"] * n,
                "hero_pos_rel": ["IP"] * n,
                "hero_pos": ["BTN"] * n,
                "n_players_active": [2] * n,
                "embedding": embeddings.tolist(),
                "schema_version": [2] * n,
            },
            schema=schema,
        )
        pq.write_table(
            table,
            tmp_dir / f"hm_embeddings_chunk_{chunk_idx:04d}.parquet",
        )
    return tmp_dir


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_two_pass_mean_std_matches_numpy_reference(
    chunked_parquet_dir_postflop: Path,
) -> None:
    """Two-pass numpy fit over pyarrow.dataset == numpy.mean / numpy.std on materialized array."""
    from tools.zscore_fit import fit_collection

    dim = 80
    mean, std, n_total = fit_collection(chunked_parquet_dir_postflop, dim=dim, street_class="postflop")

    # Materialise all embeddings the naive way (for reference comparison only)
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(chunked_parquet_dir_postflop), format="parquet")
    all_arrays = []
    for batch in dataset.to_batches():
        arr = np.stack(batch.column("embedding").to_numpy(zero_copy_only=False))
        all_arrays.append(arr)
    materialized = np.concatenate(all_arrays, axis=0)

    ref_mean = np.mean(materialized, axis=0)
    ref_std = np.std(materialized, axis=0)

    assert mean.shape == (dim,)
    assert std.shape == (dim,)
    assert n_total == materialized.shape[0]
    np.testing.assert_allclose(mean, ref_mean, atol=1e-6, rtol=0)
    np.testing.assert_allclose(std, ref_std, atol=1e-6, rtol=0)


def test_zero_std_dim_replaced_with_one(tmp_path: Path) -> None:
    """Constant dim (all values equal) -> std=1.0 in manifest + WARN log emitted."""
    from tools.zscore_fit import fit_collection, write_manifest

    dim = 80
    constant_dim_idx = 17
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    _make_chunked_parquet(
        chunk_dir,
        n_rows_per_chunk=[40, 40],
        dim=dim,
        street_class="postflop",
        constant_dim=constant_dim_idx,
        constant_value=0.42,
    )

    # Capture warning via monkeypatching the module-level logger
    warned_events: list[dict] = []

    import tools.zscore_fit as zscore_module

    original_warning = zscore_module.log.warning

    def capture_warning(event: str, **kwargs):  # type: ignore[override]
        warned_events.append({"event": event, **kwargs})
        return original_warning(event, **kwargs)

    zscore_module.log.warning = capture_warning  # type: ignore[method-assign]
    try:
        mean, std, n_total = fit_collection(chunk_dir, dim=dim, street_class="postflop")
    finally:
        zscore_module.log.warning = original_warning

    # std for constant dim must be exactly 1.0 (the guard value)
    assert std[constant_dim_idx] == 1.0, f"Expected std[{constant_dim_idx}]=1.0, got {std[constant_dim_idx]}"

    # All other dims must have std > 0 (random data)
    other_dims = [i for i in range(dim) if i != constant_dim_idx]
    assert all(std[i] > 0 for i in other_dims)

    # A WARN log must have been emitted naming the zero-std dim
    assert any(
        e["event"] == "zscore_fit.zero_std_dims" and constant_dim_idx in e.get("dims", [])
        for e in warned_events
    ), (
        f"Expected warning event 'zscore_fit.zero_std_dims' listing dim {constant_dim_idx}; got {warned_events}"
    )

    # Also verify in the manifest
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        collection="postflop_decisions",
        mean=mean,
        std=std,
        n_total=n_total,
    )
    m = json.loads(manifest_path.read_text())
    assert m["dim_stats"][constant_dim_idx]["std"] == 1.0


def test_manifest_schema_v2_round_trip(chunked_parquet_dir_postflop: Path, tmp_path: Path) -> None:
    """Manifest writes valid JSON with feature_spec_version=2, dim_stats list of {dim, name, mean, std}."""
    from tools.zscore_fit import fit_collection, load_manifest, write_manifest

    dim = 80
    mean, std, n_total = fit_collection(chunked_parquet_dir_postflop, dim=dim, street_class="postflop")

    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        collection="postflop_decisions",
        mean=mean,
        std=std,
        n_total=n_total,
    )

    # Load via stdlib json
    m = json.loads(manifest_path.read_text())

    # Schema shape assertions
    assert m["feature_spec_version"] == 3
    assert m["collection"] == "postflop_decisions"
    assert m["fit_population_n"] == 100  # 50 + 50 from chunked_parquet_dir_postflop fixture
    assert re.match(r"\d{4}-\d{2}-\d{2}", m["fit_date"]), f"Bad fit_date: {m['fit_date']}"
    assert len(m["dim_stats"]) == 80

    for entry in m["dim_stats"]:
        assert set(entry.keys()) >= {"dim", "name", "mean", "std"}
        assert isinstance(entry["dim"], int)
        assert isinstance(entry["name"], str)
        assert isinstance(entry["mean"], float)
        assert isinstance(entry["std"], float)

    # Round-trip via load_manifest
    loaded_mean, loaded_std = load_manifest(manifest_path)
    np.testing.assert_allclose(loaded_mean, mean, atol=1e-12)
    np.testing.assert_allclose(loaded_std, std, atol=1e-12)


def test_separate_preflop_postflop_manifests(tmp_path: Path) -> None:
    """One manifest per collection; preflop has 32 dims, postflop has 80 dims.

    In production, preflop and postflop chunks live in separate directories
    (different embedding sizes: 32d vs 80d). Each fit call receives its own
    directory so pyarrow.dataset sees a homogeneous schema within each directory.
    """
    from tools.zscore_fit import fit_collection, write_manifest

    preflop_dir = tmp_path / "chunks_preflop"
    preflop_dir.mkdir()
    postflop_dir = tmp_path / "chunks_postflop"
    postflop_dir.mkdir()

    # Write preflop chunks (32d)
    preflop_n = [30, 20]
    _make_chunked_parquet(
        preflop_dir,
        n_rows_per_chunk=preflop_n,
        dim=32,
        street_class="preflop",
        seed=11,
    )

    # Write postflop chunks (80d) — separate directory
    postflop_n = [40, 35]
    _make_chunked_parquet(
        postflop_dir,
        n_rows_per_chunk=postflop_n,
        dim=80,
        street_class="postflop",
        seed=22,
    )

    # Fit preflop collection (32d) from preflop-only directory
    preflop_mean, preflop_std, preflop_n_total = fit_collection(preflop_dir, dim=32, street_class="preflop")
    # Fit postflop collection (80d) from postflop-only directory
    postflop_mean, postflop_std, postflop_n_total = fit_collection(
        postflop_dir, dim=80, street_class="postflop"
    )

    # Write two distinct manifests
    preflop_manifest_path = tmp_path / "zscore_preflop.json"
    postflop_manifest_path = tmp_path / "zscore_postflop.json"

    write_manifest(
        preflop_manifest_path,
        collection="preflop_decisions",
        mean=preflop_mean,
        std=preflop_std,
        n_total=preflop_n_total,
    )
    write_manifest(
        postflop_manifest_path,
        collection="postflop_decisions",
        mean=postflop_mean,
        std=postflop_std,
        n_total=postflop_n_total,
    )

    # Load and assert independently
    pre_m = json.loads(preflop_manifest_path.read_text())
    post_m = json.loads(postflop_manifest_path.read_text())

    assert len(pre_m["dim_stats"]) == 32, f"Preflop must have 32 dims; got {len(pre_m['dim_stats'])}"
    assert len(post_m["dim_stats"]) == 80, f"Postflop must have 80 dims; got {len(post_m['dim_stats'])}"

    assert pre_m["collection"] == "preflop_decisions"
    assert post_m["collection"] == "postflop_decisions"

    # fit_population_n must reflect the respective subset
    assert pre_m["fit_population_n"] == sum(preflop_n), (
        f"Expected preflop n={sum(preflop_n)}, got {pre_m['fit_population_n']}"
    )
    assert post_m["fit_population_n"] == sum(postflop_n), (
        f"Expected postflop n={sum(postflop_n)}, got {post_m['fit_population_n']}"
    )

    # Manifests must differ (different dims, different data)
    assert pre_m["fit_population_n"] != post_m["fit_population_n"]
