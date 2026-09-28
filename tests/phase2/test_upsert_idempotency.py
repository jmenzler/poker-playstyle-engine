"""Integration tests for tools/upsert_milvus.py — upsert idempotency + Phase 1 residue cleanup.

Activated by Phase 2 Plan 06. Tests require a live Milvus instance; skip if MILVUS_HOST is unset.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

pytestmark = pytest.mark.integration

# Skip all tests in this module if Milvus is not reachable
_MILVUS_HOST = os.environ.get("MILVUS_HOST")
if not _MILVUS_HOST:
    pytestmark = [pytest.mark.integration, pytest.mark.skip(reason="MILVUS_HOST not set — deferred to PC")]


def _write_fake_manifest(path: Path, dim: int) -> None:
    """Write a zero-mean, unit-std manifest for testing (identity normalization)."""
    manifest = {
        "feature_spec_version": 3,
        "fit_population_n": 100,
        "fit_date": "2026-05-18",
        "collection": "postflop_decisions",
        "dim_stats": [{"dim": i, "name": f"dim_{i:03d}", "mean": 0.0, "std": 1.0} for i in range(dim)],
    }
    path.write_text(json.dumps(manifest, indent=2))


def _write_fake_chunks(chunks_dir: Path, decision_ids: list[str], dim: int, street_class: str) -> None:
    """Write a minimal chunked parquet file for testing."""
    n = len(decision_ids)
    rng = np.random.default_rng(seed=99)
    embeddings = rng.random((n, dim)).astype(np.float32).tolist()
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
    table = pa.table(
        {
            "id": decision_ids,
            "hand_id": decision_ids,
            "street_class": [street_class] * n,
            "pot_type": ["srp"] * n,
            "hero_pos_rel": ["IP"] * n,
            "hero_pos": ["BTN"] * n,
            "n_players_active": [2] * n,
            "embedding": embeddings,
            "schema_version": [2] * n,
        },
        schema=schema,
    )
    pq.write_table(table, chunks_dir / "hm_embeddings_chunk_0001.parquet")


def test_double_upsert_no_duplicates(milvus_uri, milvus_token, tmp_path) -> None:
    """BOOT-04: running upsert twice on same chunks -> entity count unchanged."""
    from src.db.milvus import connect
    from tools.upsert_milvus import upsert_all

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    decision_ids = [f"idem-test-pp-{i}" for i in range(10)]
    _write_fake_chunks(chunks_dir, decision_ids, dim=80, street_class="postflop")

    preflop_manifest = tmp_path / "preflop_manifest.json"
    postflop_manifest = tmp_path / "postflop_manifest.json"
    _write_fake_manifest(preflop_manifest, dim=34)
    _write_fake_manifest(postflop_manifest, dim=80)

    client = connect(milvus_uri, token=milvus_token)
    id_filter = f"decision_id in [{','.join(repr(d) for d in decision_ids)}]"
    try:
        # First upsert
        upsert_all(chunks_dir, preflop_manifest, postflop_manifest, client=client)
        client.flush("postflop_decisions")
        count1 = len(client.query("postflop_decisions", filter=id_filter, output_fields=["decision_id"]))

        # Second upsert — count must not change
        upsert_all(chunks_dir, preflop_manifest, postflop_manifest, client=client)
        client.flush("postflop_decisions")
        count2 = len(client.query("postflop_decisions", filter=id_filter, output_fields=["decision_id"]))

        assert count1 == 10, f"Expected 10 rows after first upsert, got {count1}"
        assert count2 == count1, f"Duplicate rows after second upsert: {count2} vs {count1}"
    finally:
        client.delete(collection_name="postflop_decisions", filter=id_filter)


def test_upsert_replaces_row_by_pk(milvus_uri, milvus_token) -> None:
    """Upserting decision_id='X' with new embedding -> query returns the new embedding, not stale."""
    from src.db.milvus import connect
    from tools.upsert_milvus import FEATURE_SPEC_VERSION

    client = connect(milvus_uri, token=milvus_token)
    test_id = "pk-replace-test-pp-01"
    id_filter = f'decision_id in ["{test_id}"]'
    try:
        # Insert row with all-zeros embedding
        client.upsert(
            collection_name="postflop_decisions",
            data=[
                {
                    "decision_id": test_id,
                    "embedding": [0.0] * 80,
                    "street_class": "postflop",
                    "pot_type": "srp",
                    "hero_pos_rel": "IP",
                    "n_players_active": 2,
                    "feature_spec_version": FEATURE_SPEC_VERSION,
                }
            ],
        )
        client.flush("postflop_decisions")

        # Upsert same PK with all-ones embedding
        client.upsert(
            collection_name="postflop_decisions",
            data=[
                {
                    "decision_id": test_id,
                    "embedding": [1.0] * 80,
                    "street_class": "postflop",
                    "pot_type": "srp",
                    "hero_pos_rel": "IP",
                    "n_players_active": 2,
                    "feature_spec_version": FEATURE_SPEC_VERSION,
                }
            ],
        )
        client.flush("postflop_decisions")

        results = client.query(
            collection_name="postflop_decisions",
            filter=id_filter,
            output_fields=["decision_id", "embedding"],
        )
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        emb = results[0]["embedding"]
        assert all(v == pytest.approx(1.0) for v in emb), "Embedding was not replaced by last-write PK"
    finally:
        client.delete(collection_name="postflop_decisions", filter=id_filter)


def test_rebuild_flag_drops_and_recreates(milvus_uri, milvus_token, monkeypatch, tmp_path) -> None:
    """--rebuild path: collections dropped + setup_milvus_collection.setup() called."""
    import tools.upsert_milvus as um
    from src.db.milvus import connect

    client = connect(milvus_uri, token=milvus_token)

    # Ensure collections exist before rebuild
    from scripts.setup_milvus_collection import setup

    setup(milvus_uri, token=milvus_token)

    # Perform rebuild via the rebuild() function directly (CLI test would mutate sys.argv globally)
    um.rebuild(client)

    # After rebuild, collections must exist again
    assert client.has_collection("preflop_decisions"), "preflop_decisions missing after rebuild"
    assert client.has_collection("postflop_decisions"), "postflop_decisions missing after rebuild"

    # Verify feature_spec_version scalar field is in the schema
    for coll_name in ("preflop_decisions", "postflop_decisions"):
        desc = client.describe_collection(coll_name)
        fields_by_name = {f.get("name") for f in desc.get("fields", [])}
        assert "feature_spec_version" in fields_by_name, (
            f"{coll_name} missing feature_spec_version after rebuild"
        )


def test_phase1_filter_test_residue_cleaned(milvus_uri, milvus_token) -> None:
    """Pre-upsert cleanup removes filter-test-{hu,mw}-{preflop,postflop}_decisions rows."""
    from src.db.milvus import connect
    from tools.upsert_milvus import FEATURE_SPEC_VERSION, PHASE1_FILTER_TEST_IDS, _cleanup_filter_test_residue

    client = connect(milvus_uri, token=milvus_token)

    # Insert the 4 known Phase 1 filter-test rows
    for collection, dim, street in (
        ("preflop_decisions", 32, "preflop"),
        ("postflop_decisions", 80, "postflop"),
    ):
        ids_for_coll = [i for i in PHASE1_FILTER_TEST_IDS if collection.replace("_decisions", "") in i]
        if ids_for_coll:
            client.upsert(
                collection_name=collection,
                data=[
                    {
                        "decision_id": test_id,
                        "embedding": [0.01] * dim,
                        "street_class": street,
                        "pot_type": "srp",
                        "hero_pos_rel": "IP",
                        "n_players_active": 2,
                        "feature_spec_version": FEATURE_SPEC_VERSION,
                    }
                    for test_id in ids_for_coll
                ],
            )
            client.flush(collection)

    # Run cleanup
    deleted = _cleanup_filter_test_residue(client)
    assert deleted >= 4, f"Expected at least 4 rows deleted, got {deleted}"

    # Verify all 4 IDs are gone
    for collection in ("preflop_decisions", "postflop_decisions"):
        ids_for_coll = [i for i in PHASE1_FILTER_TEST_IDS if collection.replace("_decisions", "") in i]
        if ids_for_coll:
            id_filter = f"decision_id in [{','.join(repr(d) for d in ids_for_coll)}]"
            results = client.query(
                collection_name=collection, filter=id_filter, output_fields=["decision_id"]
            )
            assert len(results) == 0, (
                f"Expected 0 rows for filter-test IDs in {collection}, got {len(results)}: {results}"
            )
