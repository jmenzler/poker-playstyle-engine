"""STOR-05: Milvus dual-collection schema + scalar hard-filter behavior.

AMENDMENT (2026-05-17 kNN pivot, FEATURES-v2.md): Tests TWO collections:
  - preflop_decisions  (34-dim HNSW COSINE)
  - postflop_decisions (80-dim HNSW COSINE)

Both must have scalar hard-filter fields:
  street_class, pot_type, hero_pos_rel, n_players_active
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

# (collection_name, embedding_dim)
EXPECTED_COLLECTIONS = [
    ("preflop_decisions", 34),
    ("postflop_decisions", 80),
]

EXPECTED_SCALAR_FIELDS = {
    "street_class",
    "pot_type",
    "hero_pos_rel",
    "n_players_active",
    "feature_spec_version",
}


@pytest.mark.parametrize("collection_name,expected_dim", EXPECTED_COLLECTIONS)
def test_collection_exists_with_hnsw_cosine(
    milvus_uri: str,
    milvus_token: str | None,
    collection_name: str,
    expected_dim: int,
) -> None:
    """Collection exists with correct dim and scalar hard-filter fields."""
    from src.db.milvus import connect

    client = connect(milvus_uri, token=milvus_token)
    assert client.has_collection(collection_name), f"Collection {collection_name!r} not found"

    desc = client.describe_collection(collection_name)
    fields_by_name = {f.get("name"): f for f in desc.get("fields", [])}

    # Embedding field with correct dim
    assert "embedding" in fields_by_name, f"No 'embedding' field in {collection_name}"
    emb = fields_by_name["embedding"]
    params = emb.get("params") or emb.get("type_params") or {}
    dim = params.get("dim") or emb.get("dim")
    assert int(dim) == expected_dim, f"{collection_name} embedding dim={dim}, expected {expected_dim}"

    # Scalar hard-filter fields
    missing_scalars = EXPECTED_SCALAR_FIELDS - set(fields_by_name.keys())
    assert not missing_scalars, f"{collection_name} missing scalar fields: {missing_scalars}"


@pytest.mark.parametrize("collection_name,dim", EXPECTED_COLLECTIONS)
def test_scalar_filter_works(
    milvus_uri: str,
    milvus_token: str | None,
    collection_name: str,
    dim: int,
) -> None:
    """Insert two vectors differing only in n_players_active; filter returns only HU (2)."""
    from src.db.milvus import connect

    client = connect(milvus_uri, token=milvus_token)

    base_vec = [0.01] * dim
    other_vec = [0.02] * dim
    test_data = [
        {
            "decision_id": f"filter-test-hu-{collection_name}",
            "embedding": base_vec,
            "street_class": "preflop" if dim == 34 else "postflop",
            "pot_type": "srp",
            "hero_pos_rel": "IP",
            "n_players_active": 2,
            "feature_spec_version": 3,
        },
        {
            "decision_id": f"filter-test-mw-{collection_name}",
            "embedding": other_vec,
            "street_class": "preflop" if dim == 34 else "postflop",
            "pot_type": "srp",
            "hero_pos_rel": "IP",
            "n_players_active": 3,
            "feature_spec_version": 3,
        },
    ]
    client.insert(collection_name=collection_name, data=test_data)
    # Flush to ensure vectors are indexed before searching (Milvus eventual consistency).
    client.flush(collection_name)
    try:
        results = client.search(
            collection_name=collection_name,
            data=[base_vec],
            limit=10,
            filter="n_players_active == 2",
            output_fields=["decision_id"],
        )
        ids = {r["entity"]["decision_id"] for r in results[0]}
        assert f"filter-test-hu-{collection_name}" in ids
        assert f"filter-test-mw-{collection_name}" not in ids
    finally:
        client.delete(
            collection_name=collection_name,
            filter=f'decision_id in ["filter-test-hu-{collection_name}","filter-test-mw-{collection_name}"]',
        )
