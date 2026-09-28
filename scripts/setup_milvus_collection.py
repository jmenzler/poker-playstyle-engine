"""STOR-05: idempotent Milvus collection setup — dual-collection schema (FEATURES-v2.md).

AMENDMENT (2026-05-17 kNN pivot): Creates TWO collections, not one:
  - preflop_decisions  — 34-dim HNSW COSINE  (FEATURES-v2.md §Preflop Model)
  - postflop_decisions — 80-dim HNSW COSINE  (FEATURES-v2.md §Postflop Model)

Both collections share the same scalar hard-filter fields:
  - street_class    VARCHAR  (preflop | postflop)
  - street          VARCHAR  (preflop | flop | turn | river)
  - pot_type        VARCHAR  (limp | srp | 3bet | 4bet | 5bet+)
  - hero_pos_rel    VARCHAR  (OOP | IP)
  - n_players_active INT64   (2 | 3+)
  - spr_x100        INT64    SPR * 100 (preflop=-1 sentinel). Range-filter ±50%
                             at query time to bound stack depth.

Safe to re-run (no-op if collections already exist).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import DataType

from src._log import configure_logging, get_logger
from src.db.milvus import connect

log = get_logger("scripts.setup_milvus_collection")

# ── Collection definitions (FEATURES-v2.md §Architecture) ────────────────────
COLLECTIONS = {
    "preflop_decisions": 34,
    "postflop_decisions": 80,
}

# Hard-filter scalar fields shared by both collections (FEATURES-v2.md §Architecture)
_SCALAR_FIELDS = [
    ("street_class", DataType.VARCHAR, {"max_length": 16}),
    ("street", DataType.VARCHAR, {"max_length": 16}),
    ("pot_type", DataType.VARCHAR, {"max_length": 8}),
    ("hero_pos_rel", DataType.VARCHAR, {"max_length": 8}),
    ("n_players_active", DataType.INT64, {}),
    ("spr_x100", DataType.INT64, {}),
    # hero_action_type (VARCHAR): the DP-level hero action label populated from
    # hm_decisions.jsonl by tools/upsert_milvus.py. DecisionEngine.decide()
    # aggregates this across k neighbors to form a blended distribution (ENGN-03).
    # Phase 2's first build did NOT populate this field — Plan 03-04 includes a
    # PC --rebuild checkpoint that re-creates collections with this schema.
    ("hero_action_type", DataType.VARCHAR, {"max_length": 16}),
    # Raw postflop-snap inputs (pot-fraction, to/facing ratio with -1 sentinel,
    # all-in flag); blend snaps these to a canonical bet_*/raise_* bucket at serve.
    ("hero_action_size_pot_frac", DataType.FLOAT, {}),
    ("raise_ratio", DataType.FLOAT, {}),
    ("hero_action_allin", DataType.BOOL, {}),
    # active (BOOL): soft-delete flag. ENGN-02 requires kNN retrieval filtered to
    # `active == true`. DP collections store only live rows by default, but the
    # explicit field lets future phases (auto-loop patching) deactivate stale
    # neighbors without dropping rows. Default = True on insert.
    ("active", DataType.BOOL, {}),
    # confidence (FLOAT in [0,1]): per-DP confidence score. ENGN-03 blend weight
    # is `similarity * confidence * gto_score`. Phase 2 DPs do not populate
    # this; upsert_milvus.py applies default 1.0 when the parquet column is
    # absent (back-fillable on a future rebuild).
    ("confidence", DataType.FLOAT, {}),
    # gto_score (FLOAT in [0,1]): per-DP solver-quality score. ENGN-03 blend
    # weight component. Default 1.0 for Phase 2 HH-sourced DPs (unsolved);
    # Phase 4 labeling will populate real values on a subset of DPs.
    ("gto_score", DataType.FLOAT, {}),
    # feature_spec_version (INT64): schema version for kNN index rebuild coordination
    # (RESEARCH.md §Schema Version Coordination). Populated at upsert time by
    # tools/upsert_milvus.py using tools.build_embedding.FEATURE_SPEC_VERSION = 2.
    # Enables Plan 06 --rebuild first run to filter stale v1 vectors.
    ("feature_spec_version", DataType.INT64, {}),
    # action_dist (VARCHAR JSON): full hero mix for patch/solver rows. Base-corpus
    # rows store "" and the blend falls back to the single hero_action_type label.
    ("action_dist", DataType.VARCHAR, {"max_length": 512}),
    # added_at / removed_at (INT64 epoch seconds): corpus-version row stamps. As-of
    # pins filter added_at <= cutoff and (removed_at == 0 or removed_at > cutoff).
    ("added_at", DataType.INT64, {}),
    ("removed_at", DataType.INT64, {}),
]


def _build_schema(client, dim: int):
    """Build a collection schema with the given embedding dim + shared scalar fields."""
    schema = client.create_schema(enable_dynamic_field=False, auto_id=False)
    schema.add_field("decision_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    for field_name, dtype, kwargs in _SCALAR_FIELDS:
        schema.add_field(field_name, dtype, **kwargs)
    return schema


def _build_index_params(client):
    """HNSW COSINE index on the embedding field."""
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )
    return index_params


def setup(uri: str, token: str | None = None) -> dict[str, bool]:
    """Create both Milvus collections if they don't exist.

    Args:
        uri: Milvus server URI, e.g. ``"http://127.0.0.1:51530"``.
        token: optional auth token (root:<password> for Milvus built-in auth).

    Returns:
        Dict mapping collection name → True if created, False if already existed.
    """
    client = connect(uri, token=token)
    results: dict[str, bool] = {}

    for collection_name, dim in COLLECTIONS.items():
        if client.has_collection(collection_name):
            log.info(
                "milvus.collection.skip",
                collection=collection_name,
                reason="exists",
            )
            results[collection_name] = False
            continue

        schema = _build_schema(client, dim)
        index_params = _build_index_params(client)

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        client.load_collection(collection_name)
        log.info(
            "milvus.collection.created",
            collection=collection_name,
            dim=dim,
        )
        results[collection_name] = True

    return results


def _build_milvus_token(raw: str | None) -> str | None:
    """Construct the pymilvus auth token from the raw password env var.

    Milvus built-in auth expects ``user:password`` format. The .env stores only
    the raw password (MILVUS_TOKEN=<password>); we prepend ``root:`` here.
    If the value already contains ``:`` (e.g. ``root:mypass``) it is passed as-is.
    """
    if raw is None:
        return None
    return raw if ":" in raw else f"root:{raw}"


if __name__ == "__main__":
    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
    host = os.environ["MILVUS_HOST"]
    port = os.environ["MILVUS_PORT"]
    raw_token = os.environ.get("MILVUS_TOKEN")
    token = _build_milvus_token(raw_token)
    uri = f"http://{host}:{port}"
    created = setup(uri, token=token)
    for name, was_created in created.items():
        status = "created" if was_created else "already exists"
        print(f"Milvus collection {name!r}: {status}.")
