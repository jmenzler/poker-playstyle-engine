"""End-to-end smoke: TimescaleDB row insert + Milvus vector insert + retrieve both.

AMENDMENT (2026-05-17 kNN pivot, FEATURES-v2.md): Tests dual-collection schema.
  - TSDB: inserts into strategy_nodes (source-of-truth store)
  - Milvus: inserts into both preflop_decisions (32-dim) and postflop_decisions (80-dim)

Validates Phase 1 success criterion #1: both stores writable + readable end-to-end.
Insert order: TimescaleDB FIRST, then Milvus (per PITFALLS.md §5).
"""

import uuid

import psycopg
import pytest

pytestmark = pytest.mark.integration


def test_timescaledb_round_trip(tsdb_dsn: str) -> None:
    """Insert a strategy_nodes row into TSDB and read it back."""
    node_id = str(uuid.uuid4())
    cluster_key = f"smoke-{node_id[:8]}"
    embedding = [0.01 * i for i in range(128)]

    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO strategy_nodes "
            "(node_id, cluster_key, embedding, action_dist, gto_score, confidence, source) "
            "VALUES (%s, %s, %s, '{}'::jsonb, 1.0, 1.0, 'seed')",
            (node_id, cluster_key, embedding),
        )
        conn.commit()

    try:
        with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT cluster_key FROM strategy_nodes WHERE node_id = %s",
                (node_id,),
            )
            row = cur.fetchone()
            assert row is not None, f"strategy_nodes row {node_id} not found"
            assert row[0] == cluster_key
    finally:
        with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM strategy_nodes WHERE node_id = %s", (node_id,))
            conn.commit()


@pytest.mark.parametrize(
    "collection_name,dim",
    [
        ("preflop_decisions", 34),
        ("postflop_decisions", 80),
    ],
)
def test_milvus_round_trip(
    milvus_uri: str,
    milvus_token: str | None,
    collection_name: str,
    dim: int,
) -> None:
    """Insert a decision vector into Milvus and retrieve it via filtered search."""
    from src.db.milvus import connect as mvconnect

    mv = mvconnect(milvus_uri, token=milvus_token)

    decision_id = f"smoke-{uuid.uuid4().hex[:12]}"
    embedding = [0.01 * i for i in range(dim)]
    doc = {
        "decision_id": decision_id,
        "embedding": embedding,
        "street_class": "preflop" if dim == 34 else "postflop",
        "street": "preflop" if dim == 34 else "flop",
        "pot_type": "srp",
        "hero_pos_rel": "IP",
        "n_players_active": 2,
        "spr_x100": 1000,
    }
    mv.insert(collection_name=collection_name, data=[doc])
    # Flush to ensure vector is indexed before searching.
    mv.flush(collection_name)

    try:
        results = mv.search(
            collection_name=collection_name,
            data=[embedding],
            limit=1,
            filter=f'decision_id == "{decision_id}"',
            output_fields=["decision_id"],
        )
        assert len(results[0]) == 1, f"Expected 1 result for {decision_id}, got {len(results[0])}"
        assert results[0][0]["entity"]["decision_id"] == decision_id
    finally:
        mv.delete(
            collection_name=collection_name,
            filter=f'decision_id == "{decision_id}"',
        )
