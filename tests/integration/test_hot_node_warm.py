"""STOR-06: >= 1,000 synthetic strategy_nodes rows loadable from TimescaleDB into in-process list.

Proves the hot-node cache architecture can be populated from TSDB on startup.
Phase 2 replaces synthetic rows with real observation-derived data.

Test lifecycle:
1. Seed 1,000 synthetic rows (any valid cluster_key/embedding pattern)
2. Load them via psycopg connection into an in-process Python list
3. Assert len(loaded) == 1,000
4. Clean up synthetic rows in finally block
"""

import uuid

import psycopg
import pytest

pytestmark = pytest.mark.integration

# Use a unique cluster_key prefix to identify synthetic rows.
# Source must be a valid value per strategy_nodes_source_check constraint.
_SYNTHETIC_PREFIX = "stor06syn_"
_SYNTHETIC_SOURCE = "seed"  # valid source per CHECK constraint
_BATCH_SIZE = 1000


def _seed_synthetic_nodes(dsn: str) -> list[str]:
    """Insert 1,000 synthetic strategy_nodes rows. Returns list of inserted node_ids."""
    node_ids = [str(uuid.uuid4()) for _ in range(_BATCH_SIZE)]
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for i, nid in enumerate(node_ids):
            cluster_key = f"{_SYNTHETIC_PREFIX}{i:04d}"
            embedding = [0.0] * 128
            cur.execute(
                "INSERT INTO strategy_nodes "
                "(node_id, cluster_key, embedding, action_dist, gto_score, confidence, source) "
                "VALUES (%s, %s, %s, '{}'::jsonb, 0.5, 0.5, %s)",
                (nid, cluster_key, embedding, _SYNTHETIC_SOURCE),
            )
        conn.commit()
    return node_ids


def _load_nodes_into_list(dsn: str) -> list[dict]:
    """Load all synthetic test rows from TSDB into an in-process Python list."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT node_id, cluster_key, embedding FROM strategy_nodes WHERE cluster_key LIKE %s",
            (f"{_SYNTHETIC_PREFIX}%",),
        )
        rows = cur.fetchall()
    return [{"node_id": r[0], "cluster_key": r[1], "embedding": r[2]} for r in rows]


def _cleanup_synthetic_nodes(dsn: str) -> None:
    """Remove all synthetic test rows."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM strategy_nodes WHERE cluster_key LIKE %s",
            (f"{_SYNTHETIC_PREFIX}%",),
        )
        conn.commit()


def test_hot_node_population(tsdb_dsn: str) -> None:
    """STOR-06: 1,000 synthetic rows seed + load into in-process list."""
    try:
        # Step 1: seed
        node_ids = _seed_synthetic_nodes(tsdb_dsn)
        assert len(node_ids) == _BATCH_SIZE

        # Step 2: load from TSDB into in-process list
        loaded_nodes = _load_nodes_into_list(tsdb_dsn)

        # Step 3: verify count
        assert len(loaded_nodes) == _BATCH_SIZE, (
            f"Expected {_BATCH_SIZE} nodes in cache, got {len(loaded_nodes)}"
        )

        # Step 4: spot-check structure
        first = loaded_nodes[0]
        assert "node_id" in first
        assert "cluster_key" in first
        assert "embedding" in first
        assert len(first["embedding"]) == 128

    finally:
        # Step 5: cleanup
        _cleanup_synthetic_nodes(tsdb_dsn)
