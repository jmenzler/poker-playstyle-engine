"""STOR-03: strategy_nodes has index on (cluster_key, active) and source CHECK constraint."""

import psycopg
import pytest

pytestmark = pytest.mark.integration


def test_cluster_active_index_exists(tsdb_dsn: str) -> None:
    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename='strategy_nodes' AND indexname='idx_strategy_nodes_cluster_active'"
        )
        row = cur.fetchone()
        assert row is not None, "idx_strategy_nodes_cluster_active missing"
        assert "cluster_key" in row[0] and "active" in row[0]


def test_source_check_constraint(tsdb_dsn: str) -> None:
    """Attempt to insert an invalid source value — must raise CheckViolation."""
    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.execute("BEGIN")
        try:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO strategy_nodes "
                    "(node_id, cluster_key, embedding, action_dist, gto_score, confidence, source) "
                    "VALUES (gen_random_uuid(), 'test', '{0.0}'::real[], '{}'::jsonb, "
                    "0.5, 0.5, 'INVALID_SOURCE')"
                )
        finally:
            cur.execute("ROLLBACK")
