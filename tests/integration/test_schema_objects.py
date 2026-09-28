"""STOR-01: all six tables exist with required columns."""

import psycopg
import pytest

pytestmark = pytest.mark.integration

REQUIRED_TABLES = {
    "observations": [
        "obs_id",
        "cluster_key",
        "embedding",
        "action_taken",
        "ev_realized",
        "source",
        "session_id",
        "solver_label",
        "ts",
    ],
    "strategy_nodes": [
        "node_id",
        "cluster_key",
        "embedding",
        "action_dist",
        "gto_score",
        "confidence",
        "source",
        "active",
        "created_at",
    ],
    "weight_overlay": [
        "overlay_id",
        "cluster_key",
        "overlay_weights",
        "source",
        "active",
        "superseded_by",
        "created_at",
    ],
    "patches": [
        "patch_id",
        "ts",
        "source",
        "cluster_key",
        "prev_node_id",
        "new_node_id",
        "pre_ev_loss",
        "post_ev_loss",
        "status",
        "validation",
        "notes",
    ],
    "metrics": [
        "metric_id",
        "session_id",
        "cluster_key",
        "metric_name",
        "value",
        "ts",
    ],
    "villain_logs": [
        "log_id",
        "session_id",
        "cluster_key",
        "villain_id",
        "position",
        "action_taken",
        "bet_size_pct",
        "ts",
    ],
}


@pytest.mark.parametrize("table,cols", list(REQUIRED_TABLES.items()))
def test_table_exists_with_columns(tsdb_dsn: str, table: str, cols: list[str]) -> None:
    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s",
            (table,),
        )
        present = {r[0] for r in cur.fetchall()}
    missing = set(cols) - present
    assert not missing, f"{table} missing columns: {missing}"
