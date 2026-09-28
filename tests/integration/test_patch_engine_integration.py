"""PatchEngine apply + rollback live round-trip against PC TimescaleDB + Milvus.

Run on PC: pytest -m integration tests/integration/test_patch_engine_integration.py -v
"""

from __future__ import annotations

import json
import uuid

import psycopg
import pytest

pytestmark = pytest.mark.integration


def test_apply_then_rollback_round_trip(
    tsdb_dsn: str,
    milvus_uri: str,
    milvus_token: str | None,
) -> None:
    """Live PatchEngine apply + rollback round-trip against PC TimescaleDB + Milvus."""
    from src.db import milvus as milvus_db
    from src.patch_engine import PatchEngine, PatchRecord, PatchSpec, ValidationResult

    unique_suffix = str(uuid.uuid4())[:8]
    cluster_key = (
        f"hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop|test_id={unique_suffix}"
    )
    decision_id = f"dp_test_{unique_suffix}"
    action_dist = {"fold": 0.1, "call": 0.2, "bet_50": 0.7}
    embedding = [0.0] * 80
    collection = "postflop_decisions"

    mclient = milvus_db.connect(milvus_uri, token=milvus_token)
    engine = PatchEngine()
    record: PatchRecord | None = None

    try:
        with psycopg.connect(tsdb_dsn) as tsdb_conn:
            spec = PatchSpec(
                cluster_key=cluster_key,
                decision_id=decision_id,
                action_dist=action_dist,
                embedding=embedding,
                gto_score=0.8,
                confidence=0.9,
                prev_node_id=None,
                source="autoloop",
            )
            validation = ValidationResult(
                seed=0,
                ev_loss_delta=0.1,
                n_hands=100,
                confidence_interval=(0.01, 0.2),
                n_cluster_hits=10,
            )

            # --- Apply ---
            record = engine.apply(
                spec,
                validation=validation,
                pre_ev_loss=0.5,
                post_ev_loss=0.4,
                _tsdb_conn=tsdb_conn,
                _milvus=mclient,
            )
            assert record.status == "applied"
            assert record.cluster_key == cluster_key
            assert record.prev_node_id is None

            with tsdb_conn.cursor() as cur:
                cur.execute(
                    "SELECT node_id, source, active FROM strategy_nodes "
                    "WHERE cluster_key = %s AND active = TRUE",
                    (cluster_key,),
                )
                sn_row = cur.fetchone()
            assert sn_row is not None, "apply must insert an active strategy_nodes row"
            db_node_id, db_source, db_active = sn_row
            assert str(db_node_id) == str(record.new_node_id)
            assert db_source == "autoloop"
            assert db_active is True

            with tsdb_conn.cursor() as cur:
                cur.execute("SELECT status, decision_id FROM patches WHERE patch_id = %s", (record.patch_id,))
                p_row = cur.fetchone()
            assert p_row is not None, "apply must insert a patches row"
            assert p_row[0] == "applied"
            assert p_row[1] == decision_id

            # Milvus row exists, keyed by decision_id, with the full action_dist
            m_rows = mclient.query(
                collection_name=collection,
                filter=f'decision_id == "{decision_id}"',
                output_fields=["decision_id", "action_dist"],
                consistency_level="Strong",
            )
            assert m_rows, "apply must upsert a Milvus row keyed by decision_id"
            assert json.loads(m_rows[0]["action_dist"]) == action_dist

            # --- Rollback ---
            rollback_record = engine.rollback(record.patch_id, _tsdb_conn=tsdb_conn, _milvus=mclient)
            assert rollback_record.status == "rolled_back"
            assert rollback_record.cluster_key == cluster_key

            with tsdb_conn.cursor() as cur:
                cur.execute("SELECT active FROM strategy_nodes WHERE node_id = %s", (record.new_node_id,))
                sn_after = cur.fetchone()
            assert sn_after is not None
            assert sn_after[0] is False, "strategy_nodes.active must be FALSE after rollback"

            # In-place status flip — no new patches row inserted
            with tsdb_conn.cursor() as cur:
                cur.execute("SELECT status FROM patches WHERE cluster_key = %s", (cluster_key,))
                patches_rows = cur.fetchall()
            assert len(patches_rows) == 1, f"Expected exactly 1 patches row, got {len(patches_rows)}"
            assert patches_rows[0][0] == "rolled_back"

            # Milvus row soft-deleted: row stays, active=False + removed_at>0
            m_after = mclient.query(
                collection_name=collection,
                filter=f'decision_id == "{decision_id}"',
                output_fields=["decision_id", "active", "removed_at"],
                consistency_level="Strong",
            )
            assert m_after, "rollback must keep the patch's Milvus row (soft-delete)"
            assert m_after[0]["active"] is False, "soft-delete must set active=False"
            assert m_after[0]["removed_at"] > 0, "soft-delete must stamp removed_at"

    finally:
        try:
            mclient.delete(collection_name=collection, ids=[decision_id])
        except Exception:
            pass
        with psycopg.connect(tsdb_dsn) as cleanup_conn, cleanup_conn.cursor() as cur:
            cur.execute("DELETE FROM patches WHERE cluster_key = %s", (cluster_key,))
            cur.execute("DELETE FROM strategy_nodes WHERE cluster_key = %s", (cluster_key,))
            cleanup_conn.commit()
