"""Phase 7 / 07-03 — cross-phase invariant: operator UI lock + suppression honored.

Closes BLOCKER 2 / INTG-02 (LOOP-01 cross-phase + PTCH-01 operator invariant).
Real TSDB insert → real `select_patch_candidates` → cluster excluded from result.

Two scenarios:
  - test_locked_cluster_skipped_by_next_autoloop_step:
        UI lock event (UPDATE strategy_nodes SET locked_from_autoloop=TRUE)
        → next select_patch_candidates DOES NOT return the locked cluster.
  - test_suppressed_cluster_skipped_by_next_autoloop_step:
        UI suppression event (INSERT INTO leak_suppressions ... active=TRUE)
        → next select_patch_candidates DOES NOT return the suppressed cluster.

PC-gated via `pytest -m integration` (requires TSDB env vars; tsdb_dsn fixture in
phase 7 conftest fails loud on missing TSDB_PASSWORD). Both tests insert direct
strategy_nodes rows with source='sim' (NOT via PatchEngine) because the slot
under test is the LOCK/SUPPRESSION gate, not the patch path. Cleanup in finally
removes all rows by cluster_key (mirrors test_no_direct_strategy_nodes_write.py
discipline).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from src._config import AutoLoopConfig
from src.autoloop.leak_detector import select_patch_candidates

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _seed_cluster(
    conn: psycopg.Connection,
    *,
    cluster_key: str,
    session_id: str,
    node_id: uuid.UUID,
    n_observations: int,
    ev_loss: float,
) -> None:
    """Seed the minimum rows for a cluster to clear the leak_detector pipeline:

    - 1 strategy_nodes row (active=TRUE, source='sim', locked=FALSE).
    - 1 metrics row with metric_name='ev_loss' value > tau_leak.
    - n_observations observations rows (clears min_observations gate).
    """
    embedding = [0.0] * 80  # 80-dim per Phase 1 lock; content irrelevant to this test
    now_ts = datetime.now(UTC)

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO strategy_nodes "
            "(node_id, cluster_key, embedding, action_dist, gto_score, confidence, "
            " source, active, locked_from_autoloop) "
            "VALUES (%s, %s, %s::real[], '{}'::jsonb, 0.5, 0.5, 'sim', TRUE, FALSE)",
            (str(node_id), cluster_key, embedding),
        )
        cur.execute(
            "INSERT INTO metrics (metric_id, session_id, cluster_key, metric_name, value, ts) "
            "VALUES (gen_random_uuid(), %s, %s, 'ev_loss', %s, %s)",
            (session_id, cluster_key, ev_loss, now_ts),
        )
        # Spread observation ts by a microsecond each so the hypertable PK
        # (obs_id, ts) is unique even with reused uuid generators across rows.
        for i in range(n_observations):
            obs_ts = now_ts + timedelta(microseconds=i)
            cur.execute(
                "INSERT INTO observations "
                "(obs_id, cluster_key, embedding, action_taken, source, session_id, ts) "
                "VALUES (gen_random_uuid(), %s, %s::real[], 'check', 'sim', %s, %s)",
                (cluster_key, embedding, session_id, obs_ts),
            )
    conn.commit()


def _cleanup_cluster(tsdb_dsn: str, cluster_key: str) -> None:
    """Delete all rows for the cluster across observations / metrics /
    leak_suppressions / strategy_nodes. Idempotent — safe to call on partial
    seed failures."""
    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM observations WHERE cluster_key = %s", (cluster_key,))
        cur.execute("DELETE FROM metrics WHERE cluster_key = %s", (cluster_key,))
        cur.execute("DELETE FROM leak_suppressions WHERE cluster_key = %s", (cluster_key,))
        cur.execute("DELETE FROM strategy_nodes WHERE cluster_key = %s", (cluster_key,))
        conn.commit()


def _run_leak_detector(session_id: str) -> list:
    """Call select_patch_candidates with a permissive config (real TSDB conn opened internally)."""
    cfg = AutoLoopConfig(
        enabled=True,
        tau_leak=0.1,
        min_observations=10,
        max_patches_per_run=10,
        n_hands_validation=10,
        bootstrap_resamples=10,
        ci_alpha=0.05,
    )
    return select_patch_candidates(session_id=session_id, config=cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_locked_cluster_skipped_by_next_autoloop_step(tsdb_dsn: str) -> None:
    """Operator UI lock (locked_from_autoloop=TRUE) → leak_detector skips cluster."""
    cluster_key = f"ck-phase7-lock-{uuid.uuid4().hex[:8]}"
    session_id = f"s-phase7-lock-{uuid.uuid4().hex[:8]}"
    node_id = uuid.uuid4()

    try:
        # 1. Arrange: seed strategy_node + metrics + observations for the cluster.
        with psycopg.connect(tsdb_dsn) as conn:
            _seed_cluster(
                conn,
                cluster_key=cluster_key,
                session_id=session_id,
                node_id=node_id,
                n_observations=25,
                ev_loss=0.5,
            )

        # 2. Sanity check: with no lock the cluster IS a candidate. This
        #    isolates the lock as the cause of skipping (rules out other gates).
        before = _run_leak_detector(session_id)
        assert any(c.cluster_key == cluster_key for c in before), (
            f"Pre-lock sanity: cluster {cluster_key} should be a candidate before locking"
        )

        # 3. UI lock event: operator clicks "lock from autoloop".
        with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE strategy_nodes SET locked_from_autoloop = TRUE WHERE cluster_key = %s",
                (cluster_key,),
            )
            conn.commit()

        # 4. Next autoloop step must NOT return the locked cluster.
        after = _run_leak_detector(session_id)
        assert all(c.cluster_key != cluster_key for c in after), (
            f"Locked cluster {cluster_key} was returned by leak_detector — operator lock broken"
        )

    finally:
        _cleanup_cluster(tsdb_dsn, cluster_key)


def test_suppressed_cluster_skipped_by_next_autoloop_step(tsdb_dsn: str) -> None:
    """Operator UI [accept as intentional] (active leak_suppressions row) → cluster skipped."""
    cluster_key = f"ck-phase7-supp-{uuid.uuid4().hex[:8]}"
    session_id = f"s-phase7-supp-{uuid.uuid4().hex[:8]}"
    node_id = uuid.uuid4()

    try:
        with psycopg.connect(tsdb_dsn) as conn:
            _seed_cluster(
                conn,
                cluster_key=cluster_key,
                session_id=session_id,
                node_id=node_id,
                n_observations=25,
                ev_loss=0.5,
            )

        before = _run_leak_detector(session_id)
        assert any(c.cluster_key == cluster_key for c in before), (
            f"Pre-suppression sanity: cluster {cluster_key} should be a candidate before suppression"
        )

        # UI [accept as intentional] event — append-only insert.
        with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO leak_suppressions (cluster_key, reason, active) "
                "VALUES (%s, 'integration-test', TRUE)",
                (cluster_key,),
            )
            conn.commit()

        after = _run_leak_detector(session_id)
        assert all(c.cluster_key != cluster_key for c in after), (
            f"Suppressed cluster {cluster_key} was returned by leak_detector — operator suppression broken"
        )

    finally:
        _cleanup_cluster(tsdb_dsn, cluster_key)
