"""SIM-03 integration test: observation count = decision count (PC-gated).

Verifies that every decision point processed by the harness produces exactly
one observation row in TimescaleDB. Any mismatch indicates one of:

  (a) ObservationWriter dropped rows (batch-flush bug, connection drop, or
      tail rows not flushed on close)
  (b) Harness miscounted total_decisions (env.step succeeded but the counter
      was not incremented, or the safety cap truncated the inner loop without
      decrementing)
  (c) Cleanup logic in a prior test leaked rows tagged with a colliding
      session_id (uuid4 prefix collisions are astronomically unlikely; the
      hex[:8] prefix is 32 bits so 2^16 sessions before ~50% birthday risk)

The test uses a fresh per-run session_id (sim03-{uuid4.hex[:8]}) and an
explicit DELETE in the finally block to guarantee isolation. n_hands=10 is
small enough to keep the test under one second on PC while still exercising
the writer's batch-flush threshold (batch_size=50 here vs. ~24 decisions per
6-max hand → ~2-4 flushes per test run).
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

pytestmark = pytest.mark.integration


def test_observation_count_equals_decision_count(tsdb_dsn: str) -> None:
    """SIM-03: COUNT(observations) = SUM(decision_points) across all hands in a session."""
    from src.decision_engine.engine import engine_from_env
    from src.sim import ObservationWriter, SimAdapter, run_record_session

    session_id = f"sim03-{uuid.uuid4().hex[:8]}"
    try:
        engine = engine_from_env(rng_seed=7)
        adapter = SimAdapter()
        with ObservationWriter(tsdb_dsn, session_id, batch_size=50) as writer:
            summary = run_record_session(
                adapter,
                engine,
                writer,
                session_seed=7,
                n_hands=10,
                session_id=session_id,
            )

        with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM observations WHERE session_id = %s",
                (session_id,),
            )
            db_count = cur.fetchone()[0]

        assert db_count == summary["total_decisions"], (
            f"SIM-03 violation: db_count={db_count} != harness_total_decisions={summary['total_decisions']}"
        )
        assert db_count > 0, "session produced zero observations"
    finally:
        with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM observations WHERE session_id = %s", (session_id,))
            conn.commit()
