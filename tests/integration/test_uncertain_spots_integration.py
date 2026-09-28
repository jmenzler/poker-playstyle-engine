"""PC-only integration test for the uncertain-spots CLI subcommand.

Inserts 4 flagged + 1 unflagged observations with varying max_neighbor_distance
values directly via psycopg, runs main() in-process with sys.argv patched, and
asserts correct filtering and NULLS LAST ordering.

Requires live TimescaleDB with migration 008 applied (flagged_sparse +
max_neighbor_distance columns).  Skip is automatic when TSDB_PASSWORD is absent.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

pytestmark = pytest.mark.integration


def _insert_obs(cur, obs_id, session_id, flagged_sparse, max_neighbor_distance):
    """Insert a minimal observations row for test purposes."""
    cur.execute(
        """
        INSERT INTO observations
            (obs_id, cluster_key, embedding, action_taken, source, session_id,
             flagged_sparse, max_neighbor_distance, ts)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """,
        (
            obs_id,
            "street_class=postflop|pot_type=srp",
            [0.0] * 80,  # dummy 80-dim postflop embedding
            "call",
            "test",
            session_id,
            flagged_sparse,
            max_neighbor_distance,
        ),
    )


@pytest.mark.integration
def test_uncertain_spots_end_to_end(tsdb_dsn: str, capsys: pytest.CaptureFixture) -> None:
    """Insert 4 flagged + 1 unflagged row, run CLI, assert ordering and filtering.

    Rows:
      obs-f1: flagged=TRUE,  distance=0.9  (should appear 1st)
      obs-f2: flagged=TRUE,  distance=0.7  (should appear 2nd)
      obs-f3: flagged=TRUE,  distance=0.5  (should appear 3rd)
      obs-f4: flagged=TRUE,  distance=NULL (NULLS LAST — should appear 4th)
      obs-u1: flagged=FALSE, distance=0.8  (excluded by WHERE flagged_sparse=TRUE)
    """
    import psycopg

    session_id = f"test-uncertain-{uuid.uuid4().hex[:8]}"

    try:
        with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
            _insert_obs(cur, f"{session_id}-f1", session_id, True, 0.9)
            _insert_obs(cur, f"{session_id}-f2", session_id, True, 0.7)
            _insert_obs(cur, f"{session_id}-f3", session_id, True, 0.5)
            _insert_obs(cur, f"{session_id}-f4", session_id, True, None)
            _insert_obs(cur, f"{session_id}-u1", session_id, False, 0.8)
            conn.commit()

        # Run main() in-process with sys.argv patched — no subprocess
        from src.cli.main import main

        original_argv = sys.argv[:]
        sys.argv = ["poker-engine", "uncertain-spots", "--session", session_id]
        try:
            exit_code = main()
        finally:
            sys.argv = original_argv

        assert exit_code == 0, f"main() returned non-zero: {exit_code}"

        captured = capsys.readouterr()
        rows = json.loads(captured.out)

        # Exactly 4 flagged rows (unflagged obs-u1 excluded)
        assert len(rows) == 4, f"Expected 4 rows, got {len(rows)}: {rows}"

        # Rows ordered by max_neighbor_distance DESC NULLS LAST
        distances = [r["max_neighbor_distance"] for r in rows]
        assert distances[0] == pytest.approx(0.9), f"First row distance should be 0.9, got {distances[0]}"
        assert distances[1] == pytest.approx(0.7), f"Second row distance should be 0.7, got {distances[1]}"
        assert distances[2] == pytest.approx(0.5), f"Third row distance should be 0.5, got {distances[2]}"
        assert distances[3] is None, f"Last row (NULLS LAST) should be null, got {distances[3]}"

        # Unflagged row is absent
        obs_ids = {r["obs_id"] for r in rows}
        assert f"{session_id}-u1" not in obs_ids, "Unflagged row must not appear in output"

    finally:
        # Always clean up test rows
        with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM observations WHERE session_id = %s", (session_id,))
            conn.commit()
