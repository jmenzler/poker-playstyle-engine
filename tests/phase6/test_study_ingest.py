"""Unit tests for src/study/ingest.py (CLI-01 / D-10).

ingest_incremental reads watermark from observations, invokes phase2_pipeline,
returns counts + watermark deltas. ingest_rebuild invokes pipeline with --rebuild.

NOTE: tools/phase2_pipeline.py does NOT currently accept a --watermark flag, so
ingest_incremental does NOT pass one (relies on Milvus upsert idempotency by
observation_id PK). The watermark BEFORE/AFTER is reported for operator visibility.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.study.ingest import ingest_incremental, ingest_rebuild


def _mock_conn(max_hand_id_before=100, max_hand_id_after=200):
    """Mock psycopg conn returning two MAX(hand_id) fetchone() values in order."""
    conn = MagicMock()
    cm = MagicMock()
    cm.__enter__ = lambda s: cm
    cm.__exit__ = lambda *a: False
    cm.fetchone.side_effect = [(max_hand_id_before,), (max_hand_id_after,)]
    conn.cursor.return_value = cm
    return conn


def _mock_runner(stdout_text):
    """Build a MagicMock subprocess-style runner returning the given stdout."""
    result = MagicMock()
    result.stdout = stdout_text
    result.returncode = 0
    return MagicMock(return_value=result)


def test_ingest_incremental_returns_watermarks_and_counts():
    conn = _mock_conn(100, 200)
    runner = _mock_runner(
        '{"event":"ingest_complete","hands_processed":100,"skipped_straddle":3,"skipped_run_it_twice":1}'
    )
    out = ingest_incremental(_tsdb_conn=conn, _runner=runner)
    assert out["watermark_before"] == 100
    assert out["watermark_after"] == 200
    assert out["hands_processed"] == 100
    assert out["hands_skipped"] == {"straddle": 3, "run_it_twice": 1}
    assert "elapsed_s" in out


def test_ingest_incremental_invokes_phase2_pipeline():
    """ingest_incremental must invoke tools.phase2_pipeline --all."""
    conn = _mock_conn(0, 0)
    runner = _mock_runner('{"hands_processed":0}')
    ingest_incremental(_tsdb_conn=conn, _runner=runner)
    runner.assert_called_once()
    argv = runner.call_args[0][0]
    assert "tools.phase2_pipeline" in " ".join(argv)
    assert "--all" in argv
    # --watermark is intentionally NOT passed (phase2_pipeline does not accept it).
    assert "--watermark" not in argv


def test_ingest_rebuild_invokes_with_rebuild_flag():
    runner = _mock_runner('{"hands_processed":500}')
    out = ingest_rebuild(_tsdb_conn=MagicMock(), _runner=runner)
    assert out["rebuild"] is True
    assert out["hands_processed"] == 500
    argv = runner.call_args[0][0]
    assert "--rebuild" in argv


def test_ingest_incremental_handles_missing_counts_gracefully():
    """If pipeline stdout has no parseable counts, return zeros (no crash)."""
    conn = _mock_conn(50, 50)
    runner = _mock_runner("pipeline.complete")
    out = ingest_incremental(_tsdb_conn=conn, _runner=runner)
    assert out["hands_processed"] == 0
    assert out["hands_skipped"] == {}
    assert out["watermark_before"] == 50
    assert out["watermark_after"] == 50
