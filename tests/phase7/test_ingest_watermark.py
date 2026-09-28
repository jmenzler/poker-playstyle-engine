# rot-allow-file
"""Phase 7 / BLOCKER 1 / INTG-01: ingest_incremental watermark uses MAX(ts).

Closes BOOT-04..06 unit verification. Cf. .planning/v1.0-MILESTONE-AUDIT.md
BLOCKER 1 + .planning/phases/07-close-gap-boot-cli-watermark/07-CONTEXT.md D-07-2.
"""

from __future__ import annotations

import subprocess

from src.study.ingest import ingest_incremental


def _fake_runner(stdout: str = "hands_processed=0\n"):
    def runner(argv):
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    return runner


def _execute_calls(mock_tsdb_conn):
    """Return list of SQL strings passed to cur.execute via the mock conn."""
    cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    return [c.args[0] for c in cur.execute.call_args_list]


def test_uses_ts_max(mock_tsdb_conn):
    """BOOT-04 / BLOCKER 1: ingest_incremental issues SELECT MAX(ts), not MAX(hand_id)."""
    cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (None,)
    ingest_incremental(_tsdb_conn=mock_tsdb_conn, _runner=_fake_runner())
    sql_calls = _execute_calls(mock_tsdb_conn)
    assert any("MAX(ts)" in s for s in sql_calls), f"Expected MAX(ts) in SQL calls, got: {sql_calls}"
    assert not any("MAX(hand_id)" in s for s in sql_calls), "MAX(hand_id) must be eradicated"


def test_watermark_pair_returned(mock_tsdb_conn):
    """BOOT-05: ingest_incremental result envelope exposes watermark_before + watermark_after."""
    cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (None,)
    out = ingest_incremental(_tsdb_conn=mock_tsdb_conn, _runner=_fake_runner())
    assert "watermark_before" in out, f"missing watermark_before in envelope: {out}"
    assert "watermark_after" in out, f"missing watermark_after in envelope: {out}"


def test_skip_reasons_parsed(mock_tsdb_conn):
    """BOOT-06: straddle + run_it_twice skip counts surface in result envelope.

    Exercises `_parse_skipped_reasons` reachability post-fix — before the fix,
    `MAX(hand_id)` raised psycopg.errors.UndefinedColumn before any pipeline
    runner was invoked, so this code path was dead.
    """
    cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (None,)
    stdout = "hands_processed=10\nskipped_straddle=3\nskipped_run_it_twice=1\n"
    out = ingest_incremental(_tsdb_conn=mock_tsdb_conn, _runner=_fake_runner(stdout))
    assert out["hands_skipped"].get("straddle") == 3, (
        f"expected 3 straddle skips, got: {out['hands_skipped']}"
    )
    assert out["hands_skipped"].get("run_it_twice") == 1, (
        f"expected 1 run_it_twice skip, got: {out['hands_skipped']}"
    )
