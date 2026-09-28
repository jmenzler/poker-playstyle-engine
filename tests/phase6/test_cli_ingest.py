"""tests/phase6/test_cli_ingest.py — Unit tests for `poker-engine ingest` CLI shim (CLI-01)."""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch


def _args(*, rebuild=False, format="text"):
    return argparse.Namespace(rebuild=rebuild, format=format)


def _extract_json_block(out: str):
    """Line-scan for the first ``{`` / ``[`` at column 0 (structlog log lines
    are timestamp-prefixed; the JSON payload starts on a fresh line)."""
    decoder = json.JSONDecoder()
    lines = out.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line and line[0] in "{[":
            obj, _end = decoder.raw_decode("".join(lines[i:]))
            return obj
    raise ValueError(f"no JSON payload found in stdout: {out!r}")


def _canned() -> dict:
    return {
        "hands_processed": 1024,
        "hands_skipped": {"straddle": 12, "run_it_twice": 3},
        "elapsed_s": 42.0,
        "watermark_before": "h-100",
        "watermark_after": "h-1124",
    }


def test_ingest_default_calls_incremental():
    """`ingest` (no flags) calls ingest_incremental."""
    with (
        patch("src.study.ingest.ingest_incremental", return_value=_canned()) as mock_inc,
        patch("src.study.ingest.ingest_rebuild") as mock_rebuild,
    ):
        from src.cli.ingest import run

        rc = run(_args(rebuild=False, format="json"))

    assert rc == 0
    mock_inc.assert_called_once()
    mock_rebuild.assert_not_called()


def test_ingest_with_rebuild_flag_calls_rebuild():
    """`ingest --rebuild` calls ingest_rebuild."""
    with (
        patch("src.study.ingest.ingest_incremental") as mock_inc,
        patch("src.study.ingest.ingest_rebuild", return_value=_canned()) as mock_rebuild,
    ):
        from src.cli.ingest import run

        rc = run(_args(rebuild=True, format="json"))

    assert rc == 0
    mock_inc.assert_not_called()
    mock_rebuild.assert_called_once()


def test_ingest_json_format_round_trips(capsys):
    """`ingest --format json` prints the full payload as JSON."""
    canned = _canned()
    with patch("src.study.ingest.ingest_incremental", return_value=canned):
        from src.cli.ingest import run

        rc = run(_args(format="json"))

    assert rc == 0
    payload = _extract_json_block(capsys.readouterr().out)
    assert payload["hands_processed"] == 1024


def test_ingest_text_format_shows_summary(capsys):
    """`ingest --format text` renders a multi-line summary."""
    with patch("src.study.ingest.ingest_incremental", return_value=_canned()):
        from src.cli.ingest import run

        rc = run(_args(format="text"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "1024" in out
    assert "ingest complete" in out.lower()


def test_ingest_unexpected_exception_returns_exit_2():
    """Generic backend exception → exit 2."""
    with patch("src.study.ingest.ingest_incremental", side_effect=RuntimeError("boom")):
        from src.cli.ingest import run

        rc = run(_args())
    assert rc == 2
