"""tests/phase6/test_cli_dashboard.py — Unit tests for `poker-engine dashboard` CLI shim (CLI-06)."""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch


def _args(*, format="text"):
    return argparse.Namespace(format=format)


def _extract_json_block(out: str):
    """Line-scan for the first ``{`` / ``[`` at column 0 (structlog prefixes
    log lines with a timestamp; the JSON payload starts on a fresh line)."""
    decoder = json.JSONDecoder()
    lines = out.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line and line[0] in "{[":
            obj, _end = decoder.raw_decode("".join(lines[i:]))
            return obj
    raise ValueError(f"no JSON payload found in stdout: {out!r}")


def _canned() -> dict:
    return {
        "loop_health": {
            "verdict": "improving",
            "narrative": "ev_loss trending down",
            "recent_mean": 0.12,
            "prior_mean": 0.18,
        },
        "ev_loss_trend": [
            {"session_id": "s1", "ts": "2026-05-19T10:00:00Z", "ev_loss": 0.42},
            {"session_id": "s2", "ts": "2026-05-19T11:00:00Z", "ev_loss": 0.31},
            {"session_id": "s3", "ts": "2026-05-19T12:00:00Z", "ev_loss": 0.18},
        ],
        "recent_activity": [
            {"ts": "2026-05-19T14:22:00Z", "event_type": "patch", "summary": "applied Δ-0.18"},
        ],
        "health": {"db": "ok", "milvus": "unknown", "solver": "unknown", "fastapi": "unknown"},
        "session": {
            "session_id": "sess-active",
            "n_obs": 1402,
            "start_ts": "2026-05-19T09:00:00Z",
            "last_ts": "2026-05-19T14:22:00Z",
        },
        "kb_growth": {
            "total_nodes": 12345,
            "by_source": {"autoloop": 10000, "solver": 1000, "manual": 1345},
            "last_7d_delta": {"autoloop": 250, "solver": 12, "manual": 4},
        },
    }


def test_dashboard_text_renders_all_sections(capsys):
    """All 6 D-NEW-27 sections appear in the rendered text output."""
    with patch("src.study.dashboard.dashboard_snapshot", return_value=_canned()):
        from src.cli.dashboard import run

        rc = run(_args(format="text"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "## LOOP HEALTH" in out
    assert "## EV_LOSS TREND" in out
    assert "## RECENT ACTIVITY" in out
    assert "## SUBSYSTEM HEALTH" in out
    assert "## SESSION" in out
    assert "## KB GROWTH" in out


def test_dashboard_text_includes_sparkline_chars(capsys):
    """ev_loss trend section includes ASCII sparkline (using sparkline glyphs)."""
    with patch("src.study.dashboard.dashboard_snapshot", return_value=_canned()):
        from src.cli.dashboard import run

        rc = run(_args(format="text"))

    assert rc == 0
    out = capsys.readouterr().out
    # sparkline emits Unicode block chars from this set
    sparkline_chars = set("▁▂▃▄▅▆▇█")
    assert any(c in out for c in sparkline_chars), f"no sparkline char found in: {out!r}"


def test_dashboard_json_format_returns_full_dict(capsys):
    """`dashboard --format json` round-trips the full dict."""
    canned = _canned()
    with patch("src.study.dashboard.dashboard_snapshot", return_value=canned):
        from src.cli.dashboard import run

        rc = run(_args(format="json"))

    assert rc == 0
    payload = _extract_json_block(capsys.readouterr().out)
    assert payload["loop_health"]["verdict"] == "improving"
    assert payload["kb_growth"]["total_nodes"] == 12345


def test_dashboard_exception_returns_exit_2(capsys):
    """Unexpected exception → exit 2 + JSON error on stderr."""
    with patch("src.study.dashboard.dashboard_snapshot", side_effect=RuntimeError("boom")):
        from src.cli.dashboard import run

        rc = run(_args())

    assert rc == 2
    err = capsys.readouterr().err
    assert "error" in err.lower()
