"""tests/phase6/test_cli_leaks.py — Unit tests for `poker-engine leaks` CLI shim (CLI-04).

Mocks ``src.study.leaks.rank_leaks`` so the shim contract is tested in isolation.
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch


def _args(*, format="text", type="both", min_n=20, limit=20):
    return argparse.Namespace(format=format, type=type, min_n=min_n, limit=limit)


def _canned_both() -> dict:
    return {
        "coverage": [
            {
                "obs_id": "obs-1",
                "cluster_key": "street_class=postflop|pot_type=srp|board_texture=dry",
                "max_neighbor_distance": 0.42,
                "session_id": "sess-1",
            }
        ],
        "strategy": [
            {
                "cluster_key": "street_class=postflop|pot_type=srp|n_players_active=2",
                "ev_loss": 0.5,
                "ci_low": 0.4,
                "ci_high": 0.6,
                "n_obs": 100,
                "source": "autoloop",
                "stage": "A",
                "shrunk": False,
                "score": 2.3,
            }
        ],
        "bucket_stats": {},
    }


def _extract_json_block(out: str):
    """Find the JSON payload in stdout. structlog log lines (timestamp-prefixed)
    do not start with ``{`` / ``[`` at column 0, so we line-scan for them."""
    decoder = json.JSONDecoder()
    lines = out.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line and line[0] in "{[":
            obj, _end = decoder.raw_decode("".join(lines[i:]))
            return obj
    raise ValueError(f"no JSON payload found in stdout: {out!r}")


def test_leaks_json_format_returns_valid_json(capsys):
    """`leaks --format json` produces parseable JSON with strategy + coverage keys."""
    canned = _canned_both()
    with patch("src.study.leaks.rank_leaks", return_value=canned):
        from src.cli.leaks import run

        rc = run(_args(format="json", type="both"))

    assert rc == 0
    payload = _extract_json_block(capsys.readouterr().out)
    assert "strategy" in payload
    assert payload["strategy"][0]["cluster_key"] == canned["strategy"][0]["cluster_key"]


def test_leaks_text_format_renders_both_sections(capsys):
    """`leaks --type both --format text` renders ## COVERAGE GAPS and ## STRATEGY LEAKS sections."""
    canned = _canned_both()
    with patch("src.study.leaks.rank_leaks", return_value=canned):
        from src.cli.leaks import run

        rc = run(_args(format="text", type="both"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "## COVERAGE GAPS" in out
    assert "## STRATEGY LEAKS" in out


def test_leaks_text_format_shows_shrunk_suffix(capsys):
    """Rows with shrunk=True render a `, shrunk` annotation."""
    canned = {
        "strategy": [
            {
                "cluster_key": "ck1",
                "ev_loss": 0.5,
                "ci_low": 0.4,
                "ci_high": 0.6,
                "n_obs": 8,
                "source": "autoloop",
                "stage": "A",
                "shrunk": True,
                "score": 1.0,
            }
        ],
        "coverage": [],
        "bucket_stats": {},
    }
    with patch("src.study.leaks.rank_leaks", return_value=canned):
        from src.cli.leaks import run

        rc = run(_args(format="text", type="strategy"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "shrunk" in out


def test_leaks_text_handles_empty_result(capsys):
    """Empty result renders '(no leaks)' rather than crashing."""
    with patch(
        "src.study.leaks.rank_leaks", return_value={"coverage": [], "strategy": [], "bucket_stats": {}}
    ):
        from src.cli.leaks import run

        rc = run(_args(format="text", type="both"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "(no leaks)" in out


def test_leaks_exception_returns_exit_2(capsys):
    """Unexpected exception → exit code 2 + JSON error on stderr."""
    with patch("src.study.leaks.rank_leaks", side_effect=RuntimeError("boom")):
        from src.cli.leaks import run

        rc = run(_args())

    assert rc == 2
    err = capsys.readouterr().err
    assert "error" in err.lower()
