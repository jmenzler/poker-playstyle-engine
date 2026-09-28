"""tests/phase6/test_cli_ab.py — Unit tests for `poker-engine ab` CLI shim (CLI-05)."""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch


def _args(cluster_key="ck1", patch_id="p1", *, format="text", n_hands=5000, seed=42):
    return argparse.Namespace(
        cluster_key=cluster_key,
        patch_id=patch_id,
        format=format,
        n_hands=n_hands,
        seed=seed,
    )


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
        "cluster_key": "ck1",
        "patch_id": "p1",
        "ev_loss_delta": -0.025,
        "ci_low": -0.04,
        "ci_high": -0.01,
        "n_hands": 5000,
        "seed": 42,
        "n_cluster_hits": 1234,
        "zero_hits": False,
    }


def test_ab_json_format_returns_delta_and_ci(capsys):
    """`ab --format json` returns dict with ev_loss_delta + ci_low + ci_high."""
    with patch("src.study.ab.run_ab", return_value=_canned()):
        from src.cli.ab import run

        rc = run(_args(format="json"))

    assert rc == 0
    payload = _extract_json_block(capsys.readouterr().out)
    assert payload["ev_loss_delta"] == -0.025
    assert payload["ci_low"] == -0.04
    assert payload["ci_high"] == -0.01


def test_ab_text_format_includes_key_fields(capsys):
    """Text format includes ev_loss_delta and a CI range."""
    with patch("src.study.ab.run_ab", return_value=_canned()):
        from src.cli.ab import run

        rc = run(_args(format="text"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "ev_loss_delta" in out
    assert "ci" in out.lower() or "CI" in out
    assert "5000" in out  # n_hands echoed


def test_ab_value_error_returns_exit_1(capsys):
    """ValueError (e.g. cluster_key mismatch) → exit 1 with JSON error on stderr."""
    with patch("src.study.ab.run_ab", side_effect=ValueError("cluster_key mismatch")):
        from src.cli.ab import run

        rc = run(_args())

    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()


def test_ab_unexpected_exception_returns_exit_2(capsys):
    """Generic exception → exit 2."""
    with patch("src.study.ab.run_ab", side_effect=RuntimeError("boom")):
        from src.cli.ab import run

        rc = run(_args())

    assert rc == 2
