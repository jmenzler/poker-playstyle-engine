"""tests/phase6/test_cli_eval.py — Unit tests for `poker-engine eval` CLI shim (D-NEW-30)."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from unittest.mock import patch


def _args(*, opponent="random", hands=100, seed=42, no_persist=True, format="text"):
    return argparse.Namespace(
        opponent=opponent,
        hands=hands,
        seed=seed,
        no_persist=no_persist,
        format=format,
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


def _canned_match_result():
    """Build a real MatchResult struct so msgspec.to_builtins works."""
    from src.eval.result import MatchResult

    return MatchResult(
        match_id="match-abc",
        opponent="random",
        hands=100,
        seed=42,
        bb_per_100=4.2,
        ci_low=1.0,
        ci_high=7.4,
        per_street={"preflop": 1.0, "flop": 2.0, "turn": 0.5, "river": 0.7},
        per_texture={"dry_rainbow": 1.5, "wet_two_tone": 0.8, "paired": 1.2, "monotone": 0.7},
        top5_profitable=[],
        top5_leaky=[],
        status="won",
        engine_version="0.1.0",
        started_at=datetime(2026, 5, 19, 10, tzinfo=UTC),
        finished_at=datetime(2026, 5, 19, 11, tzinfo=UTC),
    )


def test_eval_calls_run_match_with_args():
    """`run` forwards opponent / hands / seed / persist correctly."""
    captured = {}

    def _capture(opponent_name, **kw):
        captured["opponent_name"] = opponent_name
        captured.update(kw)
        return _canned_match_result()

    with patch("src.eval.run_match.run_match", side_effect=_capture):
        from src.cli.eval import run

        rc = run(_args(opponent="random", hands=100, seed=42, no_persist=True, format="json"))

    assert rc == 0
    assert captured.get("opponent_name") == "random"
    assert captured.get("hands") == 100
    assert captured.get("seed") == 42
    assert captured.get("persist") is False  # no_persist=True → persist=False


def test_eval_no_persist_false_sets_persist_true():
    """When --no-persist is NOT set (no_persist=False), persist=True."""
    captured = {}

    def _capture(opponent_name, **kw):
        captured.update(kw)
        return _canned_match_result()

    with patch("src.eval.run_match.run_match", side_effect=_capture):
        from src.cli.eval import run

        rc = run(_args(no_persist=False, format="json"))

    assert rc == 0
    assert captured.get("persist") is True


def test_eval_text_format_renders_match(capsys):
    """Text format prints match_id, bb/100, status."""
    with patch("src.eval.run_match.run_match", return_value=_canned_match_result()):
        from src.cli.eval import run

        rc = run(_args(format="text"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "match-abc" in out
    assert "won" in out.lower()
    assert "4.2" in out  # bb_per_100


def test_eval_json_format_returns_dict(capsys):
    """JSON format round-trips MatchResult fields."""
    with patch("src.eval.run_match.run_match", return_value=_canned_match_result()):
        from src.cli.eval import run

        rc = run(_args(format="json"))

    assert rc == 0
    payload = _extract_json_block(capsys.readouterr().out)
    assert payload["match_id"] == "match-abc"
    assert payload["status"] == "won"
    assert payload["bb_per_100"] == 4.2


def test_eval_key_error_returns_exit_1(capsys):
    """Unknown opponent (KeyError from run_match) → exit 1."""
    with patch("src.eval.run_match.run_match", side_effect=KeyError("unknown opponent 'foo'")):
        from src.cli.eval import run

        rc = run(_args(opponent="foo"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()


def test_eval_unexpected_exception_returns_exit_2():
    """Other exceptions → exit 2."""
    with patch("src.eval.run_match.run_match", side_effect=RuntimeError("boom")):
        from src.cli.eval import run

        rc = run(_args())
    assert rc == 2
