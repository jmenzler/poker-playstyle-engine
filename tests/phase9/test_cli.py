"""Phase 9 CLI eval-suite subcommand smoke tests."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch


def _make_stub_result(suite_pass: bool) -> MagicMock:
    r = MagicMock()
    r.suite_pass = suite_pass
    r.tier1_tvd = 0.05
    r.tier1_top1 = 0.80
    r.tier1_threshold_floor = 0.15
    r.tier1_regression_delta = 0.0
    r.tier1_pass = suite_pass
    r.tier2_exploitability = 2.5
    r.tier2_pass = True
    r.match_results = []
    r.match_pass = suite_pass
    return r


def test_eval_suite_cli_pass():
    """eval-suite exits 0 when suite_pass=True."""
    import msgspec

    from src.cli import eval_suite

    args = argparse.Namespace(hands=1, seed=42, format="json")
    stub = _make_stub_result(suite_pass=True)
    with (
        patch("src.eval.suite.run_suite", return_value=stub),
        patch.object(msgspec, "to_builtins", return_value={"suite_pass": True}),
    ):
        rc = eval_suite.run(args)
    assert rc == 0


def test_eval_suite_cli_fail():
    """eval-suite exits 1 when suite_pass=False."""
    import msgspec

    from src.cli import eval_suite

    args = argparse.Namespace(hands=1, seed=42, format="text")
    stub = _make_stub_result(suite_pass=False)
    with (
        patch("src.eval.suite.run_suite", return_value=stub),
        patch.object(
            msgspec,
            "to_builtins",
            return_value={
                "suite_pass": False,
                "tier1_tvd": 0.2,
                "tier1_threshold_floor": 0.15,
                "tier1_pass": False,
                "tier2_exploitability": 2.5,
                "match_pass": True,
            },
        ),
    ):
        rc = eval_suite.run(args)
    assert rc == 1


def test_eval_suite_cli_error():
    """eval-suite exits 2 on unexpected exception."""
    from src.cli import eval_suite

    args = argparse.Namespace(hands=1, seed=42, format="text")
    with patch("src.eval.suite.run_suite", side_effect=RuntimeError("boom")):
        rc = eval_suite.run(args)
    assert rc == 2


def test_eval_suite_help():
    """eval-suite --help exits 0 (argparse registered correctly)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "src.cli.main", "eval-suite", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "eval-suite" in result.stdout or "eval_suite" in result.stdout
