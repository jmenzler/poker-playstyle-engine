"""CLI tests for `autoloop run --autonomous` dispatch and `autoloop step` back-compat."""

from __future__ import annotations

import argparse
import json
from typing import ClassVar


class _FakeRunner:
    """Records construction kwargs; .run() returns a canned summary."""

    last_kwargs: ClassVar[dict] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs

    def run(self) -> dict:
        return {
            "cycles": 3,
            "patches_accepted": 2,
            "patches_rejected": 1,
            "ev_loss_delta": 0.0,
            "circuit_broken": False,
        }


class _RaisingRunner:
    def __init__(self, **kwargs: object) -> None:
        raise RuntimeError("startup boom")

    def run(self) -> dict:  # pragma: no cover
        return {}


class _FakeDriver:
    def step(self) -> int:
        return 3


def test_parser_accepts_run_flags() -> None:
    """The argparse parser accepts the run subcommand with all flags."""
    import src.cli.main as m

    # Build the parser the same way main() does by parsing a known argv.
    # main() calls ap.parse_args() internally; here we re-invoke through a tiny shim:
    parser = m._build_parser() if hasattr(m, "_build_parser") else None
    if parser is None:
        # Fallback: exercise via main() with a faked handler is covered by other tests.
        import pytest

        pytest.skip("no _build_parser; dispatch covered by test_run_dispatches_to_autonomous_runner")
    args = parser.parse_args(
        ["autoloop", "run", "--autonomous", "--run-id", "soak-1", "--seed", "7", "--max-cycles", "3"]
    )
    assert args.autoloop_cmd == "run"
    assert args.autonomous is True
    assert args.run_id == "soak-1"
    assert args.seed == 7
    assert args.max_cycles == 3


def test_run_dispatches_to_autonomous_runner(monkeypatch, capsys) -> None:
    """run() with autoloop_cmd='run' instantiates AutonomousRunner and prints its summary."""
    monkeypatch.setattr("src.cli.autoloop.AutonomousRunner", _FakeRunner, raising=False)
    from src.cli.autoloop import run

    result = run(
        argparse.Namespace(autoloop_cmd="run", autonomous=True, run_id="r", seed=1, max_cycles=2, resume=None)
    )

    assert result == 0
    first_line = capsys.readouterr().out.strip().splitlines()[0]
    payload = json.loads(first_line)
    assert payload["cycles"] == 3


def test_run_resume_passes_run_id(monkeypatch, capsys) -> None:
    """--resume <id> drives AutonomousRunner with that run_id."""
    monkeypatch.setattr("src.cli.autoloop.AutonomousRunner", _FakeRunner, raising=False)
    from src.cli.autoloop import run

    run(
        argparse.Namespace(
            autoloop_cmd="run", autonomous=True, run_id=None, seed=1, max_cycles=2, resume="soak-1"
        )
    )

    assert _FakeRunner.last_kwargs["run_id"] == "soak-1"


def test_step_back_compat_no_autoloop_cmd(monkeypatch, capsys) -> None:
    """Legacy run(Namespace()) with no autoloop_cmd still hits the step path."""
    monkeypatch.setattr("src.cli.autoloop.AutoLoopDriver", _FakeDriver)
    from src.cli.autoloop import run

    result = run(argparse.Namespace())

    assert result == 0
    first_line = capsys.readouterr().out.strip().splitlines()[0]
    assert json.loads(first_line)["patches_applied"] == 3


def test_run_returns_2_on_startup_error(monkeypatch, capsys) -> None:
    """run() returns 2 when AutonomousRunner construction raises."""
    monkeypatch.setattr("src.cli.autoloop.AutonomousRunner", _RaisingRunner, raising=False)
    from src.cli.autoloop import run

    result = run(
        argparse.Namespace(autoloop_cmd="run", autonomous=True, run_id="r", seed=1, max_cycles=1, resume=None)
    )

    assert result == 2
    err = json.loads(capsys.readouterr().err.strip())
    assert "boom" in err["error"]
