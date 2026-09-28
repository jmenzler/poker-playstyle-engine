"""tests/phase5/test_cli_autoloop.py — Unit tests for `poker-engine autoloop step` CLI shim."""

from __future__ import annotations

import argparse
import json


class _FakeDriver:
    """Fake AutoLoopDriver that returns 3 patches applied."""

    def step(self) -> int:
        return 3


class _ErrorDriver:
    """Fake AutoLoopDriver that raises RuntimeError."""

    def step(self) -> int:
        raise RuntimeError("boom — driver exploded")


def test_run_returns_zero_on_success(monkeypatch, capsys):
    """CLI run() returns 0 on success; stdout JSON contains patches_applied: 3."""
    monkeypatch.setattr("src.cli.autoloop.AutoLoopDriver", _FakeDriver)

    from src.cli.autoloop import run

    result = run(argparse.Namespace())

    assert result == 0
    captured = capsys.readouterr()
    # JSON is the first line; structlog may write additional lines to stdout
    first_line = captured.out.strip().splitlines()[0]
    payload = json.loads(first_line)
    assert payload["patches_applied"] == 3


def test_run_returns_2_on_driver_error(monkeypatch, capsys):
    """CLI run() returns 2 on driver exception; stderr JSON contains error message."""
    monkeypatch.setattr("src.cli.autoloop.AutoLoopDriver", _ErrorDriver)

    from src.cli.autoloop import run

    result = run(argparse.Namespace())

    assert result == 2
    captured = capsys.readouterr()
    err_payload = json.loads(captured.err.strip())
    assert "boom" in err_payload["error"]
