"""tests/phase5/test_cli_rollback.py — Unit tests for `poker-engine rollback <patch_id>` CLI shim."""

from __future__ import annotations

import argparse
import json
import uuid
from unittest.mock import MagicMock

from src.patch_engine import PatchRecord

_PATCH_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
_NEW_NODE_ID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


def _make_record(status: str = "rolled_back") -> PatchRecord:
    return PatchRecord(
        patch_id=_PATCH_ID,
        ts="2026-01-01T00:00:00+00:00",
        status=status,
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        new_node_id=_NEW_NODE_ID,
        prev_node_id=None,
    )


def test_run_returns_zero_on_success(monkeypatch, capsys):
    """run() returns 0 on success; stdout JSON contains status and patch_id."""
    mock_engine = MagicMock()
    mock_engine.rollback.return_value = _make_record("rolled_back")
    monkeypatch.setattr("src.cli.rollback.PatchEngine", lambda: mock_engine)

    from src.cli.rollback import run

    args = argparse.Namespace(patch_id=str(_PATCH_ID))
    result = run(args)

    assert result == 0
    captured = capsys.readouterr()
    # JSON is the first line; structlog may write additional lines to stdout
    first_line = captured.out.strip().splitlines()[0]
    payload = json.loads(first_line)
    assert payload["status"] == "rolled_back"
    assert payload["patch_id"] == str(_PATCH_ID)


def test_run_returns_1_on_invalid_uuid(monkeypatch, capsys):
    """run() returns 1 when args.patch_id is not a valid UUID."""
    from src.cli.rollback import run

    args = argparse.Namespace(patch_id="not-a-uuid")
    result = run(args)

    assert result == 1
    captured = capsys.readouterr()
    err_payload = json.loads(captured.err.strip())
    assert "invalid patch_id" in err_payload["error"]


def test_run_returns_1_on_lookup_error(monkeypatch, capsys):
    """run() returns 1 when rollback raises LookupError (patch not found)."""
    mock_engine = MagicMock()
    mock_engine.rollback.side_effect = LookupError("patch not found")
    monkeypatch.setattr("src.cli.rollback.PatchEngine", lambda: mock_engine)

    from src.cli.rollback import run

    args = argparse.Namespace(patch_id=str(_PATCH_ID))
    result = run(args)

    assert result == 1
    captured = capsys.readouterr()
    err_payload = json.loads(captured.err.strip())
    assert "patch not found" in err_payload["error"]


def test_run_returns_1_on_already_rolled_back(monkeypatch, capsys):
    """run() returns 1 when rollback raises RuntimeError (already rolled back)."""
    mock_engine = MagicMock()
    mock_engine.rollback.side_effect = RuntimeError("already rolled back")
    monkeypatch.setattr("src.cli.rollback.PatchEngine", lambda: mock_engine)

    from src.cli.rollback import run

    args = argparse.Namespace(patch_id=str(_PATCH_ID))
    result = run(args)

    assert result == 1
    captured = capsys.readouterr()
    err_payload = json.loads(captured.err.strip())
    assert "already rolled back" in err_payload["error"]
