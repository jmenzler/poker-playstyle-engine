"""tests/phase6/test_cli_patches.py — Unit tests for `poker-engine patches` CLI shim (CLI-07).

Two-level subparser:
  patches list   — wraps src.study.patches.list_patches
  patches rollback <patch_id> — delegates to src.cli.rollback.run (Phase 5)
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch


def _list_args(*, limit=20, cluster_key=None, source=None, format="text"):
    return argparse.Namespace(
        patches_command="list",
        limit=limit,
        cluster_key=cluster_key,
        source=source,
        format=format,
    )


def _rollback_args(patch_id="abc-uuid"):
    return argparse.Namespace(patches_command="rollback", patch_id=patch_id)


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


def _canned_list() -> list[dict]:
    return [
        {
            "patch_id": "p1",
            "ts": "2026-05-19T10:00:00Z",
            "cluster_key": "ck1",
            "source": "autoloop",
            "pre_ev_loss": 0.5,
            "post_ev_loss": 0.3,
            "status": "active",
            "prev_patch_id": None,
        },
    ]


def test_patches_list_calls_list_patches_with_filters():
    """`patches list --limit 5 --cluster-key ck1` calls list_patches with those filters."""
    captured = {}

    def _capture(**kw):
        captured.update(kw)
        return _canned_list()

    with patch("src.study.patches.list_patches", side_effect=_capture):
        from src.cli.patches import run

        rc = run(_list_args(limit=5, cluster_key="ck1", source="autoloop", format="json"))

    assert rc == 0
    assert captured.get("limit") == 5
    assert captured.get("cluster_key") == "ck1"
    assert captured.get("source") == "autoloop"


def test_patches_list_text_format_renders_rows(capsys):
    """Text format renders a table with column headers + row data."""
    with patch("src.study.patches.list_patches", return_value=_canned_list()):
        from src.cli.patches import run

        rc = run(_list_args(format="text"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "patch_id" in out
    assert "p1" in out


def test_patches_list_empty_text_format(capsys):
    """Empty result renders '(no patches)'."""
    with patch("src.study.patches.list_patches", return_value=[]):
        from src.cli.patches import run

        rc = run(_list_args(format="text"))

    assert rc == 0
    assert "no patches" in capsys.readouterr().out.lower()


def test_patches_rollback_delegates_to_rollback_module():
    """`patches rollback <patch_id>` delegates to src.cli.rollback.run."""
    with patch("src.cli.rollback.run", return_value=0) as mock_rollback:
        from src.cli.patches import run

        rc = run(_rollback_args(patch_id="abc-uuid"))

    assert rc == 0
    mock_rollback.assert_called_once()
    # Verify args propagation: the first positional argument should be the Namespace
    called_args = mock_rollback.call_args
    assert called_args.args[0].patch_id == "abc-uuid"


def test_patches_list_unexpected_exception_returns_exit_2():
    """Generic backend exception → exit 2."""
    with patch("src.study.patches.list_patches", side_effect=RuntimeError("boom")):
        from src.cli.patches import run

        rc = run(_list_args())
    assert rc == 2


def test_patches_list_json_format_returns_array(capsys):
    """`patches list --format json` prints a JSON array."""
    canned = _canned_list()
    with patch("src.study.patches.list_patches", return_value=canned):
        from src.cli.patches import run

        rc = run(_list_args(format="json"))

    assert rc == 0
    payload = _extract_json_block(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert payload[0]["patch_id"] == "p1"
