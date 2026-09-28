"""tests/phase6/test_cli_edit_node.py — Unit tests for `poker-engine edit-node` CLI shim (CLI-03).

Tests cover the D-11 acceptance gates:
- ``--set 'fold:0.3,call:0.5,bet_50:0.2'`` parses to {fold:0.3, call:0.5, bet_50:0.2}
- Bad ``--set`` format raises ValueError → exit 1
- Editor mode is default when neither --set nor --editor is given
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import pytest


def _args(
    cluster_key="ck1",
    *,
    set_=None,
    editor=False,
    reason=None,
    format="text",
):
    """Build a Namespace matching the argparse layout in src/cli/main.py."""
    return argparse.Namespace(
        cluster_key=cluster_key,
        set=set_,  # the --set flag is stored as args.set
        editor=editor,
        reason=reason,
        format=format,
    )


def test_parse_set_string_returns_dict():
    """`_parse_set_string('fold:0.3,call:0.5,bet_50:0.2')` parses to the expected dict."""
    from src.cli.edit_node import _parse_set_string

    out = _parse_set_string("fold:0.3,call:0.5,bet_50:0.2")
    assert out == {"fold": 0.3, "call": 0.5, "bet_50": 0.2}


def test_parse_set_string_bad_format_raises():
    """A token without ':' is rejected with ValueError."""
    from src.cli.edit_node import _parse_set_string

    with pytest.raises(ValueError):
        _parse_set_string("fold0.3,call:0.5")


def test_run_with_set_calls_edit_node_with_parsed_dict():
    """`run` parses --set and passes it as action_dist."""
    captured = {}

    def _capture(**kw):
        captured.update(kw)
        return {"patch_id": "abc", "new_node_id": "n1", "source": "manual", "reason": ""}

    with patch("src.study.edit_node.edit_node", side_effect=_capture):
        from src.cli.edit_node import run

        rc = run(_args(set_="fold:0.5,call:0.5", format="json"))

    assert rc == 0
    assert captured.get("action_dist") == {"fold": 0.5, "call": 0.5}
    assert captured.get("cluster_key") == "ck1"


def test_run_with_invalid_set_returns_exit_1(capsys):
    """A bad --set string yields exit code 1 with JSON error on stderr."""
    from src.cli.edit_node import run

    rc = run(_args(set_="fold0.5,call:0.5"))
    assert rc == 1
    err = capsys.readouterr().err
    payload = json.loads(err.strip().splitlines()[-1])
    assert "error" in payload


def test_run_with_editor_flag_calls_editor_loop():
    """`--editor` triggers _editor_loop (mocked); _parse_set_string is NOT called."""
    with (
        patch("src.cli.edit_node._editor_loop", return_value={"check": 1.0}) as mock_editor,
        patch(
            "src.study.edit_node.edit_node",
            return_value={"patch_id": "abc", "new_node_id": "n1", "source": "manual", "reason": ""},
        ),
    ):
        from src.cli.edit_node import run

        rc = run(_args(editor=True, format="json"))

    assert rc == 0
    mock_editor.assert_called_once()


def test_run_defaults_to_editor_when_neither_flag():
    """When both --set and --editor are absent, defaults to editor mode."""
    with (
        patch("src.cli.edit_node._editor_loop", return_value={"check": 1.0}) as mock_editor,
        patch(
            "src.study.edit_node.edit_node",
            return_value={"patch_id": "abc", "new_node_id": "n1", "source": "manual", "reason": ""},
        ),
    ):
        from src.cli.edit_node import run

        rc = run(_args(set_=None, editor=False, format="json"))

    assert rc == 0
    mock_editor.assert_called_once()


def test_run_validation_error_returns_exit_1(capsys):
    """ValidationError from the study backend → exit 1."""
    from src._errors import ValidationError

    with patch("src.study.edit_node.edit_node", side_effect=ValidationError("sum != 1.0")):
        from src.cli.edit_node import run

        rc = run(_args(set_="fold:0.3,call:0.5"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()


def test_run_unexpected_exception_returns_exit_2():
    """A non-ValueError unrelated exception → exit 2."""
    with patch("src.study.edit_node.edit_node", side_effect=RuntimeError("boom")):
        from src.cli.edit_node import run

        rc = run(_args(set_="fold:0.5,call:0.5"))
    assert rc == 2


def test_editor_loop_missing_action_dist_key_raises_valueerror():
    """Removing 'action_dist' in the temp JSON must raise ValueError, not KeyError."""
    from src.cli.edit_node import _editor_loop

    def _fake_run(cmd, **kw):
        with open(cmd[1], "w") as f:
            json.dump({"cluster_key": "ck1"}, f)

    with patch("subprocess.run", side_effect=_fake_run):
        with pytest.raises(ValueError):
            _editor_loop("ck1", current_dist=None)


def test_editor_loop_editor_abort_raises_valueerror():
    """A non-zero editor exit (e.g. :cq) must surface as ValueError, not CalledProcessError."""
    import subprocess

    from src.cli.edit_node import _editor_loop

    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "vi")):
        with pytest.raises(ValueError):
            _editor_loop("ck1", current_dist=None)


def test_editor_loop_editor_not_found_raises_valueerror():
    """EDITOR not on PATH (FileNotFoundError) must surface as ValueError."""
    from src.cli.edit_node import _editor_loop

    with patch("subprocess.run", side_effect=FileNotFoundError("no such editor")):
        with pytest.raises(ValueError):
            _editor_loop("ck1", current_dist=None)


def test_run_editor_abort_returns_exit_1(capsys):
    """Editor abort in editor mode maps to the documented exit-1 contract."""
    import subprocess

    from src.cli.edit_node import run

    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "vi")):
        rc = run(_args(editor=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()
