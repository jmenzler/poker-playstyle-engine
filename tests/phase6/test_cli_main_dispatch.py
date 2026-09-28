"""tests/phase6/test_cli_main_dispatch.py — Verify src/cli/main.py registers
all Phase-6 subparsers and dispatches each to the correct shim.

Each test monkeypatches the shim ``run()`` so backend imports are not exercised
— this is a pure dispatcher integration test.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def test_main_help_lists_all_phase6_subcommands(capsys):
    """`poker-engine --help` lists all 8 Phase-6 subcommands."""
    from src.cli.main import main

    with pytest.raises(SystemExit) as exc_info, patch.object(sys, "argv", ["poker-engine", "--help"]):
        main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("ingest", "similar", "edit-node", "leaks", "ab", "dashboard", "patches", "eval"):
        assert cmd in out, f"`{cmd}` not in --help output"


def test_leaks_help_shows_all_flags(capsys):
    """`poker-engine leaks --help` mentions --type, --min-n, --limit, --format."""
    from src.cli.main import main

    with (
        pytest.raises(SystemExit) as exc_info,
        patch.object(sys, "argv", ["poker-engine", "leaks", "--help"]),
    ):
        main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--type", "--min-n", "--limit", "--format"):
        assert flag in out, f"`{flag}` not in leaks --help"


def test_patches_help_shows_list_and_rollback(capsys):
    """`poker-engine patches --help` mentions list and rollback subcommands."""
    from src.cli.main import main

    with (
        pytest.raises(SystemExit) as exc_info,
        patch.object(sys, "argv", ["poker-engine", "patches", "--help"]),
    ):
        main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "list" in out
    assert "rollback" in out


@pytest.mark.parametrize(
    "argv,shim_module",
    [
        (["ingest"], "src.cli.ingest"),
        (["ingest", "--rebuild"], "src.cli.ingest"),
        (["similar", "ck1"], "src.cli.similar"),
        (["similar", "ck1", "--k", "5"], "src.cli.similar"),
        (["edit-node", "ck1", "--set", "fold:1.0"], "src.cli.edit_node"),
        (["leaks"], "src.cli.leaks"),
        (["leaks", "--type", "strategy", "--min-n", "5"], "src.cli.leaks"),
        (["ab", "ck1", "p1"], "src.cli.ab"),
        (["dashboard"], "src.cli.dashboard"),
        (["patches", "list", "--limit", "5"], "src.cli.patches"),
        (["patches", "rollback", "00000000-0000-0000-0000-000000000001"], "src.cli.patches"),
        (["eval", "--opponent", "random", "--hands", "10", "--no-persist"], "src.cli.eval"),
    ],
)
def test_dispatch_invokes_correct_shim(argv, shim_module, monkeypatch):
    """Each subcommand routes to the correct shim's run()."""
    invocations: list[str] = []

    def _stub_run(args, **kw):
        invocations.append(shim_module)
        return 0

    monkeypatch.setattr(f"{shim_module}.run", _stub_run, raising=False)

    from src.cli.main import main

    with patch.object(sys, "argv", ["poker-engine", *argv]):
        rc = main()

    assert rc == 0
    assert invocations == [shim_module], f"expected dispatch to {shim_module}, got {invocations}"


def test_existing_uncertain_spots_still_dispatches(monkeypatch):
    """Back-compat: existing `uncertain-spots` parser still routes (Phase 4)."""
    called = []

    def _stub_run(args, **kw):
        called.append("uncertain_spots")
        return 0

    monkeypatch.setattr("src.cli.uncertain_spots.run", _stub_run, raising=False)

    from src.cli.main import main

    with patch.object(sys, "argv", ["poker-engine", "uncertain-spots", "--limit", "10"]):
        rc = main()
    assert rc == 0
    assert called == ["uncertain_spots"]


def test_existing_rollback_still_dispatches(monkeypatch):
    """Back-compat: existing top-level `rollback <patch_id>` still routes (Phase 5)."""
    called = []

    def _stub_run(args, **kw):
        called.append("rollback")
        return 0

    monkeypatch.setattr("src.cli.rollback.run", _stub_run, raising=False)

    from src.cli.main import main

    with patch.object(sys, "argv", ["poker-engine", "rollback", "00000000-0000-0000-0000-000000000001"]):
        rc = main()
    assert rc == 0
    assert called == ["rollback"]
