"""Tests for src/api/main._include_routers — fail loud on a broken router.

A router ImportError must propagate (fail startup), not be swallowed into an
app that boots with endpoints silently dropped (ERR-01).
"""

from __future__ import annotations

import builtins

import pytest


def test_include_routers_propagates_import_error(monkeypatch):
    """A failing router import bubbles out of _include_routers, not swallowed."""
    import src.api.main as main

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "src.api.gaps":
            raise ImportError("simulated broken router")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="simulated broken router"):
        main._include_routers()


def test_all_routers_attach_on_real_import():
    """Sanity: the real routers all import and attach without error."""
    import src.api.main as main

    main._include_routers()  # must not raise
