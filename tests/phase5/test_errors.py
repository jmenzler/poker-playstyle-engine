"""tests/phase5/test_errors.py — PatchForbiddenError + PatchConflictError type tests.

Requirements covered (Plan 02):
- PTCH-04: PatchForbiddenError is a RuntimeError subclass
- ERR-04: PatchConflictError is a RuntimeError subclass

Tests are minimal leaf-type assertions — message construction happens at the raise site.
"""

from __future__ import annotations

import pytest

from src._errors import PatchConflictError, PatchForbiddenError


def test_patch_forbidden_error_is_runtime_error() -> None:
    """PatchForbiddenError inherits RuntimeError (not LookupError)."""
    assert issubclass(PatchForbiddenError, RuntimeError)


def test_patch_conflict_error_is_runtime_error() -> None:
    """PatchConflictError inherits RuntimeError (not LookupError)."""
    assert issubclass(PatchConflictError, RuntimeError)


def test_both_importable_from_src_errors() -> None:
    """Both error classes are importable as 'from src._errors import ...'."""
    from src._errors import PatchConflictError as PCE
    from src._errors import PatchForbiddenError as PFE

    assert PFE is PatchForbiddenError
    assert PCE is PatchConflictError


def test_patch_forbidden_error_message_format() -> None:
    """PatchForbiddenError raised with cluster_key in message is retrievable."""
    ck = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
    with pytest.raises(PatchForbiddenError, match="cluster_key="):
        raise PatchForbiddenError(f"PTCH-04: cluster_key={ck!r}")
