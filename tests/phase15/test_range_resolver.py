"""Range resolver unit tests.  long-ok

Verifies _combos_to_postflop_cli_notation output format and _resolve_ranges
fallback chain against the preflop palette at research/preflop-ranges/outputs/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PALETTE_DIR = Path("research/preflop-ranges/outputs")
PLACEHOLDER = "AA-22,AKs-A2s"


def test_notation_format():
    """_combos_to_postflop_cli_notation converts 0-1 freq dict to postflop-cli notation.

    Format rules (from WAVE0 VERDICT):
    - freq == 1.0 (within 0.001) → bare hand, no suffix
    - freq < 1.0 → "HAND:{freq:.2f}"
    - freq <= 0 → excluded
    - comma-separated, no spaces
    """
    from src.solver.range_resolver import _combos_to_postflop_cli_notation

    result = _combos_to_postflop_cli_notation({"AA": 1.0, "KK": 0.9, "AKs": 0.75})
    assert result == "AA,KK:0.90,AKs:0.75", f"got {result!r}"


def test_notation_excludes_zero():
    """Zero-frequency combos are excluded from notation."""
    from src.solver.range_resolver import _combos_to_postflop_cli_notation

    result = _combos_to_postflop_cli_notation({"AA": 1.0, "KK": 0.0, "QQ": 0.5})
    assert "KK" not in result
    assert "AA" in result
    assert "QQ" in result


def test_3bet_bb_resolve():
    """_resolve_ranges for BTN 3bet vs BB returns non-placeholder strings.

    Uses the real palette at research/preflop-ranges/outputs/.
    ip_key='BTN/3bet_vs_BB', oop_key='BB/defend_3bet_vs_BTN'.
    """
    from src.solver.range_resolver import _resolve_ranges, build_range_lookup

    if not PALETTE_DIR.exists():
        pytest.skip("palette dir not available in this environment")

    lookup = build_range_lookup(PALETTE_DIR)
    range_ip, range_oop = _resolve_ranges(
        obs_spot_features={},
        hero_pos="BTN",
        villain_pos="BB",
        pot_type="3bet",
        palette_lookup=lookup,
    )
    assert range_ip != PLACEHOLDER, f"range_ip still placeholder: {range_ip!r}"
    assert range_oop != PLACEHOLDER, f"range_oop still placeholder: {range_oop!r}"
    assert len(range_ip) > 0
    assert len(range_oop) > 0


def test_srp_resolve():
    """_resolve_ranges for an srp pot (CO open, BB defend) returns non-placeholder.

    keys: opener 'CO/open', caller 'BB/defend_vs_CO'.
    """
    from src.solver.range_resolver import _resolve_ranges, build_range_lookup

    if not PALETTE_DIR.exists():
        pytest.skip("palette dir not available in this environment")

    lookup = build_range_lookup(PALETTE_DIR)
    range_ip, range_oop = _resolve_ranges(
        obs_spot_features={},
        hero_pos="BB",
        villain_pos="CO",
        pot_type="srp",
        palette_lookup=lookup,
        opener_pos="CO",
        bettor_pos="CO",
    )
    assert range_ip != PLACEHOLDER, f"range_ip still placeholder: {range_ip!r}"
    assert range_oop != PLACEHOLDER, f"range_oop still placeholder: {range_oop!r}"


def test_limp_resolve():
    """_resolve_ranges for a limp pot returns both players' limp ranges (non-placeholder)."""
    from src.solver.range_resolver import _resolve_ranges, build_range_lookup

    if not PALETTE_DIR.exists():
        pytest.skip("palette dir not available in this environment")

    lookup = build_range_lookup(PALETTE_DIR)
    range_ip, range_oop = _resolve_ranges(
        obs_spot_features={},
        hero_pos="BB",
        villain_pos="BTN",
        pot_type="limp",
        palette_lookup=lookup,
    )
    assert range_ip != PLACEHOLDER, f"range_ip still placeholder: {range_ip!r}"
    assert range_oop != PLACEHOLDER, f"range_oop still placeholder: {range_oop!r}"


def test_fallback_to_placeholder():
    """Unknown spot falls back to placeholder AND invokes log.warning on miss."""
    from unittest.mock import patch

    from src.solver import range_resolver
    from src.solver.range_resolver import _resolve_ranges

    warning_calls = []
    original_warning = range_resolver.log.warning

    def capture_warning(event, **kw):
        warning_calls.append({"event": event, **kw})
        return original_warning(event, **kw)

    with patch.object(range_resolver.log, "warning", side_effect=capture_warning):
        range_ip, range_oop = _resolve_ranges(
            obs_spot_features={},
            hero_pos="CO",
            villain_pos="UTG",
            pot_type="5bet",
            palette_lookup={},
        )

    assert range_ip == PLACEHOLDER
    assert range_oop == PLACEHOLDER
    assert len(warning_calls) >= 1, "expected log.warning called on placeholder fallback"
    assert any("placeholder_fallback" in str(c.get("event", "")) for c in warning_calls)


def test_obs_stored_ranges_preferred():
    """obs_spot_features range_ip/range_oop win over palette lookup."""
    from src.solver.range_resolver import _resolve_ranges, build_range_lookup

    obs_features = {"range_ip": "AA,KK", "range_oop": "QQ,JJ"}
    lookup = build_range_lookup(PALETTE_DIR) if PALETTE_DIR.exists() else {}

    range_ip, range_oop = _resolve_ranges(
        obs_spot_features=obs_features,
        hero_pos="BTN",
        villain_pos="BB",
        pot_type="3bet",
        palette_lookup=lookup,
    )
    assert range_ip == "AA,KK"
    assert range_oop == "QQ,JJ"
