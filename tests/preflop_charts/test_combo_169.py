"""combo_169: map a 2-card hand to its 169-class (rank-desc + s/o/pair)."""

from __future__ import annotations

import itertools

import pytest

from tools.preflop_combo import ALL_COMBOS_169, combo_169


def test_suited_high_first():
    assert combo_169("Ah", "Kh") == "AKs"


def test_offsuit_returns_o():
    assert combo_169("Ah", "Ks") == "AKo"


def test_pair_ignores_suit():
    assert combo_169("7c", "7d") == "77"


def test_order_independent():
    assert combo_169("Kd", "Ah") == combo_169("Ah", "Kd") == "AKo"
    assert combo_169("Kh", "Ah") == combo_169("Ah", "Kh") == "AKs"


def test_low_pair():
    assert combo_169("2c", "2s") == "22"


def test_suited_low():
    assert combo_169("5h", "2h") == "52s"


def test_all_52c2_map_into_169():
    ranks = "AKQJT98765432"
    suits = "cdhs"
    deck = [r + s for r in ranks for s in suits]
    seen: set[str] = set()
    for a, b in itertools.combinations(deck, 2):
        cls = combo_169(a, b)
        assert cls is not None
        assert cls in ALL_COMBOS_169, f"{a}{b} -> {cls!r} not in ALL_COMBOS_169"
        seen.add(cls)
    # every one of the 169 classes is produced by some 2-card combo
    assert seen == set(ALL_COMBOS_169)
    assert len(ALL_COMBOS_169) == 169


@pytest.mark.parametrize("bad", [("Zh", "Kh"), ("Ah",), ("Ah", "Kh", "Qh")])
def test_malformed_raises(bad):
    with pytest.raises((ValueError, TypeError)):
        combo_169(*bad)


@pytest.mark.parametrize("bad", ["AZ", "A1", "Kx", "7 "])
def test_bad_suit_char_raises(bad):
    """A junk suit must fail loud, not silently classify e.g. AZ/KZ as 'AKs'."""
    with pytest.raises(ValueError):
        combo_169(bad, "Kh")
