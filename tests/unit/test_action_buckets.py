"""Preflop stem -> sized canonical label snapping (tools/action_buckets)."""

from __future__ import annotations

import pytest

from tools.action_buckets import (
    preflop_action_label,
    snap_postflop_action,
    snap_postflop_bet,
    snap_postflop_raise,
)


@pytest.mark.parametrize(
    "stem,pot_type,pos_rel,pos,expected",
    [
        # raise facing an open (srp) -> 3bet, sized by IP/OOP
        ("raise", "srp", "OOP", "SB", "3bet_4x"),
        ("raise", "srp", "IP", "BTN", "3bet_3x"),
        # raise in a 3bet pot -> 4bet (uniform size)
        ("raise", "3bet", "IP", "CO", "4bet_2_5x"),
        ("raise", "3bet", "OOP", "BB", "4bet_2_5x"),
        # raise in 4bet / 5bet+ -> jam
        ("raise", "4bet", "IP", "BTN", "allin"),
        ("raise", "5bet+", "OOP", "SB", "allin"),
        # opens (unopened/limp): BvB blinds -> 3bb, else 2.2bb
        ("raise", "limp", "OOP", "SB", "open_3bb"),
        ("raise", "limp", "OOP", "BB", "open_3bb"),
        ("raise", "limp", "IP", "BTN", "open_2_2bb"),
        ("raise", "limp", "IP", "CO", "open_2_2bb"),
        # missing pos -> open default
        ("raise", "limp", "IP", None, "open_2_2bb"),
        # non-raise stems pass through
        ("fold", "srp", "OOP", "SB", "fold"),
        ("call", "srp", "IP", "BTN", "call"),
        ("check", "limp", "OOP", "BB", "check"),
        ("allin", "3bet", "IP", "CO", "allin"),
        # unknown stem passes through (blend will skip if non-canonical)
        ("weird", "srp", "IP", "BTN", "weird"),
    ],
)
def test_preflop_action_label(stem, pot_type, pos_rel, pos, expected):
    assert preflop_action_label(stem, pot_type, pos_rel, pos) == expected


@pytest.mark.parametrize(
    "frac,is_allin,expected",
    [
        (0.26, False, "bet_25"),
        (0.40, False, "bet_33"),
        (0.50, False, "bet_50"),
        (0.80, False, "bet_75"),
        (1.05, False, "bet_100"),
        (1.40, False, "bet_150"),
        (2.50, False, "bet_overbet"),
        (0.50, True, "allin"),
        (None, False, "bet_50"),
    ],
)
def test_snap_postflop_bet(frac, is_allin, expected):
    assert snap_postflop_bet(frac, is_allin) == expected


@pytest.mark.parametrize(
    "ratio,is_allin,expected",
    [
        (1.1, False, "raise_min"),
        (2.4, False, "raise_2_5x"),
        (3.1, False, "raise_3x"),
        (4.2, False, "raise_pot"),
        (3.0, True, "allin"),
        (None, False, "raise_2_5x"),
    ],
)
def test_snap_postflop_raise(ratio, is_allin, expected):
    assert snap_postflop_raise(ratio, is_allin) == expected


def test_snap_postflop_action_dispatch():
    assert snap_postflop_action("bet", 0.50, -1.0, False) == "bet_50"
    assert snap_postflop_action("raise", 0.0, 3.0, False) == "raise_3x"
    assert snap_postflop_action("check", None, None, False) == "check"
    assert snap_postflop_action("call", None, None, False) == "call"
    assert snap_postflop_action("bet", 0.50, -1.0, True) == "allin"
