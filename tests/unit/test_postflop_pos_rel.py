"""Regression tests for D-12a (postflop hero_pos_rel) and D-15a (facing_pos).  long-ok

D-12a: encoder._adapt() hardcodes hero_pos_rel="IP" for all postflop streets — an OOP
hero (e.g. SB facing a BTN aggressor) should produce "OOP" not "IP".

D-15a: encoder._adapt() sets facing_pos=None unconditionally — the villain seat
(prior_street_aggressor) must be populated instead.

Both bugs live in the same _adapt() postflop branch; one fix resolves both.
"""

from __future__ import annotations

import pytest

from src.canonicalizer import Canonicalizer
from src.protocols.game_state import GameState


@pytest.fixture(scope="module")
def canonicalizer() -> Canonicalizer:
    return Canonicalizer.default()


def _oop_flop_gs() -> GameState:
    """SB hero facing BTN aggressor on the flop — hero is OOP."""
    return GameState(
        street="flop",
        hero_position="SB",
        hero_hole_cards=("Ah", "Kd"),
        board_cards=("Jc", "7s", "2h"),
        pot_size_bb=6.0,
        effective_stack_bb=97.5,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=("SB:check",),
        opponents_remaining=1,
        prior_street_aggressor="BTN",
    )


def _ip_flop_gs() -> GameState:
    """BTN hero facing UTG aggressor on the flop — hero is IP."""
    return GameState(
        street="flop",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kd"),
        board_cards=("Jc", "7s", "2h"),
        pot_size_bb=6.0,
        effective_stack_bb=97.5,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=("BTN:check",),
        opponents_remaining=1,
        prior_street_aggressor="UTG",
    )


def test_postflop_pos_rel_oop(canonicalizer: Canonicalizer) -> None:
    """D-12a: postflop SB hero vs BTN aggressor must produce hero_pos_rel='OOP'."""
    result = canonicalizer.encode(_oop_flop_gs())
    assert result.hard_filter["hero_pos_rel"] == "OOP", (
        f"Expected OOP for SB vs BTN aggressor, got {result.hard_filter['hero_pos_rel']!r}"
    )


def test_postflop_pos_rel_ip(canonicalizer: Canonicalizer) -> None:
    """D-12a: postflop BTN hero vs UTG aggressor must produce hero_pos_rel='IP'."""
    result = canonicalizer.encode(_ip_flop_gs())
    assert result.hard_filter["hero_pos_rel"] == "IP", (
        f"Expected IP for BTN vs UTG aggressor, got {result.hard_filter['hero_pos_rel']!r}"
    )


def test_postflop_facing_pos_set_from_aggressor(canonicalizer: Canonicalizer) -> None:
    """D-15a: facing_pos must equal prior_street_aggressor, not None."""
    gs = _oop_flop_gs()
    from src.canonicalizer.encoder import _adapt

    dp = _adapt(gs)
    assert dp["facing_pos"] == gs.prior_street_aggressor, (
        f"Expected facing_pos={gs.prior_street_aggressor!r}, got {dp['facing_pos']!r}"
    )
    assert dp["facing_pos"] is not None, "facing_pos must not be None on postflop"
