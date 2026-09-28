"""Tests for nav_line parsing: seat assignment, pot-fraction sizing, hero_player, runout cards."""

from __future__ import annotations

import pytest

from src.solver.nav_line import MultiwayNavError, parse_nav_line


def test_srp_turn_dp_btn_ip():
    action_sequence = [
        "BTN:open_2.5",
        "BB:call",
        "/",
        "BB:check",
        "BTN:bet_half_pot",
        "BB:call",
        "/",
        "BB:check",
    ]
    board_cards = ["Ah", "Kh", "7c", "2d"]

    nav_steps, hero_player, turn_card, river_card = parse_nav_line(
        action_sequence, hero_pos="BTN", villain_pos="BB", board_cards=board_cards
    )

    assert hero_player == 1
    assert turn_card == board_cards[3]
    assert river_card is None
    assert nav_steps == [
        {"seat": 0, "kind": "check", "frac": None},
        {"seat": 1, "kind": "bet", "frac": 0.5},
        {"seat": 0, "kind": "call", "frac": None},
        {"seat": 0, "kind": "check", "frac": None},
    ]


def test_hero_oop_seat_zero():
    action_sequence = ["BTN:open_2.5", "BB:call", "/", "BB:check"]
    _, hero_player, _, _ = parse_nav_line(
        action_sequence, hero_pos="BB", villain_pos="BTN", board_cards=["Ah", "Kh", "7c"]
    )
    assert hero_player == 0


def test_bet_and_raise_fracs():
    action_sequence = ["/", "BB:bet_half_pot", "BTN:raise_pot", "BB:call"]
    nav_steps, _, _, _ = parse_nav_line(
        action_sequence, hero_pos="BTN", villain_pos="BB", board_cards=["Ah", "Kh", "7c"]
    )
    assert nav_steps[0] == {"seat": 0, "kind": "bet", "frac": 0.5}
    assert nav_steps[1] == {"seat": 1, "kind": "raise", "frac": 1.0}
    assert nav_steps[2]["frac"] is None


def test_passive_actions_carry_none_frac():
    action_sequence = ["/", "BB:check", "BTN:check"]
    nav_steps, _, _, _ = parse_nav_line(
        action_sequence, hero_pos="BTN", villain_pos="BB", board_cards=["Ah", "Kh", "7c"]
    )
    assert all(step["frac"] is None for step in nav_steps)


def test_river_card_from_five_card_board():
    action_sequence = ["/", "BB:check", "/", "BB:check", "/", "BB:check"]
    _, _, turn_card, river_card = parse_nav_line(
        action_sequence, hero_pos="BTN", villain_pos="BB", board_cards=["Ah", "Kh", "7c", "2d", "9s"]
    )
    assert turn_card == "2d"
    assert river_card == "9s"


def test_unknown_size_defaults_to_pot():
    action_sequence = ["/", "BB:bet_weird_size"]
    nav_steps, _, _, _ = parse_nav_line(
        action_sequence, hero_pos="BTN", villain_pos="BB", board_cards=["Ah", "Kh", "7c"]
    )
    assert nav_steps[0]["frac"] == 1.0


def test_sim_call_with_no_bet_pending_maps_to_check():
    # SIM vocab emits "call" for a check; the tree only offers Check at that node.
    action_sequence = ["SB:call", "BB:raise_half_pot", "SB:call", "/", "SB:call", "BB:call"]
    nav_steps, _, _, _ = parse_nav_line(
        action_sequence, hero_pos="SB", villain_pos="BB", board_cards=["Ah", "Kh", "7c", "2d"]
    )
    assert nav_steps == [
        {"seat": 0, "kind": "check", "frac": None},
        {"seat": 1, "kind": "check", "frac": None},
    ]


def test_sim_raise_with_no_bet_pending_maps_to_bet():
    # SIM vocab emits "raise_*" for a lead bet; the tree only offers Bet at that node.
    action_sequence = ["/", "BB:raise_half_pot", "UTG:raise_pot", "BB:call"]
    nav_steps, _, _, _ = parse_nav_line(
        action_sequence, hero_pos="BB", villain_pos="UTG", board_cards=["Ah", "Kh", "7c", "2d"]
    )
    assert nav_steps == [
        {"seat": 0, "kind": "bet", "frac": 0.5},
        {"seat": 1, "kind": "raise", "frac": 1.0},
        {"seat": 0, "kind": "call", "frac": None},
    ]


def test_facing_bet_resets_each_street():
    action_sequence = ["/", "BB:raise_half_pot", "BTN:call", "/", "BB:call"]
    nav_steps, _, _, _ = parse_nav_line(
        action_sequence, hero_pos="BTN", villain_pos="BB", board_cards=["Ah", "Kh", "7c", "2d", "9s"]
    )
    assert nav_steps == [
        {"seat": 0, "kind": "bet", "frac": 0.5},
        {"seat": 1, "kind": "call", "frac": None},
        {"seat": 0, "kind": "check", "frac": None},
    ]


def test_allin_sets_facing_bet():
    action_sequence = ["/", "BB:allin", "BTN:call"]
    nav_steps, _, _, _ = parse_nav_line(
        action_sequence, hero_pos="BTN", villain_pos="BB", board_cards=["Ah", "Kh", "7c"]
    )
    assert nav_steps == [
        {"seat": 0, "kind": "allin", "frac": None},
        {"seat": 1, "kind": "call", "frac": None},
    ]


def test_position_outside_hu_pair_raises():
    # MultiwayNavError (a ValueError) so _solve_group can catch it specifically and
    # skip the multiway spot instead of tripping the db-failure circuit breaker.
    action_sequence = ["/", "CO:bet_half_pot"]
    with pytest.raises(MultiwayNavError, match="not in HU pair"):
        parse_nav_line(action_sequence, hero_pos="BTN", villain_pos="BB", board_cards=["Ah", "Kh", "7c"])
