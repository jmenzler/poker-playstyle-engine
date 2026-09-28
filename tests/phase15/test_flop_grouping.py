"""Tests for flop-spot grouping: suit-iso flop collapse, exact stack, multiway detection."""

from __future__ import annotations

import pytest

from src.solver.flop_spot import flop_spot_key, flop_start_stack_bb, is_multiway

_SRP_SEQ = ["BTN:open_2.5", "BB:call", "/", "BB:check"]


def _felt(board: list[str], stack: float, opponents: int = 1) -> dict:
    return {
        "action_sequence": _SRP_SEQ,
        "board_cards": board,
        "effective_stack_bb": stack,
        "hero_position": "BTN",
        "opponents_remaining": opponents,
        "pot_size_bb": 8.0,
        "street": "flop",
        "hero_hole_cards": ["Ac", "Kd"],
    }


def test_suit_isomorphic_flops_collapse_to_same_key():
    felt_a = _felt(["Ah", "Kh", "7c"], stack=97.0)
    felt_b = _felt(["As", "Ks", "7d"], stack=97.0)

    key_a = flop_spot_key(felt_a, _SRP_SEQ, hero_pos="BTN")
    key_b = flop_spot_key(felt_b, _SRP_SEQ, hero_pos="BTN")

    assert key_a == key_b


def test_different_flop_start_stack_distinct_keys():
    felt_a = _felt(["Ah", "Kh", "7c"], stack=97.0)
    felt_b = _felt(["Ah", "Kh", "7c"], stack=56.0)

    key_a = flop_spot_key(felt_a, _SRP_SEQ, hero_pos="BTN")
    key_b = flop_spot_key(felt_b, _SRP_SEQ, hero_pos="BTN")

    assert key_a != key_b
    assert key_a[2] == 97
    assert key_b[2] == 56


def test_key_tuple_structure():
    felt = _felt(["Ah", "Kh", "7c"], stack=100.0)
    canon_flop, range_config, stack = flop_spot_key(felt, _SRP_SEQ, hero_pos="BTN")

    assert isinstance(canon_flop, str)
    assert range_config == ("srp", "BTN", "BB", "BTN", "BTN")
    assert stack == 100


def test_flop_start_stack_prefers_per_hand_lookup():
    felt = _felt(["Ah", "Kh", "7c"], stack=42.0)
    resolved = flop_start_stack_bb(felt, {"hand-1": 110.0}, "hand-1")
    assert resolved == 110.0


def test_flop_start_stack_falls_back_to_remaining():
    felt = _felt(["Ah", "Kh", "7c"], stack=42.0)
    assert flop_start_stack_bb(felt, None, None) == 42.0
    assert flop_start_stack_bb(felt, {"other": 9.0}, "hand-1") == 42.0


def test_per_hand_stack_changes_key():
    felt = _felt(["Ah", "Kh", "7c"], stack=42.0)
    key = flop_spot_key(felt, _SRP_SEQ, hero_pos="BTN", flop_start_stack_by_hand={"h1": 88.0}, hand_id="h1")
    assert key[2] == 88


def test_is_multiway():
    assert is_multiway({"opponents_remaining": 2}) is True
    assert is_multiway({"opponents_remaining": 1}) is False


def test_fewer_than_three_flop_cards_raises():
    felt = _felt(["Ah", "Kh"], stack=100.0)
    with pytest.raises(ValueError, match="3-card flop"):
        flop_spot_key(felt, _SRP_SEQ, hero_pos="BTN")
