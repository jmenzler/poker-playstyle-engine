"""Phase 15 test fixtures."""

from __future__ import annotations

from typing import Any

import pytest

from src.protocols.game_state import GameState


@pytest.fixture
def make_game_state():
    """Factory fixture for building a valid GameState with sensible defaults.

    Callers pass only the fields they care about; everything else defaults to
    a valid flop state (mirrors the canonical fixture pattern from tests/unit/
    test_encoder.py and tests/property/conftest.py).

    Usage:
        gs = make_game_state(pot_type="3bet")
        gs = make_game_state(street="preflop", pot_size_bb=3.0)
    """

    def _factory(**overrides: Any) -> GameState:
        defaults: dict[str, Any] = {
            "street": "flop",
            "hero_position": "BTN",
            "hero_hole_cards": ("Ah", "Kd"),
            "board_cards": ("Jc", "7s", "2h"),
            "pot_size_bb": 6.0,
            "effective_stack_bb": 97.5,
            "hero_facing_bet_bb": 0.0,
            "hero_bet_size_bb": 0.0,
            "action_sequence": ("BB:check",),
            "opponents_remaining": 1,
            "prior_street_aggressor": "BTN",
        }
        defaults.update(overrides)
        return GameState(**defaults)

    return _factory
