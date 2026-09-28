"""Canonical game state type shared across all protocol layers.

GameState is the single input type to Canonicalizer.encode() and the output
type of GameStateSource.next_game_state(). It is frozen and immutable.
"""

from __future__ import annotations

from typing import Literal

import msgspec

Street = Literal["preflop", "flop", "turn", "river"]
Position = Literal["UTG", "MP", "CO", "BTN", "SB", "BB"]


class GameState(msgspec.Struct, frozen=True, kw_only=True):
    """Immutable snapshot of a decision point in a 6-max NLHE hand.

    All bet sizes are expressed in big blinds. Card strings are 2-char
    rank+suit (e.g. "Ah", "Kd"). Board cards are 0/3/4/5 elements
    depending on street.
    """

    street: Street
    hero_position: Position
    hero_hole_cards: tuple[str, str]
    board_cards: tuple[str, ...]
    pot_size_bb: float
    effective_stack_bb: float
    hero_facing_bet_bb: float
    hero_bet_size_bb: float
    action_sequence: tuple[str, ...]
    opponents_remaining: int
    prior_street_aggressor: Position | None
    pot_type: str | None = None
