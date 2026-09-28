"""Hypothesis strategies and fixtures for property-based tests.

Provides:
- card_strategy: single valid card string
- hole_cards_strategy: 2 distinct hole cards
- board_strategy(street, excluded): 0/3/4/5 board cards per street
- action_sequence_strategy(street): plausible action token sequences
- game_state_strategy(street=None): fully valid GameState
- suit_permutation_strategy: random suit permutation dict
- apply_suit_permutation(gs, perm): remap suits in a GameState
- no_network fixture: blocks socket I/O to enforce CANON-01 invariant #9
"""

from __future__ import annotations

import socket
from typing import Final

import pytest
from hypothesis import strategies as st

from src.protocols.game_state import GameState

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

RANKS: Final[tuple[str, ...]] = tuple("23456789TJQKA")
SUITS: Final[tuple[str, str, str, str]] = ("c", "d", "h", "s")
STREETS: Final[tuple[str, str, str, str]] = ("preflop", "flop", "turn", "river")
POSITIONS: Final[tuple[str, str, str, str, str, str]] = (
    "UTG",
    "MP",
    "CO",
    "BTN",
    "SB",
    "BB",
)

_BOARD_CARD_COUNT: Final[dict[str, int]] = {
    "preflop": 0,
    "flop": 3,
    "turn": 4,
    "river": 5,
}

_ACTION_VERBS: Final[tuple[str, ...]] = (
    "open_2.5",
    "call",
    "check",
    "bet_50",
    "bet_75",
    "raise_3x",
    "fold",
)

# ---------------------------------------------------------------------------
# Primitive strategies
# ---------------------------------------------------------------------------


@st.composite
def card_strategy(draw: st.DrawFn) -> str:
    """Draw a single valid 2-char card string (rank + suit)."""
    rank = draw(st.sampled_from(RANKS))
    suit = draw(st.sampled_from(SUITS))
    return rank + suit


@st.composite
def hole_cards_strategy(draw: st.DrawFn) -> tuple[str, str]:
    """Draw 2 distinct hole cards, returned sorted descending by string value."""
    cards: list[str] = []
    seen: set[str] = set()
    while len(cards) < 2:
        card = draw(card_strategy())
        if card not in seen:
            seen.add(card)
            cards.append(card)
    cards.sort(reverse=True)
    return (cards[0], cards[1])


@st.composite
def board_strategy(
    draw: st.DrawFn,
    *,
    street: str,
    excluded: set[str] | None = None,
) -> tuple[str, ...]:
    """Draw board cards appropriate for the given street.

    Returns an empty tuple for preflop. For flop/turn/river returns
    the correct count of cards, all distinct from each other and from
    the `excluded` set (typically the hero's hole cards).
    """
    count = _BOARD_CARD_COUNT[street]
    if count == 0:
        return ()

    blocked: set[str] = set(excluded) if excluded else set()
    cards: list[str] = []
    seen: set[str] = set(blocked)
    while len(cards) < count:
        card = draw(card_strategy())
        if card not in seen:
            seen.add(card)
            cards.append(card)
    return tuple(cards)


@st.composite
def action_sequence_strategy(draw: st.DrawFn, *, street: str) -> tuple[str, ...]:
    """Draw a plausible sequence of action tokens for the given street."""
    length = draw(st.integers(min_value=1, max_value=4))
    tokens: list[str] = []
    for _ in range(length):
        pos = draw(st.sampled_from(POSITIONS))
        verb = draw(st.sampled_from(_ACTION_VERBS))
        tokens.append(f"{pos}:{verb}")
    return tuple(tokens)


# ---------------------------------------------------------------------------
# GameState composite strategy
# ---------------------------------------------------------------------------


@st.composite
def game_state_strategy(
    draw: st.DrawFn,
    *,
    street: str | None = None,
) -> GameState:
    """Draw a fully valid GameState.

    All float fields are bounded (no NaN, no infinity) per the threat model
    requirement to avoid non-deterministic test failures.
    """
    chosen_street: str = street if street is not None else draw(st.sampled_from(STREETS))

    hole = draw(hole_cards_strategy())
    board = draw(board_strategy(street=chosen_street, excluded=set(hole)))

    pot_size_bb: float = draw(
        st.floats(min_value=1.5, max_value=200.0, allow_nan=False, allow_infinity=False)
    )
    effective_stack_bb: float = draw(
        st.floats(min_value=10.0, max_value=200.0, allow_nan=False, allow_infinity=False)
    )
    hero_facing_bet_bb: float = draw(
        st.floats(
            min_value=0.0,
            max_value=effective_stack_bb,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    hero_bet_size_bb: float = draw(
        st.floats(
            min_value=0.0,
            max_value=effective_stack_bb,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    action_seq = draw(action_sequence_strategy(street=chosen_street))
    hero_position = draw(st.sampled_from(POSITIONS))
    opponents_remaining = draw(st.integers(min_value=1, max_value=5))
    prior_street_aggressor = draw(st.one_of(st.sampled_from(POSITIONS), st.none()))

    return GameState(
        street=chosen_street,  # type: ignore[arg-type]
        hero_position=hero_position,  # type: ignore[arg-type]
        hero_hole_cards=hole,
        board_cards=board,
        pot_size_bb=pot_size_bb,
        effective_stack_bb=effective_stack_bb,
        hero_facing_bet_bb=hero_facing_bet_bb,
        hero_bet_size_bb=hero_bet_size_bb,
        action_sequence=action_seq,
        opponents_remaining=opponents_remaining,
        prior_street_aggressor=prior_street_aggressor,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Suit permutation strategy + helper
# ---------------------------------------------------------------------------


@st.composite
def suit_permutation_strategy(draw: st.DrawFn) -> dict[str, str]:
    """Draw a random permutation of the four suits as a mapping dict.

    Returns a dict like {'c': 'h', 'd': 's', 'h': 'c', 's': 'd'}.
    """
    permuted: list[str] = draw(st.permutations(list(SUITS)))
    return dict(zip(SUITS, permuted))


def apply_suit_permutation(gs: GameState, perm: dict[str, str]) -> GameState:
    """Return a new GameState with each card's suit remapped through perm.

    Only card suits are changed (hole cards + board cards). All other
    fields (street, positions, bet sizes, etc.) are preserved exactly.

    Args:
        gs: The original GameState.
        perm: A suit permutation mapping, e.g. {'c': 'h', ...}.

    Returns:
        A new frozen GameState with remapped card suits.
    """

    def remap(card: str) -> str:
        rank, suit = card[:-1], card[-1]
        return rank + perm[suit]

    new_hole = (remap(gs.hero_hole_cards[0]), remap(gs.hero_hole_cards[1]))
    new_board = tuple(remap(c) for c in gs.board_cards)

    return GameState(
        street=gs.street,
        hero_position=gs.hero_position,
        hero_hole_cards=new_hole,
        board_cards=new_board,
        pot_size_bb=gs.pot_size_bb,
        effective_stack_bb=gs.effective_stack_bb,
        hero_facing_bet_bb=gs.hero_facing_bet_bb,
        hero_bet_size_bb=gs.hero_bet_size_bb,
        action_sequence=gs.action_sequence,
        opponents_remaining=gs.opponents_remaining,
        prior_street_aggressor=gs.prior_street_aggressor,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block all socket I/O to enforce CANON-01 invariant #9.

    Monkeypatches socket.socket and socket.create_connection so that any
    attempt to open a network connection during encode() raises immediately.
    Tests that use this fixture assert that canonicalization is pure and
    performs zero network I/O.
    """
    _msg = "encode() attempted network I/O — violates CANON-01"

    def _raise(*args: object, **kwargs: object) -> None:
        raise AssertionError(_msg)

    monkeypatch.setattr(socket, "socket", _raise)
    monkeypatch.setattr(socket, "create_connection", _raise, raising=False)
