"""ChenBot — Chen formula preflop + treys/pot-odds postflop baseline.

Chen formula implemented fresh from public specification (thepokerbank.com);
no third-party code copied (D-09-9).
Hand evaluation via vendored treys, see src/vendor/treys/ATTRIBUTION.md.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.eval.strategy import (
    CHECK_CALL,
    FOLD,
    HALF_POT,
    POT,
    pick_first_legal,
)
from src.vendor.treys import Card, Evaluator

_TREYS_EVAL = Evaluator()

_RANK_BASE: dict[str, float] = {
    "A": 10.0,
    "K": 8.0,
    "Q": 7.0,
    "J": 6.0,
    "T": 5.0,
    "9": 4.5,
    "8": 4.0,
    "7": 3.5,
    "6": 3.0,
    "5": 2.5,
    "4": 2.0,
    "3": 1.5,
    "2": 1.0,
}

_RANK_ORDER: dict[str, int] = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12,
}


def chen_score(hand: tuple[str, ...] | list[str]) -> float:
    """Compute the Chen formula score for a two-card hand.

    Args:
        hand: Two card strings in project format (e.g. ``("As", "Kh")``).

    Returns:
        Chen score as a float (higher = stronger preflop hand).
    """
    c1, c2 = hand[0], hand[1]
    r1, s1 = c1[0].upper(), c1[1].lower()
    r2, s2 = c2[0].upper(), c2[1].lower()

    base1 = _RANK_BASE.get(r1, 1.0)
    base2 = _RANK_BASE.get(r2, 1.0)

    ord1 = _RANK_ORDER.get(r1, 0)
    ord2 = _RANK_ORDER.get(r2, 0)

    # Ensure high card is first
    if base1 < base2:
        base1, base2 = base2, base1
        r1, r2 = r2, r1
        ord1, ord2 = ord2, ord1

    is_pair = r1 == r2
    is_suited = s1 == s2
    gap = ord1 - ord2 - 1  # 0 for connectors, 1 for one-gappers, etc.

    if is_pair:
        score = max(5.0, base1 * 2.0)
    else:
        score = base1

        if is_suited:
            score += 2.0

        if gap == 0:
            pass  # connector, no penalty
        elif gap == 1:
            score -= 1.0
        elif gap == 2:
            score -= 2.0
        elif gap == 3:
            score -= 4.0
        else:
            score -= 5.0

        # Low connector bonus: gap 0-1 and both cards below queen
        if gap <= 1 and ord1 < _RANK_ORDER["Q"]:
            score += 1.0

    return math.ceil(score - 0.5) + 0.5 * (1 if (score - math.floor(score)) >= 0.5 else 0)


def postflop_hand_class(hole: list[str], board: list[str]) -> str:
    """Classify a postflop made hand using treys.

    Args:
        hole: Hero hole cards as strings (e.g. ``["Ah", "Kh"]``).
        board: Board cards as strings (e.g. ``["Qh", "Jh", "Th"]``).

    Returns:
        One of ``"strong"``, ``"medium"``, ``"weak"``, ``"draw"``.
    """
    try:
        h = [Card.new(c) for c in hole]
        b = [Card.new(c) for c in board]
        rank = _TREYS_EVAL.evaluate(h, b)
        rank_class = _TREYS_EVAL.get_rank_class(rank)
        # rank_class: 1=SF, 2=4oak, 3=FH, 4=Flush, 5=Straight, 6=3oak, 7=2pair, 8=pair, 9=HC
        if rank_class <= 5:
            return "strong"
        if rank_class == 6:
            return "strong"
        if rank_class == 7:
            return "medium"
        if rank_class == 8:
            return "weak"
        # High card — check for draw potential (4 board cards or suited hole)
        if len(board) < 4 and _has_draw_potential(hole, board):
            return "draw"
        return "weak"
    except Exception:
        return "weak"


def _has_draw_potential(hole: list[str], board: list[str]) -> bool:
    """Return True if hero has a flush draw or open-ended straight draw."""
    all_cards = hole + board
    suits = [c[1].lower() for c in all_cards]
    flush_draw = any(suits.count(s) >= 4 for s in set(suits))
    if flush_draw:
        return True
    ranks = sorted(_RANK_ORDER.get(c[0].upper(), 0) for c in all_cards)
    for i in range(len(ranks) - 3):
        window = ranks[i : i + 4]
        if window[-1] - window[0] == 3 and len(set(window)) == 4:
            return True
    return False


def _count_outs(hole: list[str], board: list[str]) -> int:
    """Estimate outs for a drawing hand using simple flush/straight out counts."""
    all_cards = hole + board
    suits = [c[1].lower() for c in all_cards]
    ranks = [_RANK_ORDER.get(c[0].upper(), 0) for c in all_cards]

    outs = 0
    for s in set(suits):
        if suits.count(s) == 4:
            outs += 9
    for r_val in range(13):
        if r_val not in ranks:
            span = [r for r in ranks if abs(r - r_val) <= 2]
            if len(span) >= 3:
                outs += 4
    return min(outs, 15)


class ChenBot:
    """Chen-formula preflop + treys/pot-odds postflop baseline.

    Preflop: Chen formula score thresholds drive raise/call/fold.
    Postflop: treys made-hand classification + Rule-of-2/4 pot odds.

    Always returns a legal action via pick_first_legal (T-9-16 mitigation).
    """

    name = "chen-bot"
    description = "Chen formula preflop + treys/pot-odds postflop"

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._raise_threshold: float = 10.0
        self._call_threshold: float = 7.0
        self._was_preflop_aggressor: bool = False

    def decide(self, state: dict[str, Any]) -> int:
        """Return a legal RLCard action int (0-4) for the given env state.

        Args:
            state: RLCard per-player state dict with ``'hand'``, ``'public_cards'``,
                and ``'legal_actions'`` keys.

        Returns:
            int in ``{0, 1, 2, 3, 4}`` (FOLD/CHECK_CALL/HALF_POT/POT/ALL_IN).
        """
        raw = state.get("raw_obs", state) if "hand" not in state else state
        hand = raw.get("hand", [])
        public_cards = raw.get("public_cards", [])
        is_preflop = len(public_cards) == 0

        if is_preflop:
            return self._decide_preflop(state, hand)
        return self._decide_postflop(state, hand, public_cards)

    def _decide_preflop(self, state: dict[str, Any], hand: list[str]) -> int:
        if len(hand) < 2:
            return pick_first_legal(state, desired=(CHECK_CALL, FOLD))

        score = chen_score(hand)
        if score >= self._raise_threshold:
            self._was_preflop_aggressor = True
            return pick_first_legal(state, desired=(HALF_POT, POT, CHECK_CALL, FOLD))
        if score >= self._call_threshold:
            self._was_preflop_aggressor = False
            return pick_first_legal(state, desired=(CHECK_CALL, FOLD))
        self._was_preflop_aggressor = False
        return pick_first_legal(state, desired=(FOLD, CHECK_CALL))

    def _decide_postflop(self, state: dict[str, Any], hand: list[str], public_cards: list[str]) -> int:
        if len(hand) < 2 or len(public_cards) < 3:
            return pick_first_legal(state, desired=(CHECK_CALL, FOLD))

        hand_class = postflop_hand_class(hand, public_cards)

        if hand_class == "strong":
            if self._was_preflop_aggressor:
                return pick_first_legal(state, desired=(HALF_POT, POT, CHECK_CALL, FOLD))
            return pick_first_legal(state, desired=(CHECK_CALL, HALF_POT, FOLD))

        if hand_class == "medium":
            return pick_first_legal(state, desired=(CHECK_CALL, FOLD))

        if hand_class == "draw":
            outs = _count_outs(hand, public_cards)
            streets_remaining = 2 if len(public_cards) == 3 else 1
            equity = outs * (4 if streets_remaining == 2 else 2) / 100.0
            # Conservative pot odds estimate: assume ~33% pot odds facing a bet
            pot_odds = 0.33
            if equity > pot_odds + 0.05:
                return pick_first_legal(state, desired=(CHECK_CALL, FOLD))
            return pick_first_legal(state, desired=(FOLD, CHECK_CALL))

        # "weak" / air — fold if possible
        return pick_first_legal(state, desired=(FOLD, CHECK_CALL))
