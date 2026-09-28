"""TAG (Tight-Aggressive) profile — solid ~18% RFI, balanced 3-bet, smart cbet.

v1 heuristic:
- Preflop: premium hands raise 80% / call 20%; non-premium fold 70% / call 30%.
- Postflop: 50% CHECK_CALL, 30% bet small (HALF_POT), 20% fold.

The same _PREMIUM_RANKS proxy as ``tight_passive.py`` is used; v2 follow-up
would calibrate the range against the project's preflop equity table.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.eval.strategy import (
    CHECK_CALL,
    FOLD,
    HALF_POT,
    POT,
    pick_first_legal,
)

_PREMIUM_RANKS: frozenset[str] = frozenset({"A", "K", "Q", "J", "T"})


class TAGStrategy:
    """Tight-aggressive heuristic. Balanced raise/call/fold proportions."""

    name = "TAG-profile"
    description = "solid 18% RFI, balanced 3-bet, smart cbet"

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def decide(self, state: dict[str, Any]) -> int:
        # hand/public_cards live under raw_obs in RLCard env state; legal_actions is top-level.
        raw = state.get("raw_obs", state) if "hand" not in state else state
        hand = raw.get("hand", [])
        is_premium = bool(hand) and all(
            isinstance(card, str) and card[:1] in _PREMIUM_RANKS for card in hand[:2]
        )
        is_preflop = len(raw.get("public_cards", [])) == 0

        if is_preflop:
            if is_premium:
                # Raise premiums 80%; call 20%.
                if self._rng.random() < 0.80:
                    return pick_first_legal(state, desired=(HALF_POT, POT, CHECK_CALL, FOLD))
                return pick_first_legal(state, desired=(CHECK_CALL, FOLD))
            # Non-premium: fold 70%, call 30%.
            if self._rng.random() < 0.70:
                return pick_first_legal(state, desired=(FOLD, CHECK_CALL))
            return pick_first_legal(state, desired=(CHECK_CALL, FOLD))

        # Postflop: balanced — 50% CHECK_CALL, 30% bet small, 20% fold.
        r = self._rng.random()
        if r < 0.50:
            return pick_first_legal(state, desired=(CHECK_CALL, FOLD))
        if r < 0.80:
            return pick_first_legal(state, desired=(HALF_POT, CHECK_CALL, FOLD))
        return pick_first_legal(state, desired=(FOLD, CHECK_CALL))
