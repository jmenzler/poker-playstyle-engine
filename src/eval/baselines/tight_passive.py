"""TAG-style nit — top-12% RFI preflop; fold to most aggression postflop.

v1 simplification: uses a hole-card rank heuristic (broadways + pocket pairs)
rather than the full preflop equity table. Calibrating the actual top-12% RFI
range against an equity table is a v2 follow-up if Eval results suggest the
baseline is too far off realistic nit play.
"""

from __future__ import annotations

from typing import Any

from src.eval.strategy import (
    CHECK_CALL,
    FOLD,
    HALF_POT,
    coerce_action,
    pick_first_legal,
)

# Approximate top-12% bucket: both hole cards are broadway or higher.
# This is a coarse proxy for the canonical nit opening range.
_PREMIUM_RANKS: frozenset[str] = frozenset({"A", "K", "Q", "J", "T"})


class TightPassiveStrategy:
    """Tight preflop, passive postflop: fold to bets, check when free."""

    name = "tight-passive"
    description = "top 12% RFI; fold to most postflop aggression"

    def __init__(self, seed: int = 42) -> None:
        # Currently deterministic — seed accepted for future randomisation (e.g. mix-in calls).
        _ = seed

    def decide(self, state: dict[str, Any]) -> int:
        # hand/public_cards live under raw_obs in RLCard env state; legal_actions is top-level.
        raw = state.get("raw_obs", state) if "hand" not in state else state
        legal = {coerce_action(a) for a in state.get("legal_actions", [])}
        is_preflop = len(raw.get("public_cards", [])) == 0
        hand = raw.get("hand", [])
        is_premium = bool(hand) and all(
            isinstance(card, str) and card[:1] in _PREMIUM_RANKS for card in hand[:2]
        )

        if is_preflop:
            if is_premium:
                # Open small with premiums; if raising not legal, limp or fold.
                return pick_first_legal(state, desired=(HALF_POT, CHECK_CALL, FOLD))
            # Non-premium preflop: limp if free, otherwise fold.
            return pick_first_legal(state, desired=(CHECK_CALL, FOLD))

        # Postflop: take CHECK_CALL when it's free (no bet faces us),
        # otherwise prefer folding.
        if CHECK_CALL in legal:
            return CHECK_CALL
        return pick_first_legal(state, desired=(FOLD, CHECK_CALL))
