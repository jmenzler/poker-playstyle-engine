"""LAG (Loose-Aggressive) profile — wide opens, frequent raises, aggressive cbet.

v1 heuristic: ~70% aggressive (HALF_POT/POT/ALL_IN/CHECK_CALL), ~20% CHECK_CALL,
~10% FOLD. Seed-driven so a fixed seed produces a reproducible session.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.eval.strategy import (
    ALL_IN,
    CHECK_CALL,
    FOLD,
    HALF_POT,
    POT,
    pick_first_legal,
)


class LAGStrategy:
    """Loose-aggressive heuristic. Skews heavily toward raising."""

    name = "LAG-profile"
    description = "wide opens, frequent 3-bets, aggressive cbet"

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def decide(self, state: dict[str, Any]) -> int:
        r = self._rng.random()
        if r < 0.70:
            return pick_first_legal(state, desired=(HALF_POT, POT, ALL_IN, CHECK_CALL, FOLD))
        if r < 0.90:
            return pick_first_legal(state, desired=(CHECK_CALL, HALF_POT, FOLD))
        return pick_first_legal(state, desired=(FOLD, CHECK_CALL))
