"""Aggressive baseline — min-raise everything; only call when no raise is legal.

Used to stress-test the engine's defence against constant pressure.
"""

from __future__ import annotations

from typing import Any

from src.eval.strategy import ALL_IN, CHECK_CALL, HALF_POT, POT, pick_first_legal


class AlwaysRaiseStrategy:
    """Prefer HALF_POT, then POT, then ALL_IN, then CHECK_CALL."""

    name = "always-raise"
    description = "min-raise everything; never call unless forced"

    def __init__(self, seed: int = 42) -> None:
        _ = seed

    def decide(self, state: dict[str, Any]) -> int:
        return pick_first_legal(state, desired=(HALF_POT, POT, ALL_IN, CHECK_CALL))
