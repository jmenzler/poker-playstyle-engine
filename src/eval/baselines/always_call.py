"""Passive baseline — call any bet; check when no bet faces; fold when CHECK_CALL is illegal.

Used as a "doormat" floor opponent. The engine should crush this.
"""

from __future__ import annotations

from typing import Any

from src.eval.strategy import CHECK_CALL, FOLD, pick_first_legal


class AlwaysCallStrategy:
    """Prefer CHECK_CALL; fall back to FOLD when CHECK_CALL is illegal."""

    name = "always-call"
    description = "call any bet ≤ pot; check otherwise"

    def __init__(self, seed: int = 42) -> None:
        # ``seed`` accepted for API consistency; this strategy is fully deterministic.
        _ = seed

    def decide(self, state: dict[str, Any]) -> int:
        return pick_first_legal(state, desired=(CHECK_CALL, FOLD))
