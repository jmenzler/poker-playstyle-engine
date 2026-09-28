"""Random opponent — uniform over legal actions. Sanity-floor baseline.

The engine MUST beat this. If it doesn't, the engine is fundamentally broken.
Determinism is provided by a seeded numpy RNG (T-06-11 mitigation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.eval.strategy import coerce_action


class RandomStrategy:
    """Uniform random selection over legal actions, seeded for reproducibility."""

    name = "random"
    description = "uniform random over legal actions"

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def decide(self, state: dict[str, Any]) -> int:
        legal = [coerce_action(a) for a in state.get("legal_actions", [])]
        if not legal:
            raise RuntimeError("RandomStrategy: no legal actions")
        return int(self._rng.choice(legal))
