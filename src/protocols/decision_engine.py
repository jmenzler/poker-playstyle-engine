"""Protocol for the decision engine (implemented in Phase 3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.protocols.game_state import GameState


@runtime_checkable
class DecisionEngine(Protocol):
    """Returns one of the canonical actions (CANONICAL_ACTIONS in
    src.decision_engine.blending) for a given decision point.

    Implemented by the ANN-retrieval + blending engine in Phase 3.
    p99 latency target: <50 ms in sim path.
    """

    def decide(self, gs: GameState) -> str: ...
