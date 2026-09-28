"""Protocol for game state sources (implemented in Phase 3 SimAdapter)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.protocols.game_state import GameState


@runtime_checkable
class GameStateSource(Protocol):
    """Supplies a sequence of decision points to the engine.

    Implemented by RLCard SimAdapter in Phase 3 and by replay sources.
    """

    def next_game_state(self) -> GameState: ...

    def has_more(self) -> bool: ...
