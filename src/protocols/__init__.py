"""Protocol interfaces for the poker-engine's four swappable layers.

Re-exports the canonical GameState type alongside all four Protocol interfaces.
Downstream code should import from this package, not from submodules directly.

Implementations:
- GameStateSource: Phase 3 (SimAdapter wrapping RLCard)
- DecisionEngine: Phase 3 (ANN retrieval + action blending)
- StrategyLayer: Phase 3 (Milvus ANN search)
- AutoLoop: Phase 5 (solver-distillation improvement loop)
"""

from src.protocols.auto_loop import AutoLoop
from src.protocols.decision_engine import DecisionEngine
from src.protocols.game_state import GameState, Position, Street
from src.protocols.game_state_source import GameStateSource
from src.protocols.strategy_layer import StrategyLayer

__all__ = [
    "AutoLoop",
    "DecisionEngine",
    "GameState",
    "GameStateSource",
    "Position",
    "StrategyLayer",
    "Street",
]
