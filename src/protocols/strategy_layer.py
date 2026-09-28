"""Protocol for the strategy retrieval layer (implemented in Phase 3)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StrategyLayer(Protocol):
    """Retrieves k nearest StrategyNodes for a given cluster key.

    Implemented via Milvus ANN search in Phase 3. Returns raw dicts
    rather than typed StrategyNode to avoid circular imports at this stage.
    """

    def retrieve(self, cluster_key: str, k: int) -> list[Any]: ...
