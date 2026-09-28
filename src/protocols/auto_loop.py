"""Protocol for the autonomous improvement loop (implemented in Phase 5)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AutoLoop(Protocol):
    """Runs one iteration of the auto-improvement loop.

    Implemented in Phase 5 after research milestone M4 locks the algorithm.
    Returns the count of patches applied in this step.
    """

    def step(self) -> int: ...
