"""src/solver — postflop-cli subprocess wrapper (Phase 4 stub; invoked in Phase 5).

Public exports:
    PostflopCliBackend  — SolverBackend implementation wrapping the postflop-cli binary
    SolverSpot          — input dataclass for a spot to solve
    SolverResult        — output dataclass carrying solved action distribution
"""

from __future__ import annotations

from src.solver.postflop_cli import PostflopCliBackend, SolverResult, SolverSpot

__all__ = ["PostflopCliBackend", "SolverResult", "SolverSpot"]
