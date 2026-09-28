"""Regression-test baseline — loads a prior engine snapshot for head-to-head comparison.

v1 STATUS: PLACEHOLDER (T-06-13 mitigation).
The snapshot mechanism (a ``snapshot_tag`` column on ``strategy_nodes`` plus a
``poker-engine snapshot create <tag>`` CLI) is deferred to v2 per Phase 6
RESEARCH.md Open Question 4 (OQ-4).

This module exists for API completeness so the Eval REGISTRY exposes 7 entries;
instantiation raises ``NotImplementedError`` with a pointer to the v2 backlog.

v2 work (out of scope for Phase 6):
- ``migrations/0XX_snapshot_strategy_nodes.sql`` — add ``snapshot_tag`` column
- CLI: ``poker-engine snapshot create <tag>`` — write tagged frozen rows
- This module: load tagged snapshot via ``SELECT ... WHERE snapshot_tag = %s``

For regression testing today, manually checkout the prior commit and run the
existing ``sim_ab`` harness from Phase 5 instead.
"""

from __future__ import annotations

from typing import Any


class PrevEngineStrategy:
    """v2 placeholder — raises NotImplementedError at construction time."""

    name = "prev-engine"
    description = "snapshot of prior engine version (regression test — v2)"

    def __init__(self, version: str = "v0.0.0", seed: int = 42) -> None:
        _ = version, seed
        raise NotImplementedError(
            "PrevEngineStrategy requires the snapshot mechanism (OQ-4), deferred to v2. "
            "See Phase 6 RESEARCH.md Open Question 4. To regression-test today, manually "
            "checkout the prior commit and run sim_ab from Phase 5 instead."
        )

    def decide(self, state: dict[str, Any]) -> int:  # pragma: no cover — unreachable
        raise NotImplementedError("see __init__ for OQ-4 v2 deferral notice")
