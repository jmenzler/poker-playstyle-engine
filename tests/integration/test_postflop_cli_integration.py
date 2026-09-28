"""PC-only integration smoke test for the postflop-cli subprocess wrapper.

Requires the postflop-cli binary on the PC at either:
  - $POSTFLOP_CLI_BIN env var, or
  - ~/postflop-cli/target/release/postflop-cli (default path)

Test is skipped when binary is absent (Mac dev environment, CI, etc.).

Phase 6 contract: solve() parses solve_mode JSON (actions_root +
aggregate_freq_root) and returns a SolverResult whose action_dist sums to
1.0 ± 0.001. NotImplementedError is no longer raised. The Phase 4/5
NotImplementedError contract was retired in plan 06-02 (OQ-1 resolved).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_postflop_cli_real_binary_smoke():
    """PC-only: end-to-end roundtrip on a real postflop-cli binary.

    Uses SOLVER.md §Spot 3 benchmark values:
        Flop Ah7c2d, BTN vs BB, pot=56, effective_stack=172

    Phase 6 assertions (post-OQ-1):
        - is_available() returns True.
        - solve() returns a SolverResult.
        - solve_time_ms > 0 (binary actually ran).
        - exploitability_pct is a non-negative percentage.
        - action_dist sums to 1.0 ± 0.001 (canonical-vocab probabilities).
    """
    from src.solver.postflop_cli import PostflopCliBackend, SolverSpot

    binary = Path(
        os.environ.get(
            "POSTFLOP_CLI_BIN",
            "~/postflop-cli/target/release/postflop-cli",
        )
    ).expanduser()

    if not binary.exists():
        pytest.skip(f"postflop-cli binary not found: {binary} — run on PC")

    backend = PostflopCliBackend(binary_path=binary)
    assert backend.is_available() is True, f"is_available() returned False for existing path {binary}"

    spot = SolverSpot(
        pot=56,
        effective_stack=172,
        board=["Ah", "7c", "2d"],
        range_ip="AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,KQo",
        range_oop="AA:0.5,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,KQo,K9s,K8s,K7s,K6s",
        max_iterations=10,
        target_exploitability_pct=5.0,
    )

    # Phase 6: OQ-1 resolved — solve() must return a populated SolverResult.
    result = backend.solve(spot, timeout_s=120.0)
    assert result.solve_time_ms > 0
    assert result.exploitability_pct >= 0.0
    total = sum(result.action_dist.values())
    assert abs(total - 1.0) <= 0.001, f"action_dist sum {total} not within 1.0 ± 0.001"
