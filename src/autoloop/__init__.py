"""Phase 5 auto-loop package.

Re-exports:
    AutoLoopDriver — implements AutoLoop protocol; step() runs one full cycle.
    ClusterCandidate — msgspec.Struct representing a patch candidate.
    select_patch_candidates — reads metrics rows, applies filters, returns ranked candidates.
    generate_candidate_action_dist — kNN-rebalance: returns (action_dist, embedding) tuple.
"""

from src.autoloop.driver import AutoLoopDriver
from src.autoloop.leak_detector import ClusterCandidate, select_patch_candidates
from src.autoloop.rebalance import generate_candidate_action_dist

__all__ = [
    "AutoLoopDriver",
    "ClusterCandidate",
    "generate_candidate_action_dist",
    "select_patch_candidates",
]
