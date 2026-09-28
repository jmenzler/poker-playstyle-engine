"""Pure sparseness-flagging helpers for KNNDecisionEngine (Phase 4 Plan 02).

Pure: no I/O, no global state.

Threshold constants are placeholders — calibrated in Phase 4.x once real
session distance distributions are observable via tools/calibrate_sparseness.py.
"""

from __future__ import annotations

from typing import Any

# TBD — calibrated in Phase 4.x; placeholder for now
SPARSE_DIST_TAU: float = 0.5

# TBD — calibrated in Phase 4.x; placeholder for now
SPARSE_N_MIN: int = 3


def compute_flagged_sparse(
    neighbors: list[dict[str, Any]],
    *,
    tau: float = SPARSE_DIST_TAU,
    n_min: int = SPARSE_N_MIN,
) -> bool:
    """Return True when the kNN neighborhood is considered sparse.

    A neighborhood is sparse if:
        (len(neighbors) < n_min) OR (max_distance > tau)

    where max_distance is the maximum COSINE distance across all neighbors.
    The tau comparison is strict (>), so max_distance == tau is NOT flagged.

    Args:
        neighbors: list of pymilvus search-result hit dicts; each hit has a
            'distance' float key (COSINE distance). Empty list is always sparse.
        tau: distance threshold above which the neighborhood is sparse.
            Defaults to SPARSE_DIST_TAU placeholder constant.
        n_min: minimum neighbor count required for a non-sparse neighborhood.
            Defaults to SPARSE_N_MIN placeholder constant.

    Returns:
        True if sparse, False otherwise.
    """
    if len(neighbors) < n_min:
        return True
    max_dist = _max_distance(neighbors)
    return max_dist > tau


def compute_max_neighbor_distance(neighbors: list[dict[str, Any]]) -> float | None:
    """Return the maximum COSINE distance among retrieved neighbors.

    Args:
        neighbors: list of pymilvus search-result hit dicts; each hit has a
            'distance' float key. Falls back to 1.0 if 'distance' key is absent.

    Returns:
        Maximum distance as float, or None when neighbors is empty.
    """
    if not neighbors:
        return None
    return _max_distance(neighbors)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _max_distance(neighbors: list[dict[str, Any]]) -> float:
    """Extract and return maximum distance from a non-empty neighbor list.

    Extracted via hit.get('distance', 1.0) so that hits missing the key
    (rare pymilvus version quirks) are treated as maximum distance (1.0).
    """
    return max(float(hit.get("distance", 1.0)) for hit in neighbors)
