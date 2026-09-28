"""Phase 4 Plan 02: unit tests for src.decision_engine.sparseness pure functions.

Tests cover all boundary conditions for compute_flagged_sparse and
compute_max_neighbor_distance before the implementation module exists (TDD RED).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# compute_flagged_sparse
# ---------------------------------------------------------------------------


def test_flagged_sparse_empty_neighbors():
    """Empty neighbor list -> flagged because n < SPARSE_N_MIN (zero neighbors)."""
    from src.decision_engine.sparseness import SPARSE_N_MIN, compute_flagged_sparse

    result = compute_flagged_sparse([], tau=0.5, n_min=SPARSE_N_MIN)
    assert result is True, "Empty neighbors must be flagged sparse (n=0 < n_min)"


def test_flagged_sparse_n_below_min():
    """n < n_min triggers flag even if distances are low."""
    from src.decision_engine.sparseness import compute_flagged_sparse

    # 2 neighbors with very low distances — n_min=3, so still flagged
    neighbors = [
        {"distance": 0.1, "entity": {}},
        {"distance": 0.05, "entity": {}},
    ]
    result = compute_flagged_sparse(neighbors, tau=0.5, n_min=3)
    assert result is True, "n=2 < n_min=3 must be flagged regardless of distances"


def test_flagged_sparse_max_distance_above_tau():
    """n >= n_min but max distance > tau -> flagged."""
    from src.decision_engine.sparseness import compute_flagged_sparse

    neighbors = [
        {"distance": 0.2, "entity": {}},
        {"distance": 0.3, "entity": {}},
        {"distance": 0.6, "entity": {}},  # > tau=0.5
    ]
    result = compute_flagged_sparse(neighbors, tau=0.5, n_min=3)
    assert result is True, "max_distance=0.6 > tau=0.5 must be flagged"


def test_flagged_sparse_not_flagged():
    """n >= n_min AND max distance <= tau -> not flagged."""
    from src.decision_engine.sparseness import compute_flagged_sparse

    neighbors = [
        {"distance": 0.1, "entity": {}},
        {"distance": 0.2, "entity": {}},
        {"distance": 0.4, "entity": {}},  # <= tau=0.5
    ]
    result = compute_flagged_sparse(neighbors, tau=0.5, n_min=3)
    assert result is False, "max_distance=0.4 <= tau=0.5 and n=3 >= n_min=3 must NOT be flagged"


def test_flagged_sparse_exactly_tau_not_flagged():
    """Boundary: max_distance == tau -> NOT flagged (flag uses strict > not >=)."""
    from src.decision_engine.sparseness import compute_flagged_sparse

    neighbors = [
        {"distance": 0.3, "entity": {}},
        {"distance": 0.3, "entity": {}},
        {"distance": 0.5, "entity": {}},  # == tau exactly
    ]
    result = compute_flagged_sparse(neighbors, tau=0.5, n_min=3)
    assert result is False, "max_distance == tau (boundary) must NOT be flagged (strict >)"


# ---------------------------------------------------------------------------
# compute_max_neighbor_distance
# ---------------------------------------------------------------------------


def test_max_neighbor_distance_empty():
    """Empty neighbors -> None (no distances exist)."""
    from src.decision_engine.sparseness import compute_max_neighbor_distance

    result = compute_max_neighbor_distance([])
    assert result is None, "Empty neighbors must return None"


def test_max_neighbor_distance_single():
    """Single neighbor -> returns that neighbor's distance."""
    from src.decision_engine.sparseness import compute_max_neighbor_distance

    neighbors = [{"distance": 0.37, "entity": {}}]
    result = compute_max_neighbor_distance(neighbors)
    assert result == pytest.approx(0.37), "Single neighbor must return its distance"


def test_max_neighbor_distance_multiple():
    """Multiple neighbors -> returns the maximum distance."""
    from src.decision_engine.sparseness import compute_max_neighbor_distance

    neighbors = [
        {"distance": 0.1, "entity": {}},
        {"distance": 0.6, "entity": {}},
        {"distance": 0.3, "entity": {}},
    ]
    result = compute_max_neighbor_distance(neighbors)
    assert result == pytest.approx(0.6), "Multiple neighbors must return maximum distance"


def test_max_neighbor_distance_uses_distance_key():
    """Distance extracted via h.get('distance', 1.0) fallback if key missing."""
    from src.decision_engine.sparseness import compute_max_neighbor_distance

    # One hit with distance key, one without (falls back to 1.0)
    neighbors = [
        {"distance": 0.2, "entity": {}},
        {"entity": {}},  # missing 'distance' -> fallback 1.0
    ]
    result = compute_max_neighbor_distance(neighbors)
    assert result == pytest.approx(1.0), "Missing 'distance' key must fall back to 1.0"
