"""Tests for z-score -> group-weight application order invariant (CONTEXT.md Decision 1).

Activated by Phase 2 Plan 06. Unit tests (no Milvus required).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# No pytestmark = pytest.mark.integration — these are pure unit tests (no live Milvus needed).


def test_apply_order_z_then_weights(synthetic_postflop_embedding, expected_postflop_group_weights) -> None:
    """CONTEXT.md Decision 1: z = (x-mu)/sigma THEN weighted = z * group_weights.

    With mean=0 and std=1 (identity z-score), result must equal vec * weights elementwise.
    """
    from tools.upsert_milvus import normalize

    vec = synthetic_postflop_embedding.astype(np.float64)
    weights = expected_postflop_group_weights
    mean = np.zeros(80, dtype=np.float64)
    std = np.ones(80, dtype=np.float64)

    result = normalize(vec, mean, std, weights)

    expected = vec * weights
    np.testing.assert_allclose(
        result, expected, atol=1e-6, err_msg="normalize() must apply z-score THEN weights"
    )


def test_extractor_output_is_unweighted(sample_postflop_dp) -> None:
    """Plan 02 invariant: extract_postflop returns vec where group weights are NOT pre-applied.

    Property assertion: the raw extractor output should be in [0, 1] range (min-maxed).
    If weights were pre-applied, the effective values would exceed [0, 1] for groups with weight > 1.
    We verify that the mean of the returned vector is within [0, 1], consistent with unweighted
    min-maxed values.
    """
    from unittest.mock import MagicMock

    from tools.feature_extractors.postflop import _GROUP_WEIGHTS, extract_postflop

    # extract_postflop requires an EquityLookup; mock it to return a deterministic equity dict.
    # Values in [0,1] are realistic (equity decile percentages).
    # Group A uses: p10..p90 (every 10), variance, mean, p90 (best-case).
    equity_dict = {f"p{p}": 0.5 for p in range(10, 100, 10)}
    equity_dict["mean"] = 0.5
    equity_dict["variance"] = 0.02  # realistic variance value in [0, 0.25]
    equity_dict["p90"] = 0.7  # best-case equity
    mock_eq = MagicMock()
    mock_eq.get.return_value = equity_dict

    result = extract_postflop(sample_postflop_dp, equity_lookup=mock_eq)
    # extract_postflop returns (vec, meta) tuple
    vec = result[0] if isinstance(result, tuple) else result
    assert isinstance(vec, np.ndarray), "extract_postflop must return np.ndarray as first element"
    assert vec.shape == (80,), f"Expected 80-dim vec, got {vec.shape}"

    # All dimensions should be in [0, 1] for an unweighted min-maxed vector
    assert vec.min() >= -0.01, (
        f"Extractor returned negative values (min={vec.min():.4f}) — unexpected for unweighted vec"
    )
    assert vec.max() <= 1.01, (
        f"Extractor returned values > 1 (max={vec.max():.4f}) — weights may be pre-applied"
    )

    # Confirm the returned vec is NOT identical to vec * canonical weights for any group with weight != 1.0
    # Build the canonical weight vector
    _POSTFLOP_GROUP_DIM_COUNTS = [
        ("A", 12),
        ("B", 8),
        ("C", 6),
        ("D", 8),
        ("E", 7),
        ("F", 16),
        ("G", 6),
        ("H", 7),
        ("J", 10),
    ]
    parts = []
    for grp, n in _POSTFLOP_GROUP_DIM_COUNTS:
        parts.extend([_GROUP_WEIGHTS[grp]] * n)
    weights = np.array(parts, dtype=np.float64)

    weighted = vec * weights
    # If weights were pre-applied, vec would not be in [0,1] for dims with weight > 1.
    # As an additional check: the two vectors must differ where weight != 1.0
    non_unit_weight_dims = np.where(weights != 1.0)[0]
    if len(non_unit_weight_dims) > 0 and not np.allclose(vec[non_unit_weight_dims], 0.0):
        # vec[dim] != weighted[dim] for non-unit-weight dims (when vec is non-zero)
        non_equal = ~np.isclose(vec[non_unit_weight_dims], weighted[non_unit_weight_dims], atol=1e-8)
        assert non_equal.any(), (
            "Extractor output looks pre-weighted: vec equals vec*weights for non-unit-weight dims. "
            "This violates Plan 02 invariant."
        )


def test_constant_dim_z_score_is_zero(synthetic_postflop_embedding) -> None:
    """std=0 -> manifest std=1.0 (zero-std guard from Plan 05) -> z=(x-mu)/1=0 for constant dim."""
    from tools.upsert_milvus import normalize

    vec = synthetic_postflop_embedding.astype(np.float64)
    dim_idx = 17  # arbitrary test dimension

    # Build manifest: mean[17] = vec[17] (so z[17] = (x-mu)/std = (x-x)/1 = 0)
    mean = np.zeros(80, dtype=np.float64)
    mean[dim_idx] = vec[dim_idx]

    # std[17] = 1.0 (zero-std guard output from zscore_fit.py)
    std = np.ones(80, dtype=np.float64)

    weights = np.ones(80, dtype=np.float64)

    result = normalize(vec, mean, std, weights)

    assert result[dim_idx] == pytest.approx(0.0, abs=1e-9), (
        f"Expected z=0 for constant dim {dim_idx} (mean=vec[{dim_idx}], std=1), got {result[dim_idx]}"
    )
