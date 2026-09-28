"""Unit tests for empirical-Bayes shrinkage estimator (CLI-04, D-06).

Plan: 06-02. Pure-numpy closed-form normal-normal estimator with method-of-moments
hyperparameters; see src/study/eb_shrinkage.py + RESEARCH.md Pattern 5.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.study.eb_shrinkage import ci_95, is_shrunk, normal_normal_eb


def test_k1_returns_input_unchanged() -> None:
    """K=1: cannot estimate τ²; return sample mean unchanged, tau2_hat=0."""
    xbar = np.array([0.42])
    n = np.array([100])
    post_mean, _post_var, mu_hat, tau2 = normal_normal_eb(xbar, n, sigma2=0.05)
    np.testing.assert_array_equal(post_mean, xbar)
    assert tau2 == 0.0
    assert mu_hat == pytest.approx(0.42)


def test_all_identical_xbar_returns_grand_mean() -> None:
    """All xbar identical → S²=0 → tau2_hat=0 → full shrinkage to grand mean."""
    xbar = np.full(10, 0.42)
    n = np.full(10, 50)
    post_mean, _post_var, mu_hat, tau2 = normal_normal_eb(xbar, n, sigma2=0.05)
    assert tau2 == 0.0
    np.testing.assert_allclose(post_mean, 0.42)
    assert mu_hat == pytest.approx(0.42)


def test_low_n_shrinks_more_than_high_n() -> None:
    """Low-n clusters should land closer to the grand mean than high-n clusters."""
    rng = np.random.default_rng(0)
    xbar = rng.normal(0.5, 0.2, size=10)
    n = np.array([5, 200, 5, 200, 5, 200, 5, 200, 5, 200], dtype=float)
    post_mean, _post_var, mu_hat, _tau2 = normal_normal_eb(xbar, n, sigma2=0.05)
    d_low = np.mean(np.abs(post_mean[::2] - mu_hat))
    d_high = np.mean(np.abs(post_mean[1::2] - mu_hat))
    assert d_low < d_high, f"low-n shrinkage failed: d_low={d_low}, d_high={d_high}"


def test_ci_95_zero_variance() -> None:
    """post_var == 0 → degenerate CI [mean, mean]."""
    pm = np.array([0.42])
    pv = np.array([0.0])
    lo, hi = ci_95(pm, pv)
    assert lo[0] == 0.42
    assert hi[0] == 0.42


def test_ci_95_normal_variance() -> None:
    """ci_95 = mean ± 1.96·sqrt(var)."""
    pm = np.array([1.0])
    pv = np.array([0.25])
    lo, hi = ci_95(pm, pv)
    np.testing.assert_allclose(lo[0], 1.0 - 1.96 * 0.5)
    np.testing.assert_allclose(hi[0], 1.0 + 1.96 * 0.5)


def test_is_shrunk_degenerate_tau2_zero() -> None:
    """tau2_hat == 0 → is_shrunk is True regardless of n (full shrinkage)."""
    assert is_shrunk(n=100, tau2_hat=0.0, sigma2=0.05) is True
    assert is_shrunk(n=5, tau2_hat=0.0, sigma2=0.05) is True


def test_is_shrunk_low_n_true_high_n_false() -> None:
    """Low-n cluster carries shrinkage; high-n cluster does not (threshold=0.1)."""
    # tau2_hat=0.05, sigma2=0.05; B = v/(v+0.05) where v=sigma2/n.
    # n=2: v=0.025 → B = 0.025/0.075 = 0.333 > 0.1 → True
    # n=200: v=0.00025 → B = 0.00025/0.05025 ≈ 0.005 < 0.1 → False
    assert is_shrunk(n=2, tau2_hat=0.05, sigma2=0.05) is True
    assert is_shrunk(n=200, tau2_hat=0.05, sigma2=0.05) is False


@given(
    k=st.integers(min_value=2, max_value=50),
    sigma2=st.floats(min_value=0.001, max_value=1.0),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=50, deadline=None)
def test_property_post_mean_bounded(k: int, sigma2: float, seed: int) -> None:
    """post_mean[i] is a convex combination of xbar_i and mu_hat (∈ [min(xbar), max(xbar)])."""
    rng = np.random.default_rng(seed)
    xbar = rng.normal(0.5, 0.2, size=k)
    n = rng.integers(10, 500, size=k).astype(float)
    post_mean, _post_var, _mu_hat, _tau2 = normal_normal_eb(xbar, n, sigma2=sigma2)
    assert np.all(post_mean >= xbar.min() - 1e-9)
    assert np.all(post_mean <= xbar.max() + 1e-9)


def test_returns_four_tuple() -> None:
    """Contract: returns (post_mean, post_var, mu_hat, tau2_hat)."""
    xbar = np.array([0.2, 0.5, 0.8])
    n = np.array([50, 100, 150], dtype=float)
    result = normal_normal_eb(xbar, n, sigma2=0.05)
    assert len(result) == 4
    post_mean, post_var, mu_hat, tau2 = result
    assert isinstance(post_mean, np.ndarray)
    assert isinstance(post_var, np.ndarray)
    assert isinstance(mu_hat, float)
    assert isinstance(tau2, float)
    assert post_mean.shape == xbar.shape
    assert post_var.shape == xbar.shape


def test_pytest_runs() -> None:
    """Trivial canary — confirms test file is discoverable."""
    assert True
