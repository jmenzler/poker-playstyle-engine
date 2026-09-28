"""Phase 9 / TVD — _total_variation_distance function (unit)."""

from __future__ import annotations

import pytest


def test_manual_vs_computed():
    """TVD matches hand calculation for known inputs.

    p={"check":0.6,"fold":0.4}, q={"check":0.5,"fold":0.5}
    TVD = 0.5 * (|0.6-0.5| + |0.4-0.5|) = 0.5 * 0.2 = 0.1
    """
    from src.metrics.ev_loss import _total_variation_distance

    p = {"check": 0.6, "fold": 0.4}
    q = {"check": 0.5, "fold": 0.5}
    assert _total_variation_distance(p, q) == pytest.approx(0.1, abs=1e-9)


def test_identical_distributions_zero():
    """TVD(p, p) == 0.0 for identical distributions."""
    from src.metrics.ev_loss import _total_variation_distance

    p = {"check": 0.6, "fold": 0.4}
    assert _total_variation_distance(p, p) == pytest.approx(0.0, abs=1e-9)


def test_disjoint_distributions_one():
    """TVD == 1.0 when distributions are fully disjoint (max distance)."""
    from src.metrics.ev_loss import _total_variation_distance

    p = {"check": 1.0}
    q = {"fold": 1.0}
    assert _total_variation_distance(p, q) == pytest.approx(1.0, abs=1e-9)


def test_partial_vocab_missing_keys_treated_as_zero():
    """Missing keys in either dict are treated as 0.0 (union vocab handling).

    Engine emits a subset, solver emits all 15 actions. Missing keys must
    contribute their full probability as mass difference.
    """
    from src.metrics.ev_loss import _total_variation_distance

    p = {"check": 0.7, "fold": 0.3}
    q = {"check": 0.5, "fold": 0.3, "call": 0.2}
    # |0.7-0.5| + |0.3-0.3| + |0.0-0.2| = 0.2 + 0.0 + 0.2 = 0.4; TVD = 0.2
    assert _total_variation_distance(p, q) == pytest.approx(0.2, abs=1e-9)


def test_bounded_zero_to_one():
    """TVD result is always in [0.0, 1.0]."""
    from src.metrics.ev_loss import _total_variation_distance

    p = {"check": 0.5, "fold": 0.3, "call": 0.2}
    q = {"bet_50": 0.4, "raise_pot": 0.6}
    result = _total_variation_distance(p, q)
    assert 0.0 <= result <= 1.0
