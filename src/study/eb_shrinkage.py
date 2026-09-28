"""Empirical Bayes shrinkage for Stage A leak ranking (CLI-04, D-06).

Closed-form normal-normal estimator per Carlin & Louis (2000), Efron-Morris.
Method-of-moments hyperparameters; no MCMC, no scipy required.

Math reference: 06-RESEARCH.md Pattern 5; 06-CONTEXT.md D-06.
Pure numpy. No I/O, no logging, no DB.

Model:
    x̄_i | μ_i ~ Normal(μ_i, σ²/n_i)        within-cluster sampling
    μ_i      ~ Normal(μ, τ²)                between-cluster prior
    posterior μ_i | x̄ ~ Normal(post_mean_i, post_var_i)

Estimator:
    μ̂      = mean(x̄)                       grand mean across clusters
    τ̂²     = max(0, var(x̄) - mean(σ²/n))   method-of-moments between-variance
    B_i    = (σ²/n_i) / (σ²/n_i + τ̂²)      per-cluster shrinkage weight ∈ [0,1)
    post_mean_i = (1 - B_i)·x̄_i + B_i·μ̂
    post_var_i  = (1 - B_i)·σ²/n_i
"""

from __future__ import annotations

import numpy as np


def normal_normal_eb(
    xbar: np.ndarray | list[float],
    n: np.ndarray | list[float],
    sigma2: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Closed-form empirical-Bayes normal-normal posterior per cluster.

    Args:
        xbar: per-cluster observed mean (e.g., ev_loss), shape (K,).
        n: per-cluster sample size, shape (K,). Must be positive elementwise.
        sigma2: within-cluster variance assumption (StudyConfig.eb_default_sigma2).

    Returns:
        Tuple ``(post_mean, post_var, mu_hat, tau2_hat)``:
            post_mean: shape (K,) — shrunken posterior mean per cluster.
            post_var:  shape (K,) — posterior variance per cluster.
            mu_hat:    grand mean across clusters (scalar).
            tau2_hat:  method-of-moments between-cluster variance (scalar, ≥ 0).

    Edge cases:
        - K=1: cannot estimate τ²; returns sample mean unchanged, tau2_hat=0,
          post_var = σ²/n (sampling variance only).
        - All x̄ identical (S²=0): tau2_hat=0; returns grand mean for every
          cluster (full shrinkage) and post_var=0.
        - sigma2 ≤ 0 or any n_i ≤ 0: caller must guarantee positives. Violation
          surfaces as numpy warnings / inf values — fail loud rather than
          silently coerce.
    """
    xbar_arr = np.asarray(xbar, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    v = sigma2 / n_arr  # per-cluster sampling variance v_i = σ²/n_i
    mu_hat = float(xbar_arr.mean())
    if len(xbar_arr) < 2:
        # Cannot estimate τ² with a single cluster: return inputs untouched.
        return xbar_arr, v, mu_hat, 0.0
    s2 = float(xbar_arr.var(ddof=1))  # sample variance of x̄ across clusters
    tau2_hat = max(0.0, s2 - float(v.mean()))  # MoM: τ̂² = max(0, var(x̄) - mean(v))
    if tau2_hat == 0.0:
        # Degenerate: full shrinkage to grand mean; posterior variance vanishes.
        return np.full_like(xbar_arr, mu_hat), v * 0.0, mu_hat, tau2_hat
    b = v / (v + tau2_hat)  # shrinkage weight in (0, 1)
    post_mean = (1 - b) * xbar_arr + b * mu_hat
    post_var = (1 - b) * v
    return post_mean, post_var, mu_hat, tau2_hat


def ci_95(post_mean: np.ndarray, post_var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """95% normal credible interval.

    Returns ``(lo, hi) = (mean - 1.96·sqrt(var), mean + 1.96·sqrt(var))``.
    Degenerates to ``[mean, mean]`` when ``post_var == 0``.
    """
    se = np.sqrt(post_var)
    return post_mean - 1.96 * se, post_mean + 1.96 * se


def is_shrunk(n: float, tau2_hat: float, sigma2: float, threshold: float = 0.1) -> bool:
    """True iff the shrinkage weight B exceeds ``threshold`` for this cluster.

    Used by the UI to render ``(n=N, shrunk)`` on low-N rows.

    - Degenerate case (``tau2_hat == 0``): always True (full shrinkage to grand mean).
    - Otherwise: ``B = v / (v + τ̂²)`` where ``v = σ²/n``.
    """
    if tau2_hat == 0:
        return True
    v = sigma2 / max(n, 1)
    return bool((v / (v + tau2_hat)) > threshold)
