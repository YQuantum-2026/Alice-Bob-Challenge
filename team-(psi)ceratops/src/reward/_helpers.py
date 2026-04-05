"""Reward functions for cat qubit online optimization.

Provides:
  - Full measurement reward (T_X/T_Z via exponential fits — slow, for validation)
  - Proxy reward (single-point expectations — fast, JIT-compatible, differentiable)
  - Batched versions for CMA-ES population evaluation

Design insight (Sivak et al. 2025):
  Instead of fitting full exponential decay curves to extract T_X and T_Z,
  measure expectation values at a single probe time. For exp(-t/T), a larger
  T yields a larger expectation at any fixed t_probe. This is fully
  differentiable through dynamiqs/JAX and requires no scipy curve fitting.
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from src.cat_qubit import (
    DEFAULT_PARAMS,
    CatQubitParams,
    measure_Tx,
    measure_Tz,
)

# ---------------------------------------------------------------------------
# Log-derivative lifetime estimation (JIT-compatible)
# ---------------------------------------------------------------------------


def _estimate_T_from_log_derivative(t_points, expectations):
    """Estimate lifetime T from the slope of log(⟨O⟩) vs t.

    For exponential decay ⟨O⟩(t) = A·exp(-t/T), we have:
        log⟨O⟩(t) = log(A) - t/T

    The slope d(log⟨O⟩)/dt = -1/T is estimated via ordinary least squares
    regression on (t_i, log⟨O⟩(t_i)). This method:
      - Is invariant to the amplitude A (absorbed by intercept)
      - Uses multiple points for noise robustness
      - Is fully JIT-compatible (closed-form linear regression)

    Parameters
    ----------
    t_points : Array, shape (n,)
        Time points (must be > 0).
    expectations : Array, shape (n,)
        Expectation values ⟨O⟩(t_i) at each time point.

    Returns
    -------
    T_est : scalar
        Estimated lifetime.
    """
    # Guard against log(0) or log(negative)
    y = jnp.log(jnp.maximum(jnp.abs(expectations), 1e-8))
    t = t_points

    # Closed-form OLS: slope = (n·Σ(ty) - Σt·Σy) / (n·Σ(t²) - (Σt)²)
    n = t.shape[0]
    sum_t = jnp.sum(t)
    sum_y = jnp.sum(y)
    sum_ty = jnp.sum(t * y)
    sum_t2 = jnp.sum(t**2)

    denom = n * sum_t2 - sum_t**2
    # denom = n² * Var(t) >= 0 by construction; clamp for float safety
    denom_safe = jnp.maximum(denom, 1e-12)
    slope = (n * sum_ty - sum_t * sum_y) / denom_safe

    # slope = -1/T, so T = -1/slope. Slope should be negative for decay.
    slope_safe = jnp.minimum(slope, -1e-8)
    T_est = -1.0 / slope_safe

    # Clamp to [1e-6, 1e4] to prevent extreme values in log(T).
    # Upper bound 1e4 μs = 10 ms is a realistic ceiling — lifetimes beyond
    # this are unresolvable with typical probe times and give false reward signal.
    return jnp.clip(T_est, 1e-6, 1e4)


# ---------------------------------------------------------------------------
# Lifetime estimation helpers (shared by multiple reward functions)
# ---------------------------------------------------------------------------


def _estimate_T_single_point(expectation, t_probe):
    """Estimate lifetime from a single expectation value at t_probe.

    T = -t_probe / log(⟨O⟩(t_probe)), assuming ⟨O⟩(t) = exp(-t/T).

    Parameters
    ----------
    expectation : scalar
        Expectation value at t_probe. Must be > 0 for valid result.
    t_probe : float
        Time at which the expectation was measured.

    Returns
    -------
    T_est : scalar
        Estimated lifetime.
    """
    e_safe = jnp.maximum(jnp.abs(expectation), 1e-6)
    log_e = jnp.minimum(jnp.log(e_safe), -1e-6)
    return -t_probe / log_e


def _compute_lifetime_score(T_Z_est, T_X_est, target_bias, w_lifetime, w_bias):
    """Compute the composite lifetime reward from T_Z and T_X estimates.

    R = w_lifetime * [log(T_Z) + log(T_X)] - w_bias * ((η - η_target) / η_target)²

    Parameters
    ----------
    T_Z_est, T_X_est : scalar
        Estimated bit-flip and phase-flip lifetimes.
    target_bias : float
        Target η = T_Z / T_X.
    w_lifetime, w_bias : float
        Weights on lifetime and bias components.

    Returns
    -------
    reward : scalar
    """
    lifetime_score = w_lifetime * (
        jnp.log(jnp.maximum(T_Z_est, 1e-6)) + jnp.log(jnp.maximum(T_X_est, 1e-6))
    )
    bias_est = T_Z_est / jnp.maximum(T_X_est, 1e-6)
    bias_penalty = -w_bias * ((bias_est - target_bias) / target_bias) ** 2
    return lifetime_score + bias_penalty


def _make_time_grid(t_probe, use_log_derivative, n_log_deriv_points):
    """Build the time save array for a lifetime measurement.

    Parameters
    ----------
    t_probe : float
        Total measurement window.
    use_log_derivative : bool
        If True, return evenly spaced points for log-derivative regression.
    n_log_deriv_points : int
        Number of measurement points (log-derivative mode only).

    Returns
    -------
    tsave : Array
        Time points including t=0.
    t_points : Array or None
        Measurement points (excluding t=0), or None for single-point mode.
    """
    if use_log_derivative:
        fracs = jnp.linspace(1.0 / n_log_deriv_points, 1.0, n_log_deriv_points)
        t_points = fracs * t_probe
        tsave = jnp.concatenate([jnp.array([0.0]), t_points])
        return tsave, t_points
    else:
        tsave = jnp.array([0.0, t_probe])
        return tsave, None


def _estimate_T_from_trace(expectations, t_points, t_probe, use_log_derivative):
    """Estimate lifetime T from expectation trace using configured method.

    Parameters
    ----------
    expectations : Array
        Expectation values (excluding t=0 for log-derivative, or just final for single-point).
    t_points : Array or None
        Time points for log-derivative fit (None for single-point).
    t_probe : float
        Probe time (single-point mode).
    use_log_derivative : bool
        Which estimation method to use.

    Returns
    -------
    T_est : scalar
    """
    if use_log_derivative:
        return _estimate_T_from_log_derivative(t_points, expectations)
    else:
        return _estimate_T_single_point(expectations, t_probe)


# ---------------------------------------------------------------------------
# Full measurement reward (baseline, NOT JIT-compatible)
# ---------------------------------------------------------------------------


def reward_full(
    x,
    target_bias=100.0,
    lambda_bias=1.0,
    tfinal_z=200.0,
    tfinal_x=1.0,
    params: CatQubitParams = DEFAULT_PARAMS,
):
    """Compute reward from full T_X/T_Z exponential decay fitting.

    R = log(T_Z) + log(T_X) - λ * ((T_Z/T_X - η_target) / η_target)²

    The bias penalty uses normalized relative error: deviations from
    the target bias are penalized proportionally to how far off they
    are as a fraction of the target.

    This is expensive (2 mesolve + 2 curve fits) and NOT JIT-compatible.
    Use for periodic validation, not inside the optimization loop.

    Parameters
    ----------
    x : array-like, shape (4,)
        [g2_re, g2_im, eps_d_re, eps_d_im].
    target_bias : float
        Target η = T_Z / T_X.
    lambda_bias : float
        Weight on the bias penalty term.
    tfinal_z, tfinal_x : float
        Simulation durations for T_Z and T_X measurement.
    params : CatQubitParams
        Fixed hardware parameters.

    Returns
    -------
    float
        Reward value.
    """
    g2_re, g2_im, eps_d_re, eps_d_im = (
        float(x[0]),
        float(x[1]),
        float(x[2]),
        float(x[3]),
    )

    Tz = measure_Tz(g2_re, g2_im, eps_d_re, eps_d_im, tfinal_z, params)
    Tx = measure_Tx(g2_re, g2_im, eps_d_re, eps_d_im, tfinal_x, params)

    Tz = max(Tz, 1e-6)
    Tx = max(Tx, 1e-6)

    # R = log(T_Z) + log(T_X) - λ * ((T_Z/T_X - η) / η)²
    lifetime_score = np.log(Tz) + np.log(Tx)
    bias = Tz / Tx
    bias_penalty = -lambda_bias * ((bias - target_bias) / target_bias) ** 2

    return float(lifetime_score + bias_penalty)
