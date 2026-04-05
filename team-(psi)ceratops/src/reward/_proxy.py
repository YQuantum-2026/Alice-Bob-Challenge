from __future__ import annotations

from collections.abc import Callable

import dynamiqs as dq
import jax.numpy as jnp
from jax import jit, vmap

from src.cat_qubit import (
    DEFAULT_PARAMS,
    CatQubitParams,
    build_hamiltonian,
    build_jump_ops,
    build_logical_ops,
    build_operators,
    compute_alpha,
)
from src.reward._helpers import (
    _compute_lifetime_score,
    _estimate_T_from_log_derivative,
)

# ---------------------------------------------------------------------------
# Proxy reward (fast, JIT-compatible, differentiable)
# ---------------------------------------------------------------------------


def _build_proxy_loss_fn(
    params: CatQubitParams = DEFAULT_PARAMS,
    t_probe_z: float = 50.0,
    t_probe_x: float = 0.3,
    target_bias: float = 100.0,
    w_lifetime: float = 1.0,
    w_bias: float = 0.5,
    use_log_derivative: bool = False,
    n_log_deriv_points: int = 3,
    extra_H=None,
) -> Callable[[jnp.ndarray], float]:
    """Build a JIT-compiled proxy reward function for given parameters.

    Returns a function: reward_fn(x) -> scalar, where x = [g2_re, g2_im, eps_d_re, eps_d_im].

    **Alpha note**: This reward uses the heuristic adiabatic elimination formula
    (compute_alpha) for state preparation, enabling JIT compilation and vmap
    batching. For the physically correct alpha-free approach, use the "vacuum"
    reward type which estimates alpha from simulation data.

    Two estimation modes:
      - Single-point (default): T = -t_probe / log(⟨O⟩(t_probe))
      - Log-derivative (use_log_derivative=True): Estimate T from slope of
        log(⟨O⟩) vs t via OLS regression on n_log_deriv_points points.
        Since d(log⟨O⟩)/dt = -1/T for exponential decay, the slope
        directly gives the lifetime. This is more robust because it is
        invariant to the initial amplitude A.

    Parameters
    ----------
    params : CatQubitParams
        Fixed hardware parameters.
    t_probe_z : float
        Probe time for T_Z measurement [us].
    t_probe_x : float
        Probe time for T_X measurement [us].
    target_bias : float
        Target η = T_Z / T_X.
    w_lifetime : float
        Weight on lifetime maximization component.
    w_bias : float
        Weight on bias targeting component.
    use_log_derivative : bool
        If True, use log-derivative regression for T estimation.
    n_log_deriv_points : int
        Number of time points for log-derivative regression.

    Returns
    -------
    callable
        JIT-compiled reward function: x -> scalar.
    """
    a, b = build_operators(params)
    jump_ops = build_jump_ops(a, b, params)

    # Pre-compute time grids based on estimation mode
    if use_log_derivative:
        # Evenly spaced points in (0, t_probe], excluding t=0
        fracs = jnp.linspace(1.0 / n_log_deriv_points, 1.0, n_log_deriv_points)
        t_points_z = fracs * t_probe_z
        t_points_x = fracs * t_probe_x
        tsave_z_arr = jnp.concatenate([jnp.array([0.0]), t_points_z])
        tsave_x_arr = jnp.concatenate([jnp.array([0.0]), t_points_x])
    else:
        tsave_z_arr = jnp.array([0.0, t_probe_z])
        tsave_x_arr = jnp.array([0.0, t_probe_x])

    @jit
    def proxy_reward(x, extra_H_override=None):
        g2_re, g2_im, eps_d_re, eps_d_im = x[0], x[1], x[2], x[3]

        # Build Hamiltonian
        H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
        # Add Hamiltonian drift perturbation if provided
        h_pert = extra_H_override if extra_H_override is not None else extra_H
        if h_pert is not None:
            H = H + h_pert

        # Estimate alpha and build logical operators + initial states
        alpha = compute_alpha(g2_re, g2_im, eps_d_re, eps_d_im, params)
        sx, sz = build_logical_ops(a, b, alpha, params)

        # --- Measure ⟨Z_L⟩ from |+z⟩ ---
        g_state = dq.coherent(params.na, alpha)
        psi_z = dq.tensor(g_state, dq.fock(params.nb, 0))

        res_z = dq.mesolve(
            H,
            jump_ops,
            psi_z,
            tsave_z_arr,
            exp_ops=[sx, sz],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )

        # --- Measure ⟨X_L⟩ from |+x⟩ ---
        e_state = dq.coherent(params.na, -alpha)
        cat_plus = dq.unit(g_state + e_state)
        psi_x = dq.tensor(cat_plus, dq.fock(params.nb, 0))

        res_x = dq.mesolve(
            H,
            jump_ops,
            psi_x,
            tsave_x_arr,
            exp_ops=[sx, sz],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )

        # --- Estimate lifetimes ---
        if use_log_derivative:
            # Log-derivative: T = -1 / slope(log⟨O⟩ vs t)
            ez_vals = res_z.expects[1, 1:].real  # skip t=0
            ex_vals = res_x.expects[0, 1:].real
            T_Z_est = _estimate_T_from_log_derivative(t_points_z, ez_vals)
            T_X_est = _estimate_T_from_log_derivative(t_points_x, ex_vals)
        else:
            # Single-point: T = -t / log(⟨O⟩)
            ez = res_z.expects[1, -1].real
            ex = res_x.expects[0, -1].real
            ez_safe = jnp.maximum(ez, 1e-6)
            ex_safe = jnp.maximum(jnp.abs(ex), 1e-6)
            log_ez = jnp.minimum(jnp.log(ez_safe), -1e-6)
            log_ex = jnp.minimum(jnp.log(ex_safe), -1e-6)
            T_Z_est = -t_probe_z / log_ez
            T_X_est = -t_probe_x / log_ex

        return _compute_lifetime_score(
            T_Z_est, T_X_est, target_bias, w_lifetime, w_bias
        )

    return proxy_reward


def _build_multipoint_proxy_fn(
    params: CatQubitParams = DEFAULT_PARAMS,
    t_probe_z: float = 50.0,
    t_probe_x: float = 0.3,
    n_points: int = 3,
    target_bias: float = 100.0,
    w_lifetime: float = 1.0,
    w_bias: float = 0.5,
    extra_H=None,
) -> Callable[[jnp.ndarray], float]:
    """Build a multi-point proxy reward for more robust T estimation.

    Instead of measuring at a single time point, measures at n_points evenly
    spaced times and estimates T via weighted geometric mean of per-point
    estimates: T_i = -t_i / log(⟨O⟩(t_i)).

    More robust to noise than single-point, still JIT-compatible and
    differentiable. No scipy needed.

    Note: The log-derivative method (_estimate_T_from_log_derivative, enabled
    via use_log_derivative=True in RewardConfig) solves the same noise-
    robustness problem more elegantly. The log-derivative OLS slope is
    invariant to the initial amplitude A, whereas the per-point estimates
    here assume A=1, making them more sensitive to amplitude uncertainty.
    Prefer log-derivative mode for new experiments.

    Parameters
    ----------
    params : CatQubitParams
    t_probe_z : float
        Max probe time for T_Z [us].
    t_probe_x : float
        Max probe time for T_X [us].
    n_points : int
        Number of measurement points (default 3).
    target_bias, w_lifetime, w_bias : float
        Reward weights.
    extra_H : QArray or None
        Static Hamiltonian perturbation (closed over, JIT-compatible).

    Returns
    -------
    callable
        JIT-compiled reward function: x (shape (4,)) -> scalar.
        Accepts optional extra_H_override kwarg for runtime drift.
    """
    a, b = build_operators(params)
    jump_ops = build_jump_ops(a, b, params)

    # Evenly spaced probe times (exclude 0)
    fracs = jnp.linspace(1.0 / n_points, 1.0, n_points)
    t_points_z = fracs * t_probe_z
    t_points_x = fracs * t_probe_x

    @jit
    def multipoint_reward(x, extra_H_override=None):
        g2_re, g2_im, eps_d_re, eps_d_im = x[0], x[1], x[2], x[3]
        H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
        h_pert = extra_H_override if extra_H_override is not None else extra_H
        if h_pert is not None:
            H = H + h_pert
        alpha = compute_alpha(g2_re, g2_im, eps_d_re, eps_d_im, params)
        sx, sz = build_logical_ops(a, b, alpha, params)

        # --- T_Z: measure ⟨Z_L⟩ at n_points times from |+z⟩ ---
        g_state = dq.coherent(params.na, alpha)
        psi_z = dq.tensor(g_state, dq.fock(params.nb, 0))
        tsave_z = jnp.concatenate([jnp.array([0.0]), t_points_z])

        res_z = dq.mesolve(
            H,
            jump_ops,
            psi_z,
            tsave_z,
            exp_ops=[sx, sz],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )
        # ⟨Z_L⟩ at each probe time (skip t=0)
        ez_vals = res_z.expects[1, 1:].real
        ez_safe = jnp.maximum(ez_vals, 1e-6)
        log_ez = jnp.minimum(jnp.log(ez_safe), -1e-6)
        T_Z_estimates = -t_points_z / log_ez
        T_Z_est = jnp.exp(jnp.mean(jnp.log(jnp.maximum(T_Z_estimates, 1e-6))))

        # --- T_X: measure ⟨X_L⟩ at n_points times from |+x⟩ ---
        e_state = dq.coherent(params.na, -alpha)
        psi_x = dq.tensor(dq.unit(g_state + e_state), dq.fock(params.nb, 0))
        tsave_x = jnp.concatenate([jnp.array([0.0]), t_points_x])

        res_x = dq.mesolve(
            H,
            jump_ops,
            psi_x,
            tsave_x,
            exp_ops=[sx, sz],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )
        ex_vals = res_x.expects[0, 1:].real
        ex_safe = jnp.maximum(jnp.abs(ex_vals), 1e-6)
        log_ex = jnp.minimum(jnp.log(ex_safe), -1e-6)
        T_X_estimates = -t_points_x / log_ex
        T_X_est = jnp.exp(jnp.mean(jnp.log(jnp.maximum(T_X_estimates, 1e-6))))

        return _compute_lifetime_score(
            T_Z_est, T_X_est, target_bias, w_lifetime, w_bias
        )

    return multipoint_reward


def build_proxy_reward(params: CatQubitParams = DEFAULT_PARAMS, **kwargs):
    """Build proxy reward function and its batched version.

    Returns
    -------
    reward_fn : callable
        JIT-compiled: x (shape (4,)) -> scalar reward.
    batched_reward_fn : callable
        JIT-compiled: xs (shape (N, 4)) -> (N,) rewards.
    """
    reward_fn = _build_proxy_loss_fn(params, **kwargs)
    batched_reward_fn = jit(vmap(reward_fn))
    return reward_fn, batched_reward_fn


# ---------------------------------------------------------------------------
# Simple loss function for CMA-ES (negated reward, for minimization)
# ---------------------------------------------------------------------------


def build_cmaes_loss(params: CatQubitParams = DEFAULT_PARAMS, **kwargs):
    """Build a loss function suitable for CMA-ES (which minimizes).

    Returns
    -------
    loss_fn : callable
        JIT-compiled: x (shape (4,)) -> scalar loss (lower = better).
    batched_loss_fn : callable
        JIT-compiled: xs (shape (N, 4)) -> (N,) losses.
    """
    reward_fn = _build_proxy_loss_fn(params, **kwargs)

    @jit
    def loss_fn(x):
        return -reward_fn(x)

    batched_loss_fn = jit(vmap(loss_fn))
    return loss_fn, batched_loss_fn
