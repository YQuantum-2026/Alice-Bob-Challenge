from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import dynamiqs as dq
import jax.numpy as jnp
from jax import jit, vmap

from src.cat_qubit import (
    CatQubitParams,
    build_hamiltonian,
    build_jump_ops,
    build_logical_ops,
    build_operators,
    compute_alpha,
)
from src.reward._enhanced import _build_enhanced_proxy_fn
from src.reward._helpers import (
    _compute_lifetime_score,
    _estimate_T_from_log_derivative,
)
from src.reward._parity import _build_parity_reward_fn
from src.reward._proxy import _build_multipoint_proxy_fn, _build_proxy_loss_fn
from src.reward._simple import _build_fidelity_reward_fn, _build_photon_reward_fn
from src.reward._spectral import _build_spectral_reward_fn
from src.reward._vacuum import _build_vacuum_reward_fn

if TYPE_CHECKING:
    from src.config import RewardConfig

# Reward types that cannot be JIT-compiled (they use data-driven alpha
# computation between mesolve calls). Batched via Python loop, not jit(vmap).
_NON_JIT_REWARDS: frozenset[str] = frozenset(
    {"vacuum", "fidelity", "parity", "spectral"}
)

# ---------------------------------------------------------------------------
# Unified reward factory
# ---------------------------------------------------------------------------


def build_reward(
    reward_type: str, params: CatQubitParams, reward_cfg: RewardConfig | None = None
):
    """Dispatch to the appropriate reward builder and return (reward_fn, batched_fn).

    Alpha-free rewards (vacuum, fidelity, parity) use vacuum-based state
    preparation with data-driven alpha from ⟨a†a⟩. These are NOT JIT-compiled
    and are batched via Python loops (~N× slower than vmap'd rewards).

    JIT-compiled rewards (proxy, enhanced_proxy, multipoint, photon) use the
    heuristic compute_alpha for state prep, enabling vmap batching.

    Parameters
    ----------
    reward_type : str
        One of "proxy", "photon", "fidelity", "parity", "multipoint",
        "spectral", "enhanced_proxy", "vacuum".
    params : CatQubitParams
        Fixed hardware parameters.
    reward_cfg : RewardConfig or None
        If None, use default kwargs for each reward type.

    Returns
    -------
    reward_fn : callable
        Reward function: x (shape (4,)) -> scalar reward.
    batched_reward_fn : callable
        Batched version: xs (shape (N, 4)) -> (N,) rewards.

    Raises
    ------
    ValueError
        If reward_type is unknown.
    """
    if reward_cfg is not None:
        cfg = reward_cfg
    else:
        from src.config import RewardConfig

        cfg = RewardConfig()

    if reward_type == "proxy":
        reward_fn = _build_proxy_loss_fn(
            params,
            t_probe_z=cfg.t_probe_z,
            t_probe_x=cfg.t_probe_x,
            target_bias=cfg.target_bias,
            w_lifetime=cfg.w_lifetime,
            w_bias=cfg.w_bias,
            use_log_derivative=cfg.use_log_derivative,
            n_log_deriv_points=cfg.n_log_deriv_points,
        )
    elif reward_type == "photon":
        reward_fn = _build_photon_reward_fn(
            params,
            n_target=cfg.n_target,
            t_steady=cfg.t_steady,
        )
    elif reward_type == "fidelity":
        reward_fn = _build_fidelity_reward_fn(
            params,
            t_steady=cfg.t_steady,
        )
    elif reward_type == "parity":
        reward_fn = _build_parity_reward_fn(
            params,
            t_probe_x=cfg.t_probe_x,
            t_probe_z=cfg.t_probe_z,
            target_bias=cfg.target_bias,
            w_lifetime=cfg.w_lifetime,
            w_bias=cfg.w_bias,
            use_log_derivative=cfg.use_log_derivative,
            n_log_deriv_points=cfg.n_log_deriv_points,
            t_settle=cfg.t_settle,
        )
    elif reward_type == "multipoint":
        reward_fn = _build_multipoint_proxy_fn(
            params,
            t_probe_z=cfg.t_probe_z,
            t_probe_x=cfg.t_probe_x,
            n_points=3,
            target_bias=cfg.target_bias,
            w_lifetime=cfg.w_lifetime,
            w_bias=cfg.w_bias,
        )
    elif reward_type == "spectral":
        reward_fn = _build_spectral_reward_fn(
            params,
            target_bias=cfg.target_bias,
            w_lifetime=cfg.w_lifetime,
            w_bias=cfg.w_bias,
        )
    elif reward_type == "enhanced_proxy":
        reward_fn = _build_enhanced_proxy_fn(
            params,
            t_probe_z=cfg.t_probe_z,
            t_probe_x=cfg.t_probe_x,
            target_bias=cfg.target_bias,
            w_lifetime=cfg.w_lifetime,
            w_bias=cfg.w_bias,
            w_buffer=cfg.w_buffer,
            w_confinement=cfg.w_confinement,
            w_margin=cfg.w_margin,
            margin_threshold=cfg.margin_threshold,
            use_log_derivative=cfg.use_log_derivative,
            n_log_deriv_points=cfg.n_log_deriv_points,
        )
    elif reward_type == "vacuum":
        reward_fn = _build_vacuum_reward_fn(
            params=params,
            t_settle=cfg.t_settle,
            t_measure_z=cfg.t_measure_z,
            t_measure_x=cfg.t_measure_x,
            n_measure_points=cfg.n_log_deriv_points,
            target_bias=cfg.target_bias,
            w_lifetime=cfg.w_lifetime,
            w_bias=cfg.w_bias,
        )
    else:
        raise ValueError(
            f"Unknown reward_type '{reward_type}'. Options: "
            "'proxy', 'photon', 'fidelity', 'parity', 'multipoint', "
            "'spectral', 'enhanced_proxy', 'vacuum'."
        )

    # Non-JIT rewards use data-driven alpha computation between mesolve calls.
    # Batch via Python loop instead of jit(vmap(...)).
    # enhanced_proxy is non-JIT when confinement is active (vacuum settle needed).
    non_jit = set(_NON_JIT_REWARDS)
    if reward_type == "enhanced_proxy" and cfg.w_confinement > 0:
        non_jit.add("enhanced_proxy")
    if reward_type in non_jit:

        def _batched_loop(xs):
            results = []
            for i in range(xs.shape[0]):
                try:
                    results.append(float(reward_fn(xs[i])))
                except Exception as e:
                    warnings.warn(
                        f"Reward eval failed for sample {i}: {e}", stacklevel=2
                    )
                    results.append(float("-inf"))
            return jnp.array(results)

        return reward_fn, _batched_loop

    batched_reward_fn = jit(vmap(reward_fn))
    return reward_fn, batched_reward_fn


# ---------------------------------------------------------------------------
# Drift-Aware Wrapper
# ---------------------------------------------------------------------------


def build_drift_aware_reward(
    reward_type: str,
    params: CatQubitParams,
    reward_cfg: RewardConfig | None = None,
    n_drift_slots: int = 6,
):
    """Build a drift-aware reward that accepts an extended parameter vector.

    The extended vector has shape (4 + n_drift_slots,):
      [g2_re, g2_im, eps_d_re, eps_d_im |
       g2_re_drift, g2_im_drift, eps_d_re_drift, eps_d_im_drift,
       detuning, kerr]

    Drift slots:
      0: g2_re offset -- added to control g2_re
      1: g2_im offset -- added to control g2_im
      2: eps_d_re offset -- added to control eps_d_re
      3: eps_d_im offset -- added to control eps_d_im
      4: detuning -- coefficient for a+a Hamiltonian term [MHz]
      5: kerr -- coefficient for (a+a)^2 Hamiltonian term [MHz]

    Parameters
    ----------
    reward_type : str
        One of "proxy", "photon", "fidelity", "parity", "multipoint",
        "spectral", "enhanced_proxy", "vacuum".
    params : CatQubitParams
        Fixed hardware parameters.
    reward_cfg : RewardConfig or None
        Reward configuration (forwarded to build_reward).
    n_drift_slots : int
        Number of drift offset parameters appended to the control vector.

    Returns
    -------
    drift_fn : callable
        JIT-compiled: x_ext (shape (4 + n_drift_slots,)) -> scalar reward.
    batched_drift_fn : callable
        JIT-compiled: xs_ext (shape (N, 4 + n_drift_slots)) -> (N,) rewards.
    """
    if reward_cfg is not None:
        cfg = reward_cfg
    else:
        from src.config import RewardConfig

        cfg = RewardConfig()

    # Mirror build_reward() logic: enhanced_proxy is non-JIT when confinement active
    non_jit = set(_NON_JIT_REWARDS)
    if reward_type == "enhanced_proxy" and cfg.w_confinement > 0:
        non_jit.add("enhanced_proxy")

    # Build operators and pre-compute drift Hamiltonian components
    a, b = build_operators(params)
    jump_ops = build_jump_ops(a, b, params)
    n_op = a.dag() @ a  # a+a  -- for detuning drift term
    n_op_sq = n_op @ n_op  # (a+a)^2 -- for Kerr drift term

    if reward_type == "proxy":
        t_probe_z = cfg.t_probe_z
        t_probe_x = cfg.t_probe_x
        target_bias = cfg.target_bias
        w_lifetime = cfg.w_lifetime
        w_bias = cfg.w_bias
        use_log_derivative = cfg.use_log_derivative
        n_log_deriv_points = cfg.n_log_deriv_points

        # Pre-compute time grids based on estimation mode
        if use_log_derivative:
            fracs = jnp.linspace(1.0 / n_log_deriv_points, 1.0, n_log_deriv_points)
            t_points_z = fracs * t_probe_z
            t_points_x = fracs * t_probe_x
            tsave_z_arr = jnp.concatenate([jnp.array([0.0]), t_points_z])
            tsave_x_arr = jnp.concatenate([jnp.array([0.0]), t_points_x])
        else:
            tsave_z_arr = jnp.array([0.0, t_probe_z])
            tsave_x_arr = jnp.array([0.0, t_probe_x])

        @jit
        def drift_fn(x_ext):
            control = x_ext[:4]
            drift = x_ext[4 : 4 + n_drift_slots]

            # Apply control offsets (drift adds to the physical parameter)
            g2_re_eff = control[0] + drift[0]
            g2_im_eff = control[1] + drift[1]
            eps_d_re_eff = control[2] + drift[2]
            eps_d_im_eff = control[3] + drift[3]

            # Build base Hamiltonian with effective parameters
            H = build_hamiltonian(
                a, b, g2_re_eff, g2_im_eff, eps_d_re_eff, eps_d_im_eff
            )

            # Add Hamiltonian drift terms: detuning * a+a + kerr * (a+a)^2
            detuning = drift[4]
            kerr = drift[5]
            H = H + detuning * n_op + kerr * n_op_sq

            # Compute alpha and logical operators from effective params
            alpha = compute_alpha(
                g2_re_eff, g2_im_eff, eps_d_re_eff, eps_d_im_eff, params
            )
            sx, sz = build_logical_ops(a, b, alpha, params)

            # --- Measure Z_L from |+z> ---
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

            # --- Measure X_L from |+x> ---
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
                ez_vals = res_z.expects[1, 1:].real
                ex_vals = res_x.expects[0, 1:].real
                T_Z_est = _estimate_T_from_log_derivative(t_points_z, ez_vals)
                T_X_est = _estimate_T_from_log_derivative(t_points_x, ex_vals)
            else:
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

    else:
        # For non-proxy reward types, apply control offsets (slots 0-3) AND
        # Hamiltonian drift terms (detuning slot 4, Kerr slot 5).
        base_reward_fn, _ = build_reward(reward_type, params, reward_cfg)

        # All reward types now support extra_H_override for Hamiltonian drift.
        def drift_fn(x_ext):
            control = x_ext[:4]
            drift = x_ext[4 : 4 + n_drift_slots]

            g2_re_eff = control[0] + drift[0]
            g2_im_eff = control[1] + drift[1]
            eps_d_re_eff = control[2] + drift[2]
            eps_d_im_eff = control[3] + drift[3]
            x_eff = jnp.array([g2_re_eff, g2_im_eff, eps_d_re_eff, eps_d_im_eff])

            # Build Hamiltonian drift perturbation from slots 4-5.
            # Always construct drift_H (even if zero) to avoid changing
            # the type of drift_H between None and array, which would
            # cause JIT recompilation if n_drift_slots changes.
            detuning = drift[4] if n_drift_slots > 4 else 0.0
            kerr = drift[5] if n_drift_slots > 5 else 0.0
            drift_H = detuning * n_op + kerr * n_op_sq

            return base_reward_fn(x_eff, extra_H_override=drift_H)

        # JIT-compile only if the reward is JIT-compatible
        if reward_type not in non_jit:
            drift_fn = jit(drift_fn)

    # Batch with Python loop for non-JIT rewards, jit(vmap) otherwise
    if reward_type in non_jit:

        def _batched_loop(xs):
            results = []
            for i in range(xs.shape[0]):
                try:
                    results.append(float(drift_fn(xs[i])))
                except Exception as e:
                    warnings.warn(
                        f"Drift reward eval failed for sample {i}: {e}", stacklevel=2
                    )
                    results.append(float("-inf"))
            return jnp.array(results)

        return drift_fn, _batched_loop

    batched_drift_fn = jit(vmap(drift_fn))
    return drift_fn, batched_drift_fn
