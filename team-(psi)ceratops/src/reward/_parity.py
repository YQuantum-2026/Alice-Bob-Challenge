from __future__ import annotations

from collections.abc import Callable

import dynamiqs as dq
import jax.numpy as jnp

from src.cat_qubit import (
    DEFAULT_PARAMS,
    CatQubitParams,
    build_hamiltonian,
    build_jump_ops,
    build_operators,
)
from src.reward._helpers import (
    _compute_lifetime_score,
    _estimate_T_from_log_derivative,
)

# ---------------------------------------------------------------------------
# Parity Decay Rate Reward
# ---------------------------------------------------------------------------


def _build_parity_reward_fn(
    params: CatQubitParams = DEFAULT_PARAMS,
    t_probe_x: float = 0.3,
    t_probe_z: float = 50.0,
    target_bias: float = 100.0,
    w_lifetime: float = 1.0,
    w_bias: float = 0.5,
    use_log_derivative: bool = False,
    n_log_deriv_points: int = 3,
    t_settle: float = 15.0,
    extra_H=None,
    **kw,
) -> Callable[[jnp.ndarray], float]:
    """Build a parity-based reward function with vacuum-based state prep.

    Uses parity operator P = exp(iπ a†a) directly as logical X, which is
    α-independent (no cat-size estimation needed for X measurement).

    Alpha-free: T_X uses parity decay from vacuum (inherently α-independent).
    T_Z uses data-driven α from ⟨a†a⟩ of the settled vacuum state, not the
    heuristic compute_alpha formula. Matches experimental protocol.

    Supports both single-point and log-derivative T estimation modes.
    See _build_proxy_loss_fn for details on the log-derivative method.

    NOT JIT-compiled (data-driven alpha computation between mesolve calls).

    Ref: Parity exp(iπa†a) as logical X. Berdou et al. (2022), arXiv:2204.09128.
    Ref: Réglade et al. (2024), Nature 629 — vacuum-based cat characterization.

    Parameters
    ----------
    params : CatQubitParams
        Fixed hardware parameters.
    t_probe_x : float
        Probe time for T_X (parity) measurement [us].
    t_probe_z : float
        Probe time for T_Z measurement [us].
    target_bias : float
        Target η = T_Z / T_X.
    w_lifetime, w_bias : float
        Reward component weights.
    use_log_derivative : bool
        If True, use log-derivative regression for T estimation.
    n_log_deriv_points : int
        Number of time points for log-derivative regression.

    Returns
    -------
    callable
        Reward function: x (shape (4,)) -> scalar.
    """
    if kw:
        import warnings

        warnings.warn(
            f"_build_parity_reward_fn: unused kwargs {list(kw.keys())}",
            stacklevel=2,
        )
    a, b = build_operators(params)
    jump_ops = build_jump_ops(a, b, params)
    parity_op = (1j * jnp.pi * a.dag() @ a).expm()
    n_a_op = a.dag() @ a  # photon number for data-driven alpha
    na, nb = params.na, params.nb

    # Vacuum initial state for settling
    psi_vacuum = dq.tensor(dq.fock(na, 0), dq.fock(nb, 0))

    if use_log_derivative:
        fracs = jnp.linspace(1.0 / n_log_deriv_points, 1.0, n_log_deriv_points)
        t_points_z = fracs * t_probe_z
        t_points_x = fracs * t_probe_x
        tsave_z_arr = jnp.concatenate([jnp.array([0.0]), t_points_z])
        # X measurement: settle + measure parity at multiple points
        tsave_x_arr = jnp.concatenate(
            [
                jnp.array([0.0, t_settle]),
                t_settle + fracs * t_probe_x,
            ]
        )
    else:
        tsave_z_arr = jnp.array([0.0, t_probe_z])
        tsave_x_arr = jnp.array([0.0, t_settle, t_settle + t_probe_x])

    def parity_reward(x, extra_H_override=None):
        g2_re, g2_im, eps_d_re, eps_d_im = x[0], x[1], x[2], x[3]

        H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
        # Add Hamiltonian drift perturbation (detuning, Kerr) if provided
        h_pert = extra_H_override if extra_H_override is not None else extra_H
        if h_pert is not None:
            H = H + h_pert

        # --- T_X: parity decay from vacuum (α-independent) ---
        # Vacuum has even parity → evolves to |C₊⟩ = |+x⟩ under two-photon dissipation.
        # Parity decay rate gives T_X directly.
        res_x = dq.mesolve(
            H,
            jump_ops,
            psi_vacuum,
            tsave_x_arr,
            exp_ops=[parity_op, n_a_op],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )

        # Data-driven α from ⟨a†a⟩ at settling time
        n_a_settled = res_x.expects[1, 1].real  # ⟨a†a⟩ at t_settle
        alpha = jnp.sqrt(jnp.maximum(n_a_settled, 0.01))

        # --- T_Z: quadrature decay from |α_est⟩ ---
        # Well orientation from drive phase (exact, not heuristic)
        g2 = g2_re + 1j * g2_im
        eps_d = eps_d_re + 1j * eps_d_im
        theta = jnp.angle(g2 * eps_d) / 2.0
        phase_factor = jnp.exp(-1j * theta)
        Q_theta = a * phase_factor + a.dag() * jnp.conj(phase_factor)

        alpha_est = alpha * jnp.exp(1j * theta)
        psi_alpha = dq.tensor(dq.coherent(na, alpha_est), dq.fock(nb, 0))

        res_z = dq.mesolve(
            H,
            jump_ops,
            psi_alpha,
            tsave_z_arr,
            exp_ops=[Q_theta],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )

        # --- Estimate lifetimes ---
        if use_log_derivative:
            # Parity values after settling (skip t=0 and t_settle)
            ex_vals = jnp.abs(res_x.expects[0, 2:])
            ez_vals = jnp.abs(res_z.expects[0, 1:].real)
            T_X_est = _estimate_T_from_log_derivative(t_points_x, ex_vals)
            T_Z_est = _estimate_T_from_log_derivative(t_points_z, ez_vals)
        else:
            ex = jnp.abs(res_x.expects[0, -1])
            ez = jnp.abs(res_z.expects[0, -1].real)
            ex_safe = jnp.maximum(ex, 1e-6)
            ez_safe = jnp.maximum(ez, 1e-6)
            log_ex = jnp.minimum(jnp.log(ex_safe), -1e-6)
            log_ez = jnp.minimum(jnp.log(ez_safe), -1e-6)
            T_X_est = -t_probe_x / log_ex
            T_Z_est = -t_probe_z / log_ez

        return _compute_lifetime_score(
            T_Z_est, T_X_est, target_bias, w_lifetime, w_bias
        )

    return parity_reward
