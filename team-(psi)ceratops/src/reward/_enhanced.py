from __future__ import annotations

from collections.abc import Callable

import dynamiqs as dq
import jax.numpy as jnp

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
    _estimate_T_from_log_derivative,
)

# ---------------------------------------------------------------------------
# Enhanced Proxy Reward (physics-augmented)
# ---------------------------------------------------------------------------


def _build_enhanced_proxy_fn(
    params: CatQubitParams = DEFAULT_PARAMS,
    t_probe_z: float = 50.0,
    t_probe_x: float = 0.3,
    target_bias: float = 100.0,
    w_lifetime: float = 1.0,
    w_bias: float = 0.5,
    w_buffer: float = 0.0,
    w_confinement: float = 0.0,
    w_margin: float = 0.0,
    margin_threshold: float = 1.0,
    use_log_derivative: bool = False,
    n_log_deriv_points: int = 3,
    extra_H=None,
) -> Callable[[jnp.ndarray], float]:
    """Build a physics-enhanced proxy reward with code-space guardrails.

    **Alpha note**: When w_confinement == 0 (default), this reward uses the
    heuristic adiabatic elimination formula (compute_alpha) for state
    preparation, enabling JIT compilation and vmap batching.

    When w_confinement > 0, the confinement penalty uses data-driven alpha
    from a quick vacuum settle (1 extra mesolve call). This makes the
    function NOT JIT-compatible. The code space projector Pi_code is built
    from the actual cat size, not the heuristic formula which is unreliable
    near the cat threshold.

    For a fully alpha-free approach, use the "vacuum" reward type.

    Extends the standard proxy reward with three additional physics-motivated
    terms that penalize parameter regimes where the cat qubit model breaks down:

    R = w_lifetime * [log(T_Z_est) + log(T_X_est)]
      - w_bias * ((eta - eta_target) / eta_target)^2
      - w_buffer * <n_b>                                 (buffer should be empty)
      - w_confinement * (1 - confinement)^2              (stay in code space)
      - w_margin * max(0, threshold - margin)^2          (distance from threshold)

    Physics rationale:
      - Buffer occupation: The adiabatic elimination formula for alpha assumes the
        buffer mode is in vacuum. When <b†b> is significant, the effective description
        breaks down and lifetime predictions become unreliable.
        Ref: Berdou et al. (2022), arXiv:2204.09128, Sec. II.A

      - Code space confinement: The code space is span{|C+>, |C->}. Single-photon
        loss (kappa_a) pushes population outside this manifold. Confinement = Tr(Pi_code * rho)
        where Pi_code = |C+><C+| + |C-><C-| projects onto the logical subspace.

      - Alpha stability margin: When |eps_2| ≈ kappa_a/4, alpha ≈ 0 — no cat state.
        The margin = (|eps_2| - kappa_a/4) / (kappa_a/4) measures distance from this
        threshold. Penalizing small margins favors drift-robust parameter regions.

    Parameters
    ----------
    params : CatQubitParams
        Fixed hardware parameters.
    t_probe_z, t_probe_x : float
        Probe times for T_Z, T_X measurement [us].
    target_bias : float
        Target eta = T_Z / T_X.
    w_lifetime, w_bias : float
        Weights on lifetime and bias terms (same as standard proxy).
    w_buffer : float
        Weight on buffer occupation penalty. 0 disables.
    w_confinement : float
        Weight on code space leakage penalty. 0 disables.
    w_margin : float
        Weight on alpha stability margin. 0 disables.
    margin_threshold : float
        Margin values below this trigger the penalty.

    Returns
    -------
    callable
        Reward function: x (shape (4,)) -> scalar. JIT-compiled when
        w_confinement == 0; non-JIT when confinement is active (vacuum
        settle needed for data-driven alpha).
    """
    a, b = build_operators(params)
    jump_ops = build_jump_ops(a, b, params)
    na = params.na
    nb = params.nb

    # Pre-build buffer number operator (closed over, not recomputed each call)
    n_b_op = b.dag() @ b

    # Parity operator for X_L measurement (alpha-independent)
    sx = (1j * jnp.pi * a.dag() @ a).expm()

    # Storage photon number (for data-driven alpha when confinement is active)
    n_a_op = a.dag() @ a

    # Pre-compute time grids
    if use_log_derivative:
        fracs = jnp.linspace(1.0 / n_log_deriv_points, 1.0, n_log_deriv_points)
        t_points_z = fracs * t_probe_z
        t_points_x = fracs * t_probe_x
        tsave_z_arr = jnp.concatenate([jnp.array([0.0]), t_points_z])
        tsave_x_arr = jnp.concatenate([jnp.array([0.0]), t_points_x])
    else:
        tsave_z_arr = jnp.array([0.0, t_probe_z])
        tsave_x_arr = jnp.array([0.0, t_probe_x])

    # Only save states when confinement is needed (partial trace requires them)
    _needs_states = w_confinement > 0
    # Confinement settling time for vacuum → cat (data-driven alpha)
    _t_settle_confinement = 10.0  # [μs], ~5-10× κ₂⁻¹

    def enhanced_proxy_reward(x, extra_H_override=None):
        g2_re, g2_im, eps_d_re, eps_d_im = x[0], x[1], x[2], x[3]

        # Build Hamiltonian
        H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
        h_pert = extra_H_override if extra_H_override is not None else extra_H
        if h_pert is not None:
            H = H + h_pert

        # NOTE: Z/X measurements use heuristic alpha (compute_alpha) for speed,
        # while the confinement projector uses vacuum-settled alpha for accuracy.
        # This inconsistency is intentional: vacuum settle is expensive, and the
        # Z/X measurements are less sensitive to alpha errors than the projector.
        # For maximum accuracy, use the "vacuum" reward type instead.
        alpha = compute_alpha(g2_re, g2_im, eps_d_re, eps_d_im, params)
        _, sz = build_logical_ops(a, b, alpha, params)

        # --- Z measurement: <Z_L>, plus buffer <n_b> ---
        g_state = dq.coherent(params.na, alpha)
        psi_z = dq.tensor(g_state, dq.fock(params.nb, 0))

        res_z = dq.mesolve(
            H,
            jump_ops,
            psi_z,
            tsave_z_arr,
            exp_ops=[sx, sz, n_b_op],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=_needs_states),
        )
        n_b = res_z.expects[2, -1].real  # <b†b> at final time

        # --- Code space confinement from the Z-measurement final state ---
        # When w_confinement > 0: use DATA-DRIVEN alpha from vacuum settle
        # (not the heuristic compute_alpha which is unreliable near threshold).
        # Cost: 1 extra mesolve call for vacuum settle.
        if _needs_states:
            rho_full = res_z.states[-1].to_jax()
            # Partial trace over buffer: assumes dq.tensor(storage, buffer) ordering,
            # so rho[i_s, j_b, k_s, l_b] = ⟨i_s,j_b|ρ|k_s,l_b⟩.
            rho_reshaped = jnp.reshape(rho_full, (na, nb, na, nb))
            rho_storage = jnp.trace(rho_reshaped, axis1=1, axis2=3)
            rho_storage = jnp.reshape(rho_storage, (na, na))

            # Data-driven alpha: vacuum settle → measure ⟨a†a⟩ → |α| = √⟨n⟩
            psi_vac = dq.tensor(dq.fock(na, 0), dq.fock(nb, 0))
            res_settle = dq.mesolve(
                H,
                jump_ops,
                psi_vac,
                jnp.array([0.0, _t_settle_confinement]),
                exp_ops=[n_a_op],
                method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
                options=dq.Options(progress_meter=False, save_states=False),
            )
            alpha_data = jnp.sqrt(jnp.maximum(res_settle.expects[0, -1].real, 0.01))

            alpha_safe = jnp.maximum(alpha_data, 1e-6)
            e_state = dq.coherent(na, -alpha_safe)
            g_state_na = dq.coherent(na, alpha_safe)
            cat_plus = dq.unit(g_state_na + e_state)
            cat_minus = dq.unit(g_state_na - e_state)
            pi_plus = (cat_plus @ cat_plus.dag()).to_jax()
            pi_minus = (cat_minus @ cat_minus.dag()).to_jax()
            pi_code = jnp.reshape(pi_plus, (na, na)) + jnp.reshape(pi_minus, (na, na))

            confinement = jnp.trace(pi_code @ rho_storage).real
        else:
            confinement = 1.0  # perfect confinement (no penalty)

        # --- X measurement: <X_L> from |+x> ---
        cat_plus_full = dq.unit(
            dq.coherent(params.na, alpha) + dq.coherent(params.na, -alpha)
        )
        psi_x = dq.tensor(cat_plus_full, dq.fock(params.nb, 0))

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

        # Lifetime score: log(T_Z) + log(T_X)
        lifetime_score = w_lifetime * (
            jnp.log(jnp.maximum(T_Z_est, 1e-6)) + jnp.log(jnp.maximum(T_X_est, 1e-6))
        )

        # Bias penalty
        bias_est = T_Z_est / jnp.maximum(T_X_est, 1e-6)
        bias_penalty = -w_bias * ((bias_est - target_bias) / target_bias) ** 2

        # --- Physics guardrail terms ---

        # Buffer occupation: penalize <n_b> > 0
        buffer_penalty = -w_buffer * n_b

        # Code space confinement: penalize leakage outside {|C+>, |C->}
        confinement_penalty = -w_confinement * (1.0 - confinement) ** 2

        # Alpha stability margin: penalize proximity to threshold
        g2 = g2_re + 1j * g2_im
        eps_d = eps_d_re + 1j * eps_d_im
        eps_2 = 2 * g2 * eps_d / params.kappa_b
        margin = (jnp.abs(eps_2) - params.kappa_a / 4) / (params.kappa_a / 4)
        margin_penalty = -w_margin * jnp.maximum(margin_threshold - margin, 0.0) ** 2

        return (
            lifetime_score
            + bias_penalty
            + buffer_penalty
            + confinement_penalty
            + margin_penalty
        )

    return enhanced_proxy_reward
