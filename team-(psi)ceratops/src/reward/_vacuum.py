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
# Vacuum-based (alpha-free) reward
# ---------------------------------------------------------------------------


def _build_vacuum_reward_fn(
    params: CatQubitParams = DEFAULT_PARAMS,
    t_settle: float = 15.0,
    t_measure_z: float = 200.0,
    t_measure_x: float = 1.0,
    n_measure_points: int = 5,
    target_bias: float = 100.0,
    w_lifetime: float = 1.0,
    w_bias: float = 0.5,
    extra_H=None,
) -> Callable[[jnp.ndarray], float]:
    """Build an alpha-free reward using vacuum-based state preparation.

    **Performance note**: This reward is NOT JIT-compatible (dynamic alpha
    computation between mesolve calls). Batching uses a Python loop, making
    it ~N× slower than vmap'd proxy rewards (e.g. ~24× for population_size=24).
    This is acceptable for CMA-ES evaluation but NOT for gradient-based methods.

    Matches the experimental protocol of Alice & Bob: cat states emerge
    naturally from dissipative dynamics without knowing α. The system
    finds its own steady states from vacuum.

    T_X (phase-flip): Vacuum |0⟩ has even parity. Under two-photon
    dissipation, parity is conserved, so vacuum evolves to the even cat
    |C₊⟩ = |+x⟩. Parity P = exp(iπa†a) serves as logical X (α-independent).
    T_X is extracted from the parity decay rate after settling.

    T_Z (bit-flip): Uses data-driven α estimated from ⟨a†a⟩ of the
    settled vacuum state. This is a MEASUREMENT, not the heuristic
    adiabatic elimination formula — the simulation tells us the actual
    cat size. A coherent state |α_est⟩ is then prepared and its
    well-aligned quadrature ⟨Q_θ⟩ decay measured. Q_θ = a·e^{-iθ} +
    a†·e^{iθ} where θ = arg(g2·eps_d)/2 is the exact well orientation
    (not a heuristic — only |α| magnitude is approximate).

    Note: |α_est⟩ is a coherent state, not the exact Lindbladian steady
    state (which is a cat superposition). For |α|² >> 1 the difference is
    exponentially small; for small α during early optimization, there may
    be a measurable discrepancy.

    2 mesolve calls per evaluation. NOT JIT-compatible (dynamic alpha
    computation between calls). Fine for CMA-ES.

    Ref: Réglade et al. "Quantum control of a cat-qubit with bit-flip
      times exceeding ten seconds." Nature 629, 778-783 (2024).
      arXiv:2307.06617.
    Ref: Berdou et al. "One hundred second bit-flip time in a two-photon
      dissipative oscillator." (2022). arXiv:2204.09128.
    Ref: Marquet et al. "Preserving phase coherence and linearity in cat
      qubits with exponential bit-flip suppression." Phys. Rev. X 15,
      011070 (2025). arXiv:2409.17556.

    Parameters
    ----------
    params : CatQubitParams
        Fixed hardware parameters (na, nb, kappa_a, kappa_b).
    t_settle : float
        Settling time for vacuum → cat/well [μs]. Should be ~5-10× κ₂⁻¹.
    t_measure_z : float
        T_Z measurement window after settling [μs].
    t_measure_x : float
        T_X measurement window after settling [μs].
    n_measure_points : int
        Number of time points for log-derivative fit.
    target_bias : float
        Target η = T_Z / T_X.
    w_lifetime : float
        Weight on lifetime maximization component.
    w_bias : float
        Weight on bias targeting component.

    Returns
    -------
    callable
        Reward function: x (shape (4,)) -> scalar.
    """
    a, b = build_operators(params)
    jump_ops = build_jump_ops(a, b, params)
    na, nb = params.na, params.nb

    # Parity operator: P = exp(iπ a†a) — α-independent logical X
    # Ref: Parity as logical X. Berdou et al. (2022), arXiv:2204.09128.
    parity_op = (1j * jnp.pi * a.dag() @ a).expm()

    # Storage photon number operator (for data-driven α estimation)
    n_a_op = a.dag() @ a

    # Vacuum initial state: |0⟩_storage ⊗ |0⟩_buffer
    psi_vacuum = dq.tensor(dq.fock(na, 0), dq.fock(nb, 0))

    # --- T_X time grid: single mesolve from 0 to t_settle + t_measure_x ---
    # Also measures ⟨a†a⟩ for data-driven α estimation.
    # Include settling checkpoint + measurement points
    fracs_x = jnp.linspace(1.0 / n_measure_points, 1.0, n_measure_points)
    t_measure_pts_x = t_settle + fracs_x * t_measure_x
    tsave_x = jnp.concatenate(
        [
            jnp.array([0.0, t_settle]),  # t=0, settling checkpoint
            t_measure_pts_x,  # post-settling measurement points
        ]
    )
    # Relative times for log-derivative fit (time since settling)
    t_rel_x = fracs_x * t_measure_x

    # --- T_Z time grid: single mesolve from |α_est⟩ ---
    # No settling needed — coherent state is already in one well.
    fracs_z = jnp.linspace(1.0 / n_measure_points, 1.0, n_measure_points)
    t_measure_pts_z = fracs_z * t_measure_z
    tsave_z = jnp.concatenate([jnp.array([0.0]), t_measure_pts_z])
    t_rel_z = fracs_z * t_measure_z

    def vacuum_reward(x, extra_H_override=None):
        g2_re, g2_im, eps_d_re, eps_d_im = x[0], x[1], x[2], x[3]

        # Build base Hamiltonian (no heuristic alpha)
        H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
        # Add Hamiltonian drift perturbation (detuning, Kerr) if provided
        h_pert = extra_H_override if extra_H_override is not None else extra_H
        if h_pert is not None:
            H = H + h_pert

        # --- Well-aligned quadrature operator ---
        # θ = arg(g2 · eps_d) / 2 is the exact well orientation in phase space.
        # This is NOT a heuristic: the drive phase determines the well axis.
        # Only |α| (magnitude) is approximate under adiabatic elimination.
        # When g2 and eps_d are both real: θ=0, Q_θ = a + a† (x-quadrature).
        g2 = g2_re + 1j * g2_im
        eps_d = eps_d_re + 1j * eps_d_im
        theta = jnp.angle(g2 * eps_d) / 2.0

        # ===============================================================
        # T_X: phase-flip time from parity decay + data-driven α
        #
        # Vacuum (even parity) → even cat |C₊⟩ = |+x⟩ under two-photon
        # dissipation. Parity conserved by pair processes; single-photon
        # loss flips parity → phase flips.
        #
        # We also measure ⟨a†a⟩ at settling time to get a DATA-DRIVEN
        # estimate of |α|. This is NOT the heuristic adiabatic elimination
        # formula — it's measuring the cat size from the actual dynamics,
        # exactly as done experimentally via photon number calibration.
        #
        # Ref: Réglade et al. (2024), Nature 629, Sec. "Phase-flip time"
        # Ref: Marquet et al. (2025), Phys. Rev. X 15, 011070,
        #   Sec. "Cat qubit characterization"
        # ===============================================================
        res_x = dq.mesolve(
            H,
            jump_ops,
            psi_vacuum,
            tsave_x,
            exp_ops=[parity_op, n_a_op],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )
        # Post-settling parity values (skip t=0 and t_settle)
        parity_vals = jnp.abs(res_x.expects[0, 2:].real)
        T_X_est = _estimate_T_from_log_derivative(t_rel_x, parity_vals)

        # Data-driven α: |α_est| = √⟨a†a⟩ at settling time
        # This is a measurement, not a heuristic — the simulation tells us
        # the actual cat size for these specific control parameters.
        n_a_settled = res_x.expects[1, 1].real  # ⟨a†a⟩ at t_settle
        alpha_mag = jnp.sqrt(jnp.maximum(n_a_settled, 0.01))

        # Full complex α: magnitude from measurement, phase from drive (exact)
        alpha_est = alpha_mag * jnp.exp(1j * theta)

        # ===============================================================
        # T_Z: bit-flip time from quadrature decay
        #
        # Prepare |α_est⟩ (coherent state in one well, using data-driven α)
        # and measure ⟨Q_θ⟩ decay. The master equation ensemble average of
        # ⟨a⟩ decays as exp(-t/T_Z) as bit-flips mix both wells.
        #
        # ⟨Q_θ⟩(t) = 2|α|·exp(-t/T_Z), and the log-derivative slope
        # gives -1/T_Z regardless of |α| (amplitude-invariant).
        #
        # No settling needed: |α_est⟩ is already localized in one well.
        #
        # Ref: Berdou et al. (2022), arXiv:2204.09128, Sec. "Bit-flip
        #   measurement via fluorescence detection"
        # Ref: Réglade et al. (2024), Nature 629, Sec. "Bit-flip time"
        # ===============================================================
        phase_factor = jnp.exp(-1j * theta)
        Q_theta = a * phase_factor + a.dag() * jnp.conj(phase_factor)

        psi_alpha = dq.tensor(dq.coherent(na, alpha_est), dq.fock(nb, 0))
        res_z = dq.mesolve(
            H,
            jump_ops,
            psi_alpha,
            tsave_z,
            exp_ops=[Q_theta],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )
        # Quadrature decay (skip t=0 to avoid initial transients)
        quad_vals = jnp.abs(res_z.expects[0, 1:].real)
        T_Z_est = _estimate_T_from_log_derivative(t_rel_z, quad_vals)

        # ===============================================================
        # Composite reward: log(T_Z) + log(T_X) - bias penalty
        # ===============================================================
        return _compute_lifetime_score(
            T_Z_est, T_X_est, target_bias, w_lifetime, w_bias
        )

    return vacuum_reward
