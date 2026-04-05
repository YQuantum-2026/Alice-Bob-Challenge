from __future__ import annotations

from collections.abc import Callable

import dynamiqs as dq
import jax.numpy as jnp
from jax import jit

from src.cat_qubit import (
    DEFAULT_PARAMS,
    CatQubitParams,
    build_hamiltonian,
    build_jump_ops,
    build_operators,
)

# ---------------------------------------------------------------------------
# Photon Number Proxy
# ---------------------------------------------------------------------------


def _build_photon_reward_fn(
    params: CatQubitParams = DEFAULT_PARAMS,
    n_target: float = 4.0,
    t_steady: float = 5.0,
    extra_H=None,
    **kw,
) -> Callable[[jnp.ndarray], float]:
    """Build a JIT-compiled photon-number proxy reward function.

    Simulates from vacuum |0⟩⊗|0⟩ and measures ⟨a†a⟩ at t_steady.
    R = -|⟨n⟩ - n_target|². Fastest reward: 1 mesolve call only.

    # Ref: Photon number as cat size proxy, ⟨n⟩ ≈ |α|². Challenge notebook Sec. 2.

    Parameters
    ----------
    params : CatQubitParams
        Fixed hardware parameters.
    n_target : float
        Target mean photon number (≈ |alpha_target|²).
    t_steady : float
        Time to reach steady state [us].
    extra_H : QArray or None
        Static Hamiltonian perturbation (closed over, JIT-compatible).

    Returns
    -------
    callable
        JIT-compiled reward function: x (shape (4,)) -> scalar.
        Accepts optional extra_H_override kwarg for runtime drift.
    """
    if kw:
        import warnings

        warnings.warn(
            f"_build_photon_reward_fn: unused kwargs {list(kw.keys())}",
            stacklevel=2,
        )
    a, b = build_operators(params)
    jump_ops = build_jump_ops(a, b, params)
    # Number operator: a†a (full tensor product space)
    n_op = a.dag() @ a

    @jit
    def photon_reward(x, extra_H_override=None):
        g2_re, g2_im, eps_d_re, eps_d_im = x[0], x[1], x[2], x[3]

        H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
        h_pert = extra_H_override if extra_H_override is not None else extra_H
        if h_pert is not None:
            H = H + h_pert

        # Initial state: vacuum |0⟩⊗|0⟩
        psi0 = dq.tensor(dq.fock(params.na, 0), dq.fock(params.nb, 0))

        tsave = jnp.array([0.0, t_steady])
        res = dq.mesolve(
            H,
            jump_ops,
            psi0,
            tsave,
            exp_ops=[n_op],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )
        n_mean = res.expects[0, -1].real  # ⟨a†a⟩ at t_steady

        # R = -|⟨n⟩ - n_target|²
        return -((n_mean - n_target) ** 2)

    return photon_reward


# ---------------------------------------------------------------------------
# Fidelity-Based Reward
# ---------------------------------------------------------------------------


def _build_fidelity_reward_fn(
    params: CatQubitParams = DEFAULT_PARAMS,
    t_steady: float = 5.0,
    extra_H=None,
    **kw,
) -> Callable[[jnp.ndarray], float]:
    """Build a fidelity-based reward function using vacuum-based state prep.

    Simulates from vacuum, computes fidelity of the storage mode with the
    target even cat state |C_+⟩ = N(|α⟩ + |−α⟩).

    Alpha-free: uses data-driven α estimated from ⟨a†a⟩ of the settled
    vacuum state instead of the heuristic adiabatic elimination formula
    (compute_alpha). This matches the experimental protocol.

    Ref: State fidelity F(ρ, σ) = Tr(ρσ) for pure target. Challenge notebook.
    Ref: Réglade et al. (2024), Nature 629 — vacuum-based cat characterization.

    Parameters
    ----------
    params : CatQubitParams
        Fixed hardware parameters.
    t_steady : float
        Time to reach steady state [us].

    Returns
    -------
    callable
        Reward function: x (shape (4,)) -> scalar. NOT JIT-compiled
        (data-driven alpha computation between steps).
    """
    if kw:
        import warnings

        warnings.warn(
            f"_build_fidelity_reward_fn: unused kwargs {list(kw.keys())}",
            stacklevel=2,
        )
    a, b = build_operators(params)
    jump_ops = build_jump_ops(a, b, params)
    na = params.na
    nb = params.nb
    n_a_op = a.dag() @ a  # photon number for data-driven alpha

    def fidelity_reward(x, extra_H_override=None):
        g2_re, g2_im, eps_d_re, eps_d_im = x[0], x[1], x[2], x[3]

        H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
        # Add Hamiltonian drift perturbation (detuning, Kerr) if provided
        h_pert = extra_H_override if extra_H_override is not None else extra_H
        if h_pert is not None:
            H = H + h_pert

        # Initial state: vacuum |0⟩⊗|0⟩
        psi0 = dq.tensor(dq.fock(na, 0), dq.fock(nb, 0))

        tsave = jnp.array([0.0, t_steady])
        res = dq.mesolve(
            H,
            jump_ops,
            psi0,
            tsave,
            exp_ops=[n_a_op],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=True),
        )

        # Data-driven α: |α_est| = √⟨a†a⟩ at steady state
        # This is a measurement, not a heuristic — the simulation tells us
        # the actual cat size for these specific control parameters.
        n_a_settled = res.expects[0, -1].real
        alpha = jnp.sqrt(jnp.maximum(n_a_settled, 0.01))

        # Final density matrix: shape (na*nb, na*nb)
        rho_full = res.states[-1].to_jax()

        # Partial trace over buffer: assumes dq.tensor(storage, buffer) ordering,
        # so rho[i_s, j_b, k_s, l_b] = ⟨i_s,j_b|ρ|k_s,l_b⟩. Tracing axis1=1,
        # axis2=3 sums over the buffer indices to yield the storage density matrix.
        rho_reshaped = jnp.reshape(rho_full, (na, nb, na, nb))
        rho_storage = jnp.trace(rho_reshaped, axis1=1, axis2=3)  # (na, na)

        # Build target even cat state: |C_+⟩ = N(|α⟩ + |−α⟩)
        # Uses data-driven alpha (not compute_alpha heuristic)
        cat_ket = dq.unit(dq.coherent(na, alpha) + dq.coherent(na, -alpha))
        target_dm = (cat_ket @ cat_ket.dag()).to_jax()
        target_dm = jnp.reshape(target_dm, (na, na))
        rho_storage = jnp.reshape(rho_storage, (na, na))

        # Fidelity: F = Tr(target_dm @ rho_storage) for pure target
        fidelity = jnp.trace(target_dm @ rho_storage).real

        return fidelity

    return fidelity_reward
