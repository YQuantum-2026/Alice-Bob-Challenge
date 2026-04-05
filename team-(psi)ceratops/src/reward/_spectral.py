from __future__ import annotations

import warnings
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
from src.reward._helpers import _compute_lifetime_score

# ---------------------------------------------------------------------------
# Reward F: Liouvillian Spectral Gap (fastest, no time evolution)
# ---------------------------------------------------------------------------


def _classify_eigenmodes_by_parity(
    eigvecs: jnp.ndarray,
    sorted_idx: jnp.ndarray,
    a: dq.QArray,
    n: int,
) -> tuple[float, float]:
    """Classify the two slowest Lindbladian eigenmodes using the parity operator.

    For cat qubits, the bit-flip (T_Z) decay mode has a strong parity
    signature because it involves transitions between even and odd cat
    states. The phase-flip (T_X) mode involves coherence decay and has
    a weaker parity signature.

    We compute Tr(P @ rho_k) for each eigenmode k, where P = exp(i*pi*a†a)
    is the parity operator and rho_k is the eigenvector reshaped as a matrix.
    The mode with larger |Tr(P @ rho_k)| is the bit-flip mode (T_Z).

    Parameters
    ----------
    eigvecs : jnp.ndarray
        Eigenvectors of the Lindbladian (n^2 x n^2 matrix).
    sorted_idx : jnp.ndarray
        Indices that sort eigenvalues by descending real part.
    a : dq.QArray
        Storage annihilation operator (tensor product with buffer identity).
    n : int
        Physical Hilbert space dimension (na * nb).

    Returns
    -------
    parity_1 : float
        |Tr(P @ rho_1)| for the first slowest mode.
    parity_2 : float
        |Tr(P @ rho_2)| for the second slowest mode.
    """
    # Parity operator: P = exp(i*pi*a†a) in the full storage-buffer space
    P = (1j * jnp.pi * a.dag() @ a).expm()
    P_dense = P.to_jax()

    # Extract the two slowest-decaying eigenvectors (indices 1, 2 after sorting)
    v1 = eigvecs[:, sorted_idx[1]].reshape(n, n)
    v2 = eigvecs[:, sorted_idx[2]].reshape(n, n)

    # Tr(P @ rho_k) gives the parity signature of each eigenmode
    parity_1 = float(jnp.abs(jnp.trace(P_dense @ v1)))
    parity_2 = float(jnp.abs(jnp.trace(P_dense @ v2)))

    return parity_1, parity_2


def _build_spectral_reward_fn(
    params: CatQubitParams = DEFAULT_PARAMS,
    target_bias: float = 100.0,
    w_lifetime: float = 1.0,
    w_bias: float = 0.5,
    extra_H=None,
    **kw,
) -> Callable[[jnp.ndarray], float]:
    """Build a reward based on Liouvillian eigenvalue decomposition.

    Instead of time-evolving and fitting decays, compute lifetimes directly
    from the spectral gap of the Lindbladian superoperator L:

      T_slow = 1 / |Re(lambda_1)|

    where lambda_1 is the eigenvalue with the smallest non-zero real part.

    The Lindbladian eigenvalues encode ALL decay timescales of the system.
    The two slowest (closest to zero) correspond to T_Z (bit-flip) and
    T_X (phase-flip).

    Eigenmode classification uses the parity operator P = exp(i*pi*a†a):
    the mode with larger |Tr(P @ rho_k)| is the bit-flip mode (T_Z),
    because bit-flip transitions involve parity changes. Falls back to
    max/min heuristic when parity signals are ambiguous.

    R = log(T_Z) + log(T_X) - lambda * ((T_Z/T_X - eta) / eta)^2

    Ref: arXiv:2511.13308 — "switching rate = dissipative gap = -Re(lambda_1)"
    Ref: Berdou et al. (2022), arXiv:2204.09128 — Lindbladian for cat qubits.

    Advantages:
      - No mesolve (no ODE integration)
      - No curve fitting
      - Exact lifetimes (not proxy estimates)
      - Single matrix eigendecomposition

    Limitations:
      - O(n^4) eigendecomposition cost (n = na*nb)
      - Not JIT-compatible with current JAX eigvals for complex non-symmetric matrices
      - Best for small-to-medium Hilbert spaces (na*nb <= 75)

    Parameters
    ----------
    params : CatQubitParams
        Fixed hardware parameters.
    target_bias : float
        Target eta = T_Z / T_X.
    w_lifetime : float
        Weight on lifetime component.
    w_bias : float
        Weight on bias penalty.

    Returns
    -------
    callable
        Reward function: x (shape (4,)) -> scalar. NOT JIT-compiled (uses eig).
    """
    if kw:
        warnings.warn(
            f"_build_spectral_reward_fn: unused kwargs {list(kw.keys())}",
            stacklevel=2,
        )
    a, b = build_operators(params)
    jump_ops = build_jump_ops(a, b, params)
    n = params.na * params.nb

    def spectral_reward(x, extra_H_override=None):
        g2_re, g2_im, eps_d_re, eps_d_im = x[0], x[1], x[2], x[3]

        H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
        # Add Hamiltonian drift perturbation (detuning, Kerr) if provided
        h_pert = extra_H_override if extra_H_override is not None else extra_H
        if h_pert is not None:
            H = H + h_pert

        # Build Lindbladian superoperator (n^2 x n^2 matrix)
        # L rho = -i[H, rho] + sum_k D[L_k](rho)
        L = dq.slindbladian(H, jump_ops)
        L_dense = L.to_jax()

        # Full eigendecomposition to get both eigenvalues and eigenvectors
        eigvals, eigvecs = jnp.linalg.eig(L_dense)
        real_parts = eigvals.real

        # Sort by real part (descending: 0 is largest, then negative)
        sorted_idx = jnp.argsort(-real_parts)
        sorted_real = real_parts[sorted_idx]

        # lambda_0 ≈ 0 (steady state), lambda_1 = slowest decay, lambda_2 = next
        gap_1 = jnp.maximum(jnp.abs(sorted_real[1]), 1e-10)
        gap_2 = jnp.maximum(jnp.abs(sorted_real[2]), 1e-10)

        T_1 = 1.0 / gap_1
        T_2 = 1.0 / gap_2

        # Parity-based eigenmode classification: the mode with larger
        # parity signature is the bit-flip (T_Z) mode, because bit-flip
        # involves transitions between even and odd cat states.
        parity_1, parity_2 = _classify_eigenmodes_by_parity(eigvecs, sorted_idx, a, n)

        # Use parity classification when signals are distinguishable,
        # fall back to max/min heuristic otherwise
        parity_ratio = max(parity_1, parity_2) / max(min(parity_1, parity_2), 1e-15)
        if parity_ratio > 1.5:
            # Clear parity separation — use classification
            if parity_1 > parity_2:
                T_Z_est = T_1  # mode 1 is bit-flip
                T_X_est = T_2
            else:
                T_Z_est = T_2  # mode 2 is bit-flip
                T_X_est = T_1
        else:
            # Ambiguous parity — fall back to max/min heuristic
            T_Z_est = jnp.maximum(T_1, T_2)
            T_X_est = jnp.minimum(T_1, T_2)
            warnings.warn(
                f"Spectral reward: parity signals ambiguous "
                f"(|Tr(P@rho_1)|={parity_1:.3g}, |Tr(P@rho_2)|={parity_2:.3g}, "
                f"ratio={parity_ratio:.2f}). "
                "Falling back to T_Z=max, T_X=min heuristic.",
                stacklevel=2,
            )

        return _compute_lifetime_score(
            T_Z_est, T_X_est, target_bias, w_lifetime, w_bias
        )

    return spectral_reward
