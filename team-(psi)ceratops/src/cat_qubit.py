"""Core cat qubit simulation module.

Provides:
  - Operator construction for the storage-buffer system
  - Hamiltonian building with complex g2 and eps_d control knobs
  - Cat size (alpha) estimation from effective two-photon parameters
  - Lindblad simulation of logical decay (T_X, T_Z measurement)
  - Robust exponential fitting for lifetime extraction

Physics:
  H/hbar = conj(g2) a^2 b† + g2 (a†)^2 b - eps_d b† - conj(eps_d) b
  L_b = sqrt(kappa_b) * b,  L_a = sqrt(kappa_a) * a
  Effective: kappa_2 = 4|g2|^2 / kappa_b,  eps_2 = 2*g2*eps_d / kappa_b
  alpha = sqrt(2/kappa_2 * (eps_2 - kappa_a/4))

Reference:
  Berdou et al. "One hundred second bit-flip time in a two-photon
  dissipative oscillator." PRX Quantum 4, 020350 (2023). arXiv:2204.09128

Additional references:
  Sivak et al. "Real-time quantum error correction beyond break-even."
  Nature (2023). arXiv:2211.09116
  Sivak et al. "Reinforcement Learning Control of QEC."
  arXiv:2511.08493 (2025).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

warnings.filterwarnings(
    "ignore", message=".*SparseDIAQArray.*converted to a DenseQArray.*"
)

# ---------------------------------------------------------------------------
# System configuration
# ---------------------------------------------------------------------------


@dataclass
class CatQubitParams:
    """Fixed hardware parameters (not tuned by the online optimizer).

    Parameters
    ----------
    na : int
        Storage mode Hilbert space truncation dimension.
    nb : int
        Buffer mode Hilbert space truncation dimension.
    kappa_b : float
        Buffer single-photon loss rate [MHz].
    kappa_a : float
        Storage single-photon loss rate [MHz].
    use_double : bool
        If True, ``run.py`` calls ``dq.set_precision("double")`` at startup
        for float64/complex128 arithmetic. Default: False.
    """

    na: int = 15
    nb: int = 5
    kappa_b: float = 10.0
    kappa_a: float = 1.0
    use_double: bool = (
        False  # True → float64/complex128 (better precision, slower on GPU)
    )


DEFAULT_PARAMS = CatQubitParams()


# ---------------------------------------------------------------------------
# Operator construction
# ---------------------------------------------------------------------------


def build_operators(params: CatQubitParams = DEFAULT_PARAMS):
    """Construct tensor-product annihilation operators for the storage-buffer system.

    Parameters
    ----------
    params : CatQubitParams
        System dimensions.

    Returns
    -------
    a : QArray
        Storage mode annihilation operator (na*nb x na*nb).
    b : QArray
        Buffer mode annihilation operator (na*nb x na*nb).
    """
    a = dq.tensor(dq.destroy(params.na), dq.eye(params.nb))
    b = dq.tensor(dq.eye(params.na), dq.destroy(params.nb))
    return a, b


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------


def build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im):
    """Build the two-photon exchange + buffer drive Hamiltonian.

    H = conj(g2) a^2 b† + g2 (a†)^2 b - eps_d b† - conj(eps_d) b

    Parameters
    ----------
    a, b : QArray
        Storage and buffer annihilation operators.
    g2_re, g2_im : float
        Real and imaginary parts of the two-photon coupling g2.
    eps_d_re, eps_d_im : float
        Real and imaginary parts of the buffer drive amplitude eps_d.

    Returns
    -------
    H : QArray
        System Hamiltonian.
    """
    g2 = g2_re + 1j * g2_im
    eps_d = eps_d_re + 1j * eps_d_im

    # Ref: Berdou et al. (2022), arXiv:2204.09128, Eq. (1) — two-photon exchange Hamiltonian
    # conj(g2) * (a @ a) @ b† + g2 * (a† @ a†) @ b - eps_d * b† - conj(eps_d) * b
    H = (
        jnp.conj(g2) * a @ a @ b.dag()
        + g2 * a.dag() @ a.dag() @ b
        - eps_d * b.dag()
        - jnp.conj(eps_d) * b
    )
    return H


# ---------------------------------------------------------------------------
# Jump operators (Lindblad dissipation)
# ---------------------------------------------------------------------------


def build_jump_ops(a, b, params: CatQubitParams = DEFAULT_PARAMS):
    """Build Lindblad jump operators for buffer and storage loss.

    Parameters
    ----------
    a, b : QArray
        Storage and buffer annihilation operators.
    params : CatQubitParams
        Contains kappa_b and kappa_a.

    Returns
    -------
    list[QArray]
        [L_b, L_a] jump operators with rates absorbed.
    """
    # Ref: Lindblad jump operators. Berdou et al. (2022), arXiv:2204.09128, Sec. II
    L_b = jnp.sqrt(params.kappa_b) * b
    L_a = jnp.sqrt(params.kappa_a) * a
    return [L_b, L_a]


# ---------------------------------------------------------------------------
# TLS extension (two-level system defect coupling)
# ---------------------------------------------------------------------------


def build_tls_operators(params: CatQubitParams = DEFAULT_PARAMS):
    """Construct operators in the TLS-extended Hilbert space.

    Extends the storage-buffer system by tensoring with a 2-level system:
      storage(na) ⊗ buffer(nb) ⊗ TLS(2)  →  total dim = na*nb*2

    Parameters
    ----------
    params : CatQubitParams
        System dimensions (na, nb for storage and buffer).

    Returns
    -------
    a_ext : QArray
        Storage annihilation operator in extended space (na*nb*2 x na*nb*2).
    b_ext : QArray
        Buffer annihilation operator in extended space.
    sigma_z : QArray
        TLS sigma_z operator in extended space.
    sigma_m : QArray
        TLS sigma_minus (lowering) operator in extended space.
    """
    # Standard cat qubit operators tensored with TLS identity
    a_ext = dq.tensor(dq.destroy(params.na), dq.eye(params.nb), dq.eye(2))
    b_ext = dq.tensor(dq.eye(params.na), dq.destroy(params.nb), dq.eye(2))

    # TLS operators tensored with cat qubit identity
    sigma_z = dq.tensor(dq.eye(params.na), dq.eye(params.nb), dq.sigmaz())
    sigma_m = dq.tensor(dq.eye(params.na), dq.eye(params.nb), dq.sigmam())

    return a_ext, b_ext, sigma_z, sigma_m


def build_tls_hamiltonian_term(a_ext, sigma_z, sigma_m, g_tls, omega_tls=0.0):
    """Build the TLS coupling Hamiltonian term.

    H_TLS = omega_tls * sigma_z / 2 + g_tls * (a† sigma_- + a sigma_+)

    The Jaynes-Cummings interaction couples the storage mode to the
    TLS defect, causing energy exchange and decoherence.

    Parameters
    ----------
    a_ext : QArray
        Storage annihilation operator in extended space.
    sigma_z : QArray
        TLS sigma_z in extended space.
    sigma_m : QArray
        TLS sigma_minus in extended space.
    g_tls : float
        TLS-storage coupling strength [MHz].
    omega_tls : float
        TLS detuning from storage frequency [MHz]. 0 = resonant.

    Returns
    -------
    H_tls : QArray
        TLS Hamiltonian term.

    Reference
    ---------
    Challenge notebook, "Drift and Noise Modeling" — TLS defect coupling.
    """
    sigma_p = sigma_m.dag()
    H_tls = omega_tls / 2 * sigma_z + g_tls * (a_ext.dag() @ sigma_m + a_ext @ sigma_p)
    return H_tls


def build_tls_jump_ops(a_ext, b_ext, sigma_m, params: CatQubitParams, gamma_tls: float):
    """Build jump operators in the TLS-extended space.

    Returns the standard buffer + storage loss operators extended to the
    TLS space, plus the TLS decay operator sqrt(gamma_tls) * sigma_-.

    Parameters
    ----------
    a_ext, b_ext : QArray
        Storage and buffer operators in extended space.
    sigma_m : QArray
        TLS sigma_minus in extended space.
    params : CatQubitParams
        Contains kappa_b and kappa_a.
    gamma_tls : float
        TLS decay rate [MHz].

    Returns
    -------
    list[QArray]
        [L_b, L_a, L_tls] jump operators with rates absorbed.
    """
    L_b = jnp.sqrt(params.kappa_b) * b_ext
    L_a = jnp.sqrt(params.kappa_a) * a_ext
    L_tls = jnp.sqrt(gamma_tls) * sigma_m
    return [L_b, L_a, L_tls]


# ---------------------------------------------------------------------------
# Cat size estimation
# ---------------------------------------------------------------------------


def compute_alpha(
    g2_re, g2_im, eps_d_re, eps_d_im, params: CatQubitParams = DEFAULT_PARAMS
):
    """Estimate the stabilized cat size alpha from control parameters.

    Uses the adiabatic elimination result:
      kappa_2 = 4|g2|^2 / kappa_b
      eps_2   = 2 * g2 * eps_d / kappa_b   (complex-valued)
      alpha   = sqrt(2/kappa_2 * (|eps_2| - kappa_a/4))

    Note: eps_2 is complex (product of complex g2 and eps_d).
    The magnitude |eps_2| is taken to give a real-valued cat size.

    Parameters
    ----------
    g2_re, g2_im : float
        Real/imag parts of g2.
    eps_d_re, eps_d_im : float
        Real/imag parts of eps_d.
    params : CatQubitParams
        Fixed hardware parameters.

    Returns
    -------
    float
        Estimated cat size |alpha|.

    Alpha-free compliance note
    --------------------------
    This heuristic is used by: proxy, enhanced_proxy, multipoint rewards.
    These are fast/JIT-compatible but approximate (ignores squeezing λ, breaks
    near threshold |ε₂| ≈ κ_a/4).

    Alpha-free rewards (vacuum, parity, fidelity, spectral) estimate α from
    actual ⟨a†a⟩ via vacuum-based state prep instead. The default reward is
    "vacuum" (alpha-free), matching Alice & Bob's experimental protocol
    (Réglade et al. 2024).
    """
    g2 = g2_re + 1j * g2_im
    eps_d = eps_d_re + 1j * eps_d_im

    # Ref: Adiabatic elimination result. Berdou et al. (2022), arXiv:2204.09128, Eq. (2)-(4)
    kappa_2 = 4 * jnp.abs(g2) ** 2 / params.kappa_b
    eps_2 = 2 * g2 * eps_d / params.kappa_b

    # Guard against division by zero when g2=0 (kappa_2=0)
    kappa_2_safe = jnp.maximum(kappa_2, 1e-12)
    # Guard against negative sqrt argument (|eps_2| < kappa_a/4)
    arg = jnp.maximum(2 / kappa_2_safe * (jnp.abs(eps_2) - params.kappa_a / 4), 0.0)
    return jnp.sqrt(arg)


# ---------------------------------------------------------------------------
# Logical operators
# ---------------------------------------------------------------------------


def build_logical_ops(a, b, alpha, params: CatQubitParams = DEFAULT_PARAMS):
    """Build logical X (parity) and Z operators for the cat qubit.

    Logical X = parity operator exp(i*pi*a†a), tensored with identity on buffer.
    Logical Z = |alpha><alpha| - |-alpha><-alpha|, tensored with identity on buffer.

    Note: The logical Z uses coherent-state projectors, which are approximate
    logical operators valid when |α|² >> 1 (overlap ⟨α|−α⟩ = exp(−2|α|²) ≈ 0).
    For |α| < 1, expectation values may not reach ±1, degrading lifetime
    estimation accuracy. This is the standard construction (Berdou et al. 2022)
    and is exact in the regime the optimizer converges to (α > 1.5).

    Parameters
    ----------
    a, b : QArray
        Storage and buffer annihilation operators (for dimension reference).
    alpha : float
        Estimated cat size.
    params : CatQubitParams
        System dimensions.

    Returns
    -------
    sx : QArray
        Logical X (parity) operator.
    sz : QArray
        Logical Z (coherent state projector difference).
    """
    # Ref: Parity as logical X. Tutorial: Introduction to Cats, Sec. "Cat State Definitions"
    # Parity on storage: exp(i*pi*a_local†*a_local)
    sx = (1j * jnp.pi * a.dag() @ a).expm()

    # Ref: Coherent state projectors as logical Z. Berdou et al. (2022)
    # Logical Z: |+alpha><+alpha| - |-alpha><-alpha| on storage, tensored with I_b
    plus_alpha = dq.coherent(params.na, alpha)
    minus_alpha = dq.coherent(params.na, -alpha)
    sz_local = plus_alpha @ plus_alpha.dag() - minus_alpha @ minus_alpha.dag()
    sz = dq.tensor(sz_local, dq.eye(params.nb))

    return sx, sz


# ---------------------------------------------------------------------------
# Initial states
# ---------------------------------------------------------------------------


def build_initial_states(alpha, params: CatQubitParams = DEFAULT_PARAMS):
    """Build logical basis states for lifetime measurement.

    Parameters
    ----------
    alpha : float
        Estimated cat size.
    params : CatQubitParams
        System dimensions.

    Returns
    -------
    dict[str, QArray]
        Mapping from label ("+z", "-z", "+x", "-x") to initial state
        (storage ⊗ buffer vacuum).
    """
    g_state = dq.coherent(params.na, alpha)  # |+z> ≈ |alpha>
    e_state = dq.coherent(params.na, -alpha)  # |-z> ≈ |-alpha>

    states = {
        "+z": g_state,
        "-z": e_state,
        "+x": dq.unit(g_state + e_state),  # even cat
        "-x": dq.unit(g_state - e_state),  # odd cat
    }

    # Tensor with buffer vacuum
    vac_b = dq.fock(params.nb, 0)
    return {k: dq.tensor(v, vac_b) for k, v in states.items()}


# ---------------------------------------------------------------------------
# Simulation: measure lifetime
# ---------------------------------------------------------------------------


def simulate_lifetime(
    g2_re,
    g2_im,
    eps_d_re,
    eps_d_im,
    initial_state_label: str,
    tfinal: float,
    npoints: int = 100,
    params: CatQubitParams = DEFAULT_PARAMS,
):
    """Run Lindblad simulation and return expectation value traces.

    Parameters
    ----------
    g2_re, g2_im, eps_d_re, eps_d_im : float
        Control knobs.
    initial_state_label : str
        One of "+z", "-z", "+x", "-x".
    tfinal : float
        Total simulation time [us].
    npoints : int
        Number of save points.
    params : CatQubitParams
        Fixed hardware parameters.

    Returns
    -------
    tsave : Array
        Time points.
    exp_x : Array
        Expectation values of logical X (parity).
    exp_z : Array
        Expectation values of logical Z.
    """
    a, b = build_operators(params)
    H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
    jump_ops = build_jump_ops(a, b, params)

    alpha = compute_alpha(g2_re, g2_im, eps_d_re, eps_d_im, params)
    sx, sz = build_logical_ops(a, b, alpha, params)

    init_states = build_initial_states(alpha, params)
    psi0 = init_states[initial_state_label]

    tsave = jnp.linspace(0, tfinal, npoints)

    result = dq.mesolve(
        H,
        jump_ops,
        psi0,
        tsave,
        exp_ops=[sx, sz],
        options=dq.Options(progress_meter=False),
    )

    exp_x = result.expects[0, :].real
    exp_z = result.expects[1, :].real

    return tsave, exp_x, exp_z


# ---------------------------------------------------------------------------
# Exponential fitting
# ---------------------------------------------------------------------------


def robust_exp_fit(t, y):
    """Fit y(t) = A * exp(-t/tau) + C using robust least squares.

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Measured values.

    Returns
    -------
    dict
        "tau": fitted lifetime,
        "popt": (A, tau, C) parameter array,
        "y_fit": fitted curve values.
    """
    # Ref: Exponential decay fitting for lifetime extraction. Challenge notebook, Cell 39
    t_np = np.asarray(t)
    y_np = np.asarray(y)

    A0 = float(y_np.max() - y_np.min())
    C0 = float(y_np.min())
    tau0 = float(t_np.max() - t_np.min())
    p0 = [A0, tau0, C0]

    res = least_squares(
        lambda p, t, y: p[0] * np.exp(-t / p[1]) + p[2] - y,
        p0,
        args=(t_np, y_np),
        bounds=([0, 1e-12, -np.inf], [np.inf, np.inf, np.inf]),
        loss="soft_l1",
        f_scale=0.1,
    )

    A, tau, C = res.x

    # Fit quality: true mean squared residual from the residual vector
    # (res.cost is 0.5*sum(rho(r²)) with soft_l1 loss, not sum(r²))
    msr = float(np.mean(res.fun**2))
    fit_ok = (A > 0) and (msr < 0.1) and (tau > 1e-10)

    if not fit_ok:
        warnings.warn(
            f"robust_exp_fit: poor fit quality (A={A:.3g}, tau={tau:.3g}, "
            f"MSR={msr:.3g}). Returning NaN lifetime.",
            stacklevel=2,
        )
        tau = float("nan")

    y_fit = A * np.exp(-t_np / max(tau if not np.isnan(tau) else 1.0, 1e-12)) + C

    return {"tau": tau, "popt": res.x, "y_fit": y_fit, "msr": msr}


# ---------------------------------------------------------------------------
# T_X and T_Z measurement
# ---------------------------------------------------------------------------


def measure_Tz(
    g2_re,
    g2_im,
    eps_d_re,
    eps_d_im,
    tfinal: float = 200.0,
    params: CatQubitParams = DEFAULT_PARAMS,
):
    """Measure bit-flip lifetime T_Z by fitting ⟨Z_L⟩ decay from |+z⟩.

    Parameters
    ----------
    g2_re, g2_im, eps_d_re, eps_d_im : float
        Control knobs.
    tfinal : float
        Simulation duration [us]. Should be ~ T_Z.
    params : CatQubitParams
        Fixed hardware parameters.

    Returns
    -------
    float
        Fitted T_Z [us].
    """
    tsave, _, exp_z = simulate_lifetime(
        g2_re, g2_im, eps_d_re, eps_d_im, "+z", tfinal, params=params
    )
    fit = robust_exp_fit(tsave, exp_z)
    return fit["tau"]


def measure_Tx(
    g2_re,
    g2_im,
    eps_d_re,
    eps_d_im,
    tfinal: float = 1.0,
    params: CatQubitParams = DEFAULT_PARAMS,
):
    """Measure phase-flip lifetime T_X by fitting ⟨X_L⟩ decay from |+x⟩.

    Parameters
    ----------
    g2_re, g2_im, eps_d_re, eps_d_im : float
        Control knobs.
    tfinal : float
        Simulation duration [us]. Should be ~ T_X.
    params : CatQubitParams
        Fixed hardware parameters.

    Returns
    -------
    float
        Fitted T_X [us].
    """
    tsave, exp_x, _ = simulate_lifetime(
        g2_re, g2_im, eps_d_re, eps_d_im, "+x", tfinal, params=params
    )
    fit = robust_exp_fit(tsave, exp_x)
    return fit["tau"]


def measure_lifetimes(
    g2_re,
    g2_im,
    eps_d_re,
    eps_d_im,
    tfinal_z: float = 200.0,
    tfinal_x: float = 1.0,
    params: CatQubitParams = DEFAULT_PARAMS,
):
    """Measure both T_Z and T_X lifetimes.

    Returns
    -------
    dict
        "Tz": bit-flip lifetime, "Tx": phase-flip lifetime, "bias": Tz/Tx.
    """
    Tz = measure_Tz(g2_re, g2_im, eps_d_re, eps_d_im, tfinal_z, params)
    Tx = measure_Tx(g2_re, g2_im, eps_d_re, eps_d_im, tfinal_x, params)
    if np.isnan(Tz):
        warnings.warn(
            "measure_lifetimes: T_Z fit returned NaN (poor decay signal)", stacklevel=2
        )
    if np.isnan(Tx):
        warnings.warn(
            "measure_lifetimes: T_X fit returned NaN (poor decay signal)", stacklevel=2
        )
    if np.isnan(Tz) or np.isnan(Tx):
        return {"Tz": Tz, "Tx": Tx, "bias": float("nan")}
    Tx_safe = max(Tx, 1e-6)
    return {"Tz": Tz, "Tx": Tx, "bias": Tz / Tx_safe}
