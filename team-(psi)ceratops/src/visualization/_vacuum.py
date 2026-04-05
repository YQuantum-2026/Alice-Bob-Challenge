from __future__ import annotations

import warnings
from pathlib import Path

import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from src.cat_qubit import (
    CatQubitParams,
    build_hamiltonian,
    build_jump_ops,
    build_operators,
)
from src.visualization._wigner import _simulate_for_wigner

# ---------------------------------------------------------------------------
# Vacuum-based (alpha-free) Wigner animation
# ---------------------------------------------------------------------------


def animate_vacuum_wigner(
    g2_re: float,
    g2_im: float,
    eps_d_re: float,
    eps_d_im: float,
    params: CatQubitParams,
    tfinal: float = 20.0,
    nframes: int = 200,
    save_path: str = "figures/vacuum_wigner.mp4",
    fps: int = 10,
) -> str:
    """Wigner animation showing cat state formation from vacuum.

    Starts from vacuum (alpha-free) and shows the storage-mode Wigner
    function as the cat state naturally forms under two-photon dissipation.
    This is the physically correct evolution — no heuristic alpha needed.

    Ref: Réglade et al. "Quantum control of a cat-qubit with bit-flip
      times exceeding ten seconds." Nature 629, 778-783 (2024).

    Parameters
    ----------
    g2_re, g2_im, eps_d_re, eps_d_im : float
        Control parameters.
    params : CatQubitParams
        System parameters.
    tfinal : float
        Simulation duration [μs]. Should be >5× κ₂⁻¹ to see full settling.
    nframes : int
        Number of animation frames.
    save_path : str
        Output MP4 file path.
    fps : int
        Frames per second.

    Returns
    -------
    str
        Path to saved animation.
    """
    # Reuse existing _simulate_for_wigner — it already starts from vacuum
    tsave, states_a = _simulate_for_wigner(
        g2_re, g2_im, eps_d_re, eps_d_im, params, tfinal, nframes
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def update(frame: int) -> None:
        ax.cla()
        dq.plot.wigner(states_a[frame], ax=ax)
        ax.set_title(f"Cat State from Vacuum | t = {float(tsave[frame]):.2f} μs")

    ani = FuncAnimation(fig, update, frames=len(tsave))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)

    print(f"[viz] Vacuum Wigner animation: {save_path}")
    return save_path


def plot_vacuum_diagnostics(
    g2_re: float,
    g2_im: float,
    eps_d_re: float,
    eps_d_im: float,
    params: CatQubitParams,
    t_settle: float = 15.0,
    t_measure_x: float = 1.0,
    t_measure_z: float = 50.0,
    save_path: str = "figures/vacuum_diagnostics.png",
) -> str:
    """Plot parity, quadrature, and photon number vs time from vacuum.

    Three-panel diagnostic showing the alpha-free measurement protocol:
      1. ⟨P⟩(t) — parity decay (T_X measurement)
      2. |⟨Q_θ⟩|(t) — quadrature decay from |α_est⟩ (T_Z measurement)
      3. ⟨a†a⟩(t) — photon number buildup from vacuum

    Ref: Berdou et al. arXiv:2204.09128; Réglade et al. Nature 629 (2024).

    Parameters
    ----------
    g2_re, g2_im, eps_d_re, eps_d_im : float
        Control parameters.
    params : CatQubitParams
        System parameters.
    t_settle, t_measure_x, t_measure_z : float
        Settling and measurement times [μs].
    save_path : str
        Output image path.

    Returns
    -------
    str
        Path to saved figure.
    """
    a, b = build_operators(params)
    H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
    jump_ops = build_jump_ops(a, b, params)

    parity_op = (1j * jnp.pi * a.dag() @ a).expm()
    n_a_op = a.dag() @ a
    psi_vacuum = dq.tensor(dq.fock(params.na, 0), dq.fock(params.nb, 0))

    # Well-aligned quadrature
    g2 = g2_re + 1j * g2_im
    eps_d = eps_d_re + 1j * eps_d_im
    theta = float(jnp.angle(g2 * eps_d) / 2.0)
    phase = np.exp(-1j * theta)
    Q_theta = a * phase + a.dag() * np.conj(phase)

    # --- Panel 1 + 3: Vacuum → cat, measure parity + photon number ---
    tfinal_x = t_settle + t_measure_x
    tsave_x = jnp.linspace(0, tfinal_x, 200)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*SparseDIAQArray.*converted to a DenseQArray.*"
        )
        res_x = dq.mesolve(
            H,
            jump_ops,
            psi_vacuum,
            tsave_x,
            exp_ops=[parity_op, n_a_op],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )

    parity = np.array(res_x.expects[0, :].real)
    n_photon = np.array(res_x.expects[1, :].real)
    t_x_arr = np.array(tsave_x)

    # Data-driven alpha from settled photon number
    settle_idx = np.searchsorted(t_x_arr, t_settle)
    n_settled = n_photon[settle_idx] if settle_idx < len(n_photon) else n_photon[-1]
    alpha_est = np.sqrt(max(n_settled, 0.01))

    # --- Panel 2: Quadrature decay from |α_est⟩ ---
    alpha_complex = alpha_est * np.exp(1j * theta)
    psi_alpha = dq.tensor(dq.coherent(params.na, alpha_complex), dq.fock(params.nb, 0))
    tsave_z = jnp.linspace(0, t_measure_z, 200)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*SparseDIAQArray.*converted to a DenseQArray.*"
        )
        res_z = dq.mesolve(
            H,
            jump_ops,
            psi_alpha,
            tsave_z,
            exp_ops=[Q_theta],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )

    quad = np.array(res_z.expects[0, :].real)
    t_z_arr = np.array(tsave_z)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Parity decay (T_X)
    ax = axes[0]
    ax.plot(t_x_arr, parity, "b-", linewidth=1.5)
    ax.axvline(t_settle, color="gray", linestyle="--", alpha=0.7, label="t_settle")
    ax.set_xlabel("Time [μs]")
    ax.set_ylabel(r"$\langle P \rangle$")
    ax.set_title(r"Parity Decay ($T_X$ measurement)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Quadrature decay (T_Z)
    ax = axes[1]
    ax.plot(t_z_arr, np.abs(quad), "r-", linewidth=1.5)
    ax.set_xlabel("Time [μs]")
    ax.set_ylabel(r"$|\langle Q_\theta \rangle|$")
    ax.set_title(rf"Quadrature Decay ($T_Z$, $|\alpha_{{est}}|$={alpha_est:.2f})")
    ax.grid(True, alpha=0.3)

    # Panel 3: Photon number buildup
    ax = axes[2]
    ax.plot(t_x_arr, n_photon, "g-", linewidth=1.5)
    ax.axvline(t_settle, color="gray", linestyle="--", alpha=0.7, label="t_settle")
    ax.axhline(
        alpha_est**2,
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=rf"$|\alpha_{{est}}|^2$={alpha_est**2:.2f}",
    )
    ax.set_xlabel("Time [μs]")
    ax.set_ylabel(r"$\langle a^\dagger a \rangle$")
    ax.set_title("Photon Number Buildup")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Vacuum-Based (α-free) Cat Qubit Diagnostics", fontsize=13)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[viz] Vacuum diagnostics: {save_path}")
    print(f"      α_est = {alpha_est:.3f} (data-driven from ⟨a†a⟩ = {n_settled:.3f})")
    return save_path
