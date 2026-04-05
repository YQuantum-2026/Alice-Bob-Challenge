from __future__ import annotations

import warnings
from pathlib import Path

import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from src.cat_qubit import (
    CatQubitParams,
    build_jump_ops,
    build_operators,
)
from src.moon_cat import build_moon_hamiltonian
from src.visualization._wigner import _simulate_for_wigner

# ---------------------------------------------------------------------------
# Moon cat Wigner comparison (static figure)
# ---------------------------------------------------------------------------


def plot_moon_cat_wigner(
    standard_params: np.ndarray | list[float],
    moon_params: np.ndarray | list[float],
    cat_params: CatQubitParams,
    save_path: str | None = None,
    tfinal: float = 10.0,
    npixels: int = 100,
    xmax: float = 5.0,
) -> None:
    """Side-by-side Wigner function comparison for standard vs moon cat.

    Simulates both parameter sets from vacuum via mesolve, partial-traces
    over the buffer mode, and plots the final-time Wigner function of the
    storage mode for each.

    Parameters
    ----------
    standard_params : array-like of 4 floats
        [g2_re, g2_im, eps_d_re, eps_d_im] for the standard cat.
    moon_params : array-like of 5 floats
        [g2_re, g2_im, eps_d_re, eps_d_im, lam] for the moon cat.
    cat_params : CatQubitParams
        Hilbert space dimensions and hardware parameters.
    save_path : str or None
        If set, save figure to this path.
    tfinal : float
        Simulation time [us] to approximate steady state.
    npixels : int
        Resolution of the Wigner function grid.
    xmax : float
        Phase-space extent: plot from -xmax to +xmax in both quadratures.

    Reference
    ---------
    Rousseau et al. (2025), arXiv:2502.07892 — moon cat squeezing extension.
    """

    std_p = np.asarray(standard_params, dtype=float)
    moon_p = np.asarray(moon_params, dtype=float)

    # --- Simulate standard cat ---
    _, states_std = _simulate_for_wigner(
        g2_re=std_p[0],
        g2_im=std_p[1],
        eps_d_re=std_p[2],
        eps_d_im=std_p[3],
        params=cat_params,
        tfinal=tfinal,
        nframes=2,
    )
    rho_std = states_std[-1]  # final-time storage density matrix

    # --- Simulate moon cat ---
    a, b = build_operators(cat_params)
    H_moon = build_moon_hamiltonian(
        a,
        b,
        g2_re=moon_p[0],
        g2_im=moon_p[1],
        eps_d_re=moon_p[2],
        eps_d_im=moon_p[3],
        lam=moon_p[4],
    )
    jump_ops = build_jump_ops(a, b, cat_params)
    psi0 = dq.tensor(dq.fock(cat_params.na, 0), dq.fock(cat_params.nb, 0))
    tsave_moon = jnp.linspace(0, tfinal, 2)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*SparseDIAQArray.*converted to a DenseQArray.*"
        )
        result_moon = dq.mesolve(
            H_moon,
            jump_ops,
            psi0,
            tsave_moon,
            options=dq.Options(progress_meter=False, save_states=True),
        )
    rho_moon = dq.ptrace(result_moon.states, 0)[-1]

    # --- Plot side by side ---
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    dq.plot.wigner(rho_std, ax=ax_left)
    ax_left.set_title("Standard Cat")
    ax_left.set_xlabel(r"$\mathrm{Re}(\alpha)$")
    ax_left.set_ylabel(r"$\mathrm{Im}(\alpha)$")

    lam_val = float(moon_p[4])
    dq.plot.wigner(rho_moon, ax=ax_right)
    ax_right.set_title(rf"Moon Cat ($\lambda$={lam_val:.2f})")
    ax_right.set_xlabel(r"$\mathrm{Re}(\alpha)$")
    ax_right.set_ylabel(r"$\mathrm{Im}(\alpha)$")

    fig.suptitle("Wigner Function: Standard vs Moon Cat", fontsize=13)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[viz] {save_path}")
    plt.close(fig)
