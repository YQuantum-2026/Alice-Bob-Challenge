from __future__ import annotations

from pathlib import Path
from typing import Any

import dynamiqs as dq
import matplotlib.pyplot as plt
import numpy as np

from src.cat_qubit import (
    CatQubitParams,
    robust_exp_fit,
    simulate_lifetime,
)
from src.visualization._wigner import _simulate_for_wigner

# ---------------------------------------------------------------------------
# Lifetime decay plots (replicating challenge cells 40-41)
# ---------------------------------------------------------------------------


def plot_lifetime_decay(
    g2_re: float = 1.0,
    g2_im: float = 0.0,
    eps_d_re: float = 4.0,
    eps_d_im: float = 0.0,
    initial_state: str = "+z",
    tfinal: float = 200.0,
    npoints: int = 200,
    params: CatQubitParams | None = None,
    save_path: str | None = None,
    ax: Any = None,
) -> dict:
    """Plot exponential decay of a logical expectation value with fit.

    Replicates challenge notebook cells 40 (T_Z) and 41 (T_X). Runs the
    full Lindblad simulation, measures the relevant logical operator, and
    overlays an exponential fit to extract the lifetime.

    Parameters
    ----------
    g2_re, g2_im, eps_d_re, eps_d_im : float
        Control parameters.
    initial_state : str
        One of "+z" (for T_Z measurement) or "+x" (for T_X measurement).
    tfinal : float
        Simulation duration [us].
    npoints : int
        Number of time points.
    params : CatQubitParams or None
        System parameters. Default: CatQubitParams(na=15, nb=5).
    save_path : str or None
        If provided, saves the figure. Otherwise shows inline.
    ax : matplotlib Axes or None
        If provided, plots on this axes instead of creating a new figure.

    Returns
    -------
    dict
        "tau": fitted lifetime, "popt": fit parameters, "y_fit": fit curve.
    """

    if params is None:
        params = CatQubitParams(na=15, nb=5, kappa_b=10.0, kappa_a=1.0)

    tsave, exp_x, exp_z = simulate_lifetime(
        g2_re,
        g2_im,
        eps_d_re,
        eps_d_im,
        initial_state,
        tfinal,
        npoints,
        params,
    )

    # Select the relevant observable
    if initial_state in ("+z", "-z"):
        data = exp_z
        label_obs = r"$\langle Z_L \rangle$"
        label_T = "$T_Z$"
    else:
        data = exp_x
        label_obs = r"$\langle X_L \rangle$"
        label_T = "$T_X$"

    t_np = np.asarray(tsave)
    y_np = np.asarray(data)
    fit = robust_exp_fit(t_np, y_np)
    tau = fit["tau"]
    y_fit = fit["y_fit"]

    # Plot
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    ax.plot(t_np, y_np, linewidth=1.5, label=label_obs)
    ax.plot(
        t_np,
        y_fit,
        "--",
        linewidth=1.5,
        label=f"Exponential Fit, {label_T} = {tau:.2f} μs",
    )
    ax.set_xlabel("Time [μs]")
    ax.set_ylabel("Expectation Value")
    ax.set_title(f"Lifetime Measurement from |{initial_state}⟩")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if own_fig:
        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[viz] Lifetime decay plot: {save_path}")
        plt.close(fig)

    return fit


def plot_lifetimes_pair(
    g2_re: float = 1.0,
    g2_im: float = 0.0,
    eps_d_re: float = 4.0,
    eps_d_im: float = 0.0,
    params: CatQubitParams | None = None,
    tfinal_z: float = 200.0,
    tfinal_x: float = 1.0,
    npoints: int = 200,
    save_path: str = "figures/challenge_lifetime_decay.png",
) -> dict:
    """Plot T_Z and T_X lifetime decays side-by-side.

    Replicates challenge notebook cells 40 + 41 in a single 1x2 figure.

    Parameters
    ----------
    g2_re, g2_im, eps_d_re, eps_d_im : float
        Control parameters.
    params : CatQubitParams or None
        System parameters.
    tfinal_z : float
        T_Z simulation duration [us].
    tfinal_x : float
        T_X simulation duration [us].
    npoints : int
        Time points per measurement.
    save_path : str
        Output PNG path.

    Returns
    -------
    dict
        "Tz": fitted T_Z, "Tx": fitted T_X, "bias": T_Z/T_X.
    """
    if params is None:
        params = CatQubitParams(na=15, nb=5, kappa_b=10.0, kappa_a=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fit_z = plot_lifetime_decay(
        g2_re,
        g2_im,
        eps_d_re,
        eps_d_im,
        initial_state="+z",
        tfinal=tfinal_z,
        npoints=npoints,
        params=params,
        ax=axes[0],
    )

    fit_x = plot_lifetime_decay(
        g2_re,
        g2_im,
        eps_d_re,
        eps_d_im,
        initial_state="+x",
        tfinal=tfinal_x,
        npoints=npoints,
        params=params,
        ax=axes[1],
    )

    Tz = fit_z["tau"]
    Tx = fit_x["tau"]
    bias = Tz / max(Tx, 1e-6)

    fig.suptitle(
        rf"Cat Qubit Lifetimes | $T_Z$ = {Tz:.2f} μs, $T_X$ = {Tx:.4f} μs, "
        rf"$\eta$ = $T_Z/T_X$ = {bias:.1f}",
        fontsize=12,
    )
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[viz] Lifetime pair plot: {save_path}")
    print(f"      T_Z = {Tz:.2f} μs, T_X = {Tx:.4f} μs, η = {bias:.1f}")
    return {"Tz": Tz, "Tx": Tx, "bias": bias}


# ---------------------------------------------------------------------------
# Static Wigner snapshots
# ---------------------------------------------------------------------------


def plot_wigner_snapshots(
    g2_re: float = 1.0,
    g2_im: float = 0.0,
    eps_d_re: float = 4.0,
    eps_d_im: float = 0.0,
    params: CatQubitParams | None = None,
    times: list[float] | None = None,
    tfinal: float = 5.0,
    save_path: str = "figures/challenge_wigner_snapshots.png",
) -> str:
    """Plot Wigner function at multiple time points.

    Similar to dq.plot.wigner_mosaic from the dynamiqs tutorials.

    Parameters
    ----------
    g2_re, g2_im, eps_d_re, eps_d_im : float
        Control parameters.
    params : CatQubitParams or None
        System parameters.
    times : list[float] or None
        Specific time points to snapshot. If None, uses 6 evenly spaced.
    tfinal : float
        Total simulation time [us].
    save_path : str
        Output PNG path.

    Returns
    -------
    str
        Path to saved figure.
    """
    if params is None:
        params = CatQubitParams(na=15, nb=5, kappa_b=10.0, kappa_a=1.0)

    if times is None:
        times = [0.0, tfinal * 0.2, tfinal * 0.4, tfinal * 0.6, tfinal * 0.8, tfinal]

    n_panels = len(times)
    ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols

    # Run simulation with enough frames
    nframes = max(200, n_panels * 10)
    tsave_all, states_a = _simulate_for_wigner(
        g2_re,
        g2_im,
        eps_d_re,
        eps_d_im,
        params,
        tfinal,
        nframes,
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes_flat = [axes] if n_panels == 1 else axes.flatten()

    tsave_np = np.asarray(tsave_all)

    for i, t_target in enumerate(times):
        if i >= len(axes_flat):
            break
        idx = int(np.argmin(np.abs(tsave_np - t_target)))
        ax = axes_flat[i]
        dq.plot.wigner(states_a[idx], ax=ax)
        ax.set_title(f"t = {float(tsave_all[idx]):.2f} μs")

    # Hide unused panels
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Wigner Function Snapshots (from vacuum)", fontsize=13)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[viz] Wigner snapshots: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Even/odd cat state visualization (from tutorial)
# ---------------------------------------------------------------------------


def plot_cat_states(
    alpha: float = 2.0,
    na: int = 15,
    save_path: str = "figures/challenge_cat_states.png",
) -> str:
    """Plot even and odd cat states: Wigner + Fock distributions.

    2x2 grid: (even Wigner, even Fock, odd Wigner, odd Fock).
    Replicates the Introduction to Cats tutorial visualization.

    Parameters
    ----------
    alpha : float
        Cat size.
    na : int
        Hilbert space truncation.
    save_path : str
        Output PNG path.

    Returns
    -------
    str
        Path to saved figure.
    """
    cat_even = dq.unit(dq.coherent(na, alpha) + dq.coherent(na, -alpha))
    cat_odd = dq.unit(dq.coherent(na, alpha) - dq.coherent(na, -alpha))

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    dq.plot.wigner(cat_even, ax=axes[0, 0])
    axes[0, 0].set_title(rf"Even Cat $|C_+\rangle$ — Wigner ($\alpha$={alpha})")

    dq.plot.fock(cat_even, ax=axes[0, 1])
    axes[0, 1].set_title(r"Even Cat $|C_+\rangle$ — Fock")

    dq.plot.wigner(cat_odd, ax=axes[1, 0])
    axes[1, 0].set_title(rf"Odd Cat $|C_-\rangle$ — Wigner ($\alpha$={alpha})")

    dq.plot.fock(cat_odd, ax=axes[1, 1])
    axes[1, 1].set_title(r"Odd Cat $|C_-\rangle$ — Fock")

    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[viz] Cat states plot: {save_path}")
    return save_path
