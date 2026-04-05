"""Wigner function animations for cat qubit phase-space visualization.

Provides:
  - _simulate_for_wigner(): run mesolve and extract storage-mode density matrices
  - animate_wigner(): single-parameter Wigner function animation
  - animate_alpha_sweep(): multi-panel animation across cat sizes
  - animate_before_after(): side-by-side default vs optimized comparison
  - generate_all_animations(): batch generation from benchmark results
  - plot_moon_cat_wigner(): static side-by-side Wigner comparison (standard vs moon)

Physics:
  The Wigner function W(x, p) is a quasi-probability distribution in phase space
  that fully characterizes the quantum state. For cat states |alpha> + |-alpha>,
  the Wigner function shows two Gaussian blobs (coherent components) with
  interference fringes between them. Larger alpha yields more separated blobs
  and finer fringes.

Reference:
  Berdou et al. "One hundred second bit-flip time in a two-photon
  dissipative oscillator." PRX Quantum 4, 020350 (2023). arXiv:2204.09128
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from tqdm import tqdm

from src.cat_qubit import (
    CatQubitParams,
    build_hamiltonian,
    build_jump_ops,
    build_operators,
    compute_alpha,
)

# ---------------------------------------------------------------------------
# Internal simulation helper
# ---------------------------------------------------------------------------


def _simulate_for_wigner(
    g2_re: float,
    g2_im: float,
    eps_d_re: float,
    eps_d_im: float,
    params: CatQubitParams,
    tfinal: float = 5.0,
    nframes: int = 200,
) -> tuple[jnp.ndarray, Any]:
    """Run Lindblad simulation and extract storage-mode density matrices.

    Builds the full storage-buffer Hamiltonian and jump operators, evolves
    from vacuum via mesolve with save_states=True, then partial-traces out
    the buffer mode to yield storage-only density matrices for Wigner plots.

    Parameters
    ----------
    g2_re, g2_im : float
        Real and imaginary parts of the two-photon coupling g2.
    eps_d_re, eps_d_im : float
        Real and imaginary parts of the buffer drive amplitude eps_d.
    params : CatQubitParams
        Hilbert space dimensions and hardware parameters.
    tfinal : float
        Total simulation time [us].
    nframes : int
        Number of time frames to save.

    Returns
    -------
    tsave : jnp.ndarray
        Time points, shape (nframes,).
    states_a : QArray
        Storage-mode density matrices at each time point, obtained via
        partial trace over the buffer mode.
    """
    a, b = build_operators(params)
    H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
    jump_ops = build_jump_ops(a, b, params)

    # Initial state: vacuum in both modes
    psi0 = dq.tensor(dq.fock(params.na, 0), dq.fock(params.nb, 0))
    tsave = jnp.linspace(0, tfinal, nframes)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*SparseDIAQArray.*converted to a DenseQArray.*"
        )
        result = dq.mesolve(
            H,
            jump_ops,
            psi0,
            tsave,
            options=dq.Options(progress_meter=False, save_states=True),
        )

    # Partial trace over buffer (index 1) to get storage-mode states
    states_a = dq.ptrace(result.states, 0)

    return tsave, states_a


# ---------------------------------------------------------------------------
# Single-parameter Wigner animation
# ---------------------------------------------------------------------------


def animate_wigner(
    g2_re: float,
    g2_im: float,
    eps_d_re: float,
    eps_d_im: float,
    params: CatQubitParams,
    tfinal: float = 5.0,
    nframes: int = 200,
    save_path: str = "figures/wigner.mp4",
    fps: int = 10,
    title: str | None = None,
) -> str:
    """Create a Wigner function animation for a single parameter set.

    Simulates the cat qubit evolution from vacuum and renders the Wigner
    function of the storage mode at each time step as an MP4 animation.

    Parameters
    ----------
    g2_re, g2_im : float
        Real and imaginary parts of g2.
    eps_d_re, eps_d_im : float
        Real and imaginary parts of eps_d.
    params : CatQubitParams
        System parameters.
    tfinal : float
        Simulation duration [us].
    nframes : int
        Number of animation frames.
    save_path : str
        Output MP4 file path.
    fps : int
        Frames per second in output video.
    title : str or None
        Custom title prefix. If None, auto-generates from alpha.

    Returns
    -------
    str
        The save_path where the animation was written.
    """
    tsave, states_a = _simulate_for_wigner(
        g2_re, g2_im, eps_d_re, eps_d_im, params, tfinal, nframes
    )

    alpha = compute_alpha(g2_re, g2_im, eps_d_re, eps_d_im, params)
    alpha_val = float(alpha)

    if title is None:
        title = f"Cat Qubit Wigner (alpha={alpha_val:.2f})"

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def update(frame: int) -> None:
        ax.cla()
        dq.plot.wigner(states_a[frame], ax=ax)
        ax.set_title(f"{title} | t = {float(tsave[frame]):.2f} us")

    ani = FuncAnimation(fig, update, frames=len(tsave))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)

    return save_path


# ---------------------------------------------------------------------------
# Alpha sweep animation (multi-panel)
# ---------------------------------------------------------------------------


def animate_alpha_sweep(
    params: CatQubitParams,
    alphas: list[float] | None = None,
    tfinal: float = 5.0,
    nframes: int = 200,
    save_path: str = "figures/wigner_alpha_sweep.mp4",
    fps: int = 10,
) -> str:
    """Create a multi-panel Wigner animation sweeping across cat sizes.

    For each target alpha, computes the required eps_d (with g2=1+0j) and
    simulates the evolution. All panels animate simultaneously in a 2x3 grid.

    The drive amplitude is derived from the adiabatic elimination relation:
      eps_d = alpha^2 + kappa_b * kappa_a / 8

    Parameters
    ----------
    params : CatQubitParams
        System parameters.
    alphas : list[float] or None
        Target cat sizes. Defaults to [0.5, 1.0, 1.5, 2.0, 2.5].
    tfinal : float
        Simulation duration [us].
    nframes : int
        Number of animation frames.
    save_path : str
        Output MP4 file path.
    fps : int
        Frames per second.

    Returns
    -------
    str
        The save_path where the animation was written.
    """
    if alphas is None:
        alphas = [0.5, 1.0, 1.5, 2.0, 2.5]

    # For each alpha, compute eps_d needed with g2 = 1 + 0j
    # eps_d = alpha^2 + kappa_b * kappa_a / 8
    g2_re, g2_im = 1.0, 0.0
    simulations: list[tuple[jnp.ndarray, Any, float]] = []

    for alpha_target in tqdm(alphas, desc="Simulating alpha sweep"):
        eps_d_re = float(alpha_target**2 + params.kappa_b * params.kappa_a / 8.0)
        eps_d_im = 0.0

        tsave, states_a = _simulate_for_wigner(
            g2_re, g2_im, eps_d_re, eps_d_im, params, tfinal, nframes
        )
        actual_alpha = float(compute_alpha(g2_re, g2_im, eps_d_re, eps_d_im, params))
        simulations.append((tsave, states_a, actual_alpha))

    # Create 2x3 grid
    n_panels = len(alphas)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = axes.flatten()

    # Hide unused panels
    for i in range(n_panels, nrows * ncols):
        axes_flat[i].set_visible(False)

    def update(frame: int) -> None:
        for i, (tsave_i, states_a_i, alpha_i) in enumerate(simulations):
            if i >= len(axes_flat):
                break
            ax = axes_flat[i]
            ax.cla()
            dq.plot.wigner(states_a_i[frame], ax=ax)
            ax.set_title(
                f"alpha={alpha_i:.2f} | t={float(tsave_i[frame]):.2f} us",
                fontsize=9,
            )

    ani = FuncAnimation(fig, update, frames=nframes)
    fig.suptitle("Cat Qubit Alpha Sweep", fontsize=14, y=1.02)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)

    return save_path


# ---------------------------------------------------------------------------
# Before/after comparison animation
# ---------------------------------------------------------------------------


def animate_before_after(
    best_params: dict[str, Any],
    params: CatQubitParams,
    tfinal: float = 5.0,
    nframes: int = 200,
    save_path: str = "figures/wigner_before_after.mp4",
    fps: int = 10,
) -> str:
    """Create a side-by-side animation comparing default vs optimized parameters.

    Left panel shows the default cat qubit (g2=1+0j, eps_d=4+0j), right panel
    shows the optimized parameters from the benchmark.

    Parameters
    ----------
    best_params : dict
        Must contain keys: "g2_re", "g2_im", "eps_d_re", "eps_d_im".
        Optionally "label" for the title.
    params : CatQubitParams
        System parameters.
    tfinal : float
        Simulation duration [us].
    nframes : int
        Number of animation frames.
    save_path : str
        Output MP4 file path.
    fps : int
        Frames per second.

    Returns
    -------
    str
        The save_path where the animation was written.
    """
    # Default parameters
    default_g2_re, default_g2_im = 1.0, 0.0
    default_eps_d_re, default_eps_d_im = 4.0, 0.0

    # Simulate default
    tsave_def, states_def = _simulate_for_wigner(
        default_g2_re,
        default_g2_im,
        default_eps_d_re,
        default_eps_d_im,
        params,
        tfinal,
        nframes,
    )
    alpha_def = float(
        compute_alpha(
            default_g2_re, default_g2_im, default_eps_d_re, default_eps_d_im, params
        )
    )

    # Simulate optimized
    opt_g2_re = float(best_params["g2_re"])
    opt_g2_im = float(best_params["g2_im"])
    opt_eps_d_re = float(best_params["eps_d_re"])
    opt_eps_d_im = float(best_params["eps_d_im"])

    tsave_opt, states_opt = _simulate_for_wigner(
        opt_g2_re,
        opt_g2_im,
        opt_eps_d_re,
        opt_eps_d_im,
        params,
        tfinal,
        nframes,
    )
    alpha_opt = float(
        compute_alpha(opt_g2_re, opt_g2_im, opt_eps_d_re, opt_eps_d_im, params)
    )

    opt_label = best_params.get("label", "Optimized")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

    def update(frame: int) -> None:
        ax_left.cla()
        dq.plot.wigner(states_def[frame], ax=ax_left)
        ax_left.set_title(
            f"Default (alpha={alpha_def:.2f}) | t={float(tsave_def[frame]):.2f} us"
        )

        ax_right.cla()
        dq.plot.wigner(states_opt[frame], ax=ax_right)
        ax_right.set_title(
            f"{opt_label} (alpha={alpha_opt:.2f}) | t={float(tsave_opt[frame]):.2f} us"
        )

    ani = FuncAnimation(fig, update, frames=nframes)
    fig.suptitle("Default vs Optimized Cat Qubit", fontsize=14)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)

    return save_path


# ---------------------------------------------------------------------------
# Batch generation from benchmark results
# ---------------------------------------------------------------------------


def generate_all_animations(
    all_results: dict[str, Any],
    cfg: Any,
    tfinal: float = 5.0,
    nframes: int = 200,
    fps: int = 10,
) -> list[str]:
    """Generate all Wigner function animations from benchmark results.

    Called from run.py after the benchmark completes. Produces three types
    of animation:
      a) Per-optimizer: Wigner of best params for each optimizer with proxy/no-drift
      b) Alpha sweep: multi-panel animation across cat sizes
      c) Before/after: best overall result vs default parameters

    Parameters
    ----------
    all_results : dict
        Must contain key "benchmark" mapping to a list of RunResult objects.
        Each RunResult has: .optimizer_type, .reward_type, .drift_type,
        .param_history (list of np.ndarray), .reward_history (list of float).
    cfg : RunConfig
        Configuration object with .cat_params.
    tfinal : float
        Simulation duration [us] for each animation.
    nframes : int
        Number of animation frames.
    fps : int
        Frames per second in output videos.

    Returns
    -------
    list[str]
        Paths to all generated animation files.
    """
    if "benchmark" not in all_results:
        print("[viz] No benchmark results found, skipping animations.")
        return []

    results = all_results["benchmark"]
    params = cfg.cat_params
    generated: list[str] = []

    # (a) Per-optimizer: for each unique optimizer with proxy reward / no drift
    proxy_no_drift = [
        r for r in results if r.reward_type == "proxy" and r.drift_type == "none"
    ]

    if proxy_no_drift:
        seen_optimizers: set[str] = set()
        for r in tqdm(proxy_no_drift, desc="Per-optimizer Wigner animations"):
            if r.optimizer_type in seen_optimizers:
                continue
            seen_optimizers.add(r.optimizer_type)

            # Get best parameters (last entry in param_history)
            if not r.param_history:
                continue
            best = r.param_history[-1]

            save_path = f"figures/wigner_{r.optimizer_type}.mp4"
            try:
                animate_wigner(
                    g2_re=float(best[0]),
                    g2_im=float(best[1]),
                    eps_d_re=float(best[2]),
                    eps_d_im=float(best[3]),
                    params=params,
                    tfinal=tfinal,
                    nframes=nframes,
                    save_path=save_path,
                    fps=fps,
                    title=f"{r.optimizer_type} best",
                )
                generated.append(save_path)
                print(f"[viz] {save_path}")
            except Exception as e:
                print(f"[viz] Failed {save_path}: {e}")

    # (b) Alpha sweep
    try:
        sweep_path = animate_alpha_sweep(
            params=params,
            tfinal=tfinal,
            nframes=nframes,
            save_path="figures/wigner_alpha_sweep.mp4",
            fps=fps,
        )
        generated.append(sweep_path)
        print(f"[viz] {sweep_path}")
    except Exception as e:
        print(f"[viz] Failed alpha sweep: {e}")

    # (c) Before/after: best overall result vs default
    if results:
        # Find the result with highest final reward
        best_result = max(
            results,
            key=lambda r: r.reward_history[-1] if r.reward_history else float("-inf"),
        )
        if best_result.param_history:
            best = best_result.param_history[-1]
            best_params = {
                "g2_re": float(best[0]),
                "g2_im": float(best[1]),
                "eps_d_re": float(best[2]),
                "eps_d_im": float(best[3]),
                "label": best_result.label,
            }
            try:
                ba_path = animate_before_after(
                    best_params=best_params,
                    params=params,
                    tfinal=tfinal,
                    nframes=nframes,
                    save_path="figures/wigner_before_after.mp4",
                    fps=fps,
                )
                generated.append(ba_path)
                print(f"[viz] {ba_path}")
            except Exception as e:
                print(f"[viz] Failed before/after: {e}")

    print(f"[viz] Generated {len(generated)} animation(s)")
    return generated
