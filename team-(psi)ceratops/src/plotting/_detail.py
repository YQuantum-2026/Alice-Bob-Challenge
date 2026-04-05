from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Logical X and Z decay plots
# ---------------------------------------------------------------------------


def plot_logical_decay(
    params_list: list[dict],
    cat_params=None,
    save_path: str | None = None,
):
    """Plot logical X and Z expectation value decay over time.

    Shows how ⟨X_L⟩ and ⟨Z_L⟩ evolve from their initial values,
    visualizing the bit-flip (Z) and phase-flip (X) decoherence channels.

    Parameters
    ----------
    params_list : list[dict]
        Each dict has keys: g2_re, g2_im, eps_d_re, eps_d_im, label.
        Multiple parameter sets are plotted on the same axes for comparison.
    cat_params : CatQubitParams, optional
        Hilbert space and hardware config. Uses default if None.
    save_path : str, optional
        Path to save the figure.
    """
    import warnings

    from src.cat_qubit import (
        DEFAULT_PARAMS,
        simulate_lifetime,
    )

    if cat_params is None:
        cat_params = DEFAULT_PARAMS

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(params_list), 1)))

    for idx, p in enumerate(params_list):
        label = p.get("label", f"Params {idx}")
        color = colors[idx]

        # --- Z decay (bit-flip): from |+z⟩, measure ⟨Z_L⟩ ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsave_z, exp_x_z, exp_z_z = simulate_lifetime(
                p["g2_re"],
                p["g2_im"],
                p["eps_d_re"],
                p["eps_d_im"],
                initial_state_label="+z",
                tfinal=200.0,
                npoints=150,
                params=cat_params,
            )
        axes[0].plot(
            np.asarray(tsave_z),
            np.asarray(exp_z_z),
            label=label,
            color=color,
            linewidth=1.5,
        )

        # --- X decay (phase-flip): from |+x⟩, measure ⟨X_L⟩ ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsave_x, exp_x_x, exp_z_x = simulate_lifetime(
                p["g2_re"],
                p["g2_im"],
                p["eps_d_re"],
                p["eps_d_im"],
                initial_state_label="+x",
                tfinal=1.0,
                npoints=150,
                params=cat_params,
            )
        axes[1].plot(
            np.asarray(tsave_x),
            np.asarray(exp_x_x),
            label=label,
            color=color,
            linewidth=1.5,
        )

    # --- Format Z decay panel ---
    axes[0].set_xlabel(r"Time [$\mu$s]")
    axes[0].set_ylabel(r"$\langle Z_L \rangle$")
    axes[0].set_title(
        r"Bit-Flip Channel: $\langle Z_L \rangle$ Decay from $|{+z}\rangle$"
    )
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    # --- Format X decay panel ---
    axes[1].set_xlabel(r"Time [$\mu$s]")
    axes[1].set_ylabel(r"$\langle X_L \rangle$")
    axes[1].set_title(
        r"Phase-Flip Channel: $\langle X_L \rangle$ Decay from $|{+x}\rangle$"
    )
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[plots] {save_path}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Challenge-style detail plots
# ---------------------------------------------------------------------------


def plot_run_detail(result, save_path=None):
    """Challenge-style 2-panel figure: reward convergence + parameter tracking.

    Top: Reward vs epoch with std fill_between bands.
    Bottom: All 4 control params vs epoch as colored lines.

    Matches the plotting style from Alice-Bob challenge notebook cell 18.
    """
    rh = np.array(result.reward_history)
    rsh = np.array(result.reward_std_history)
    ph = np.array(result.param_history)  # shape (n_epochs, 4)
    epochs = np.arange(len(rh))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top: Reward vs epoch
    ax = axes[0]
    ax.plot(epochs, rh, label="Mean reward", color="#1f77b4")
    ax.fill_between(epochs, rh - rsh, rh + rsh, alpha=0.2, color="#1f77b4")
    ax.set_ylabel("Reward")
    ax.set_title(
        f"{result.optimizer_type}/{result.reward_type}/{result.drift_type}: Reward vs Epoch"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: Parameters vs epoch
    ax = axes[1]
    param_labels = [
        r"Re($g_2$)",
        r"Im($g_2$)",
        r"Re($\varepsilon_d$)",
        r"Im($\varepsilon_d$)",
    ]
    param_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (label, color) in enumerate(zip(param_labels, param_colors)):
        ax.plot(epochs, ph[:, i], label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter value")
    ax.set_title("Parameter Convergence")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[plots] {save_path}")
    plt.show()
    plt.close(fig)


def plot_drift_detail(result, save_path=None):
    """Challenge-style 3-panel figure: reward + parameter tracking + drift trajectory.

    Top: Reward vs epoch with std bands.
    Middle: Control params vs epoch with dashed drift trajectory overlay.
    Bottom: Raw drift offsets over time.

    Matches the plotting style from Alice-Bob challenge notebook cell 24.
    """
    rh = np.array(result.reward_history)
    rsh = np.array(result.reward_std_history)
    ph = np.array(result.param_history)
    epochs = np.arange(len(rh))

    # Extract drift trajectories
    drift_g2_re = np.array([d.get("g2_re", 0) for d in result.drift_offset_history])
    drift_g2_im = np.array([d.get("g2_im", 0) for d in result.drift_offset_history])
    drift_epsd_re = np.array(
        [d.get("eps_d_re", 0) for d in result.drift_offset_history]
    )
    np.array([d.get("eps_d_im", 0) for d in result.drift_offset_history])
    drift_detuning = np.array(
        [d.get("detuning", 0) for d in result.drift_offset_history]
    )
    drift_kerr = np.array([d.get("kerr", 0) for d in result.drift_offset_history])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # Top: Reward vs epoch
    ax = axes[0]
    ax.plot(epochs, rh, label="Mean reward", color="#1f77b4")
    ax.fill_between(epochs, rh - rsh, rh + rsh, alpha=0.2, color="#1f77b4")
    ax.set_ylabel("Reward")
    ax.set_title(
        f"{result.optimizer_type}/{result.reward_type}/{result.drift_type}: Reward Under Drift"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: Parameters + drift trajectories
    ax = axes[1]
    param_labels = [
        r"Re($g_2$)",
        r"Im($g_2$)",
        r"Re($\varepsilon_d$)",
        r"Im($\varepsilon_d$)",
    ]
    param_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (label, color) in enumerate(zip(param_labels, param_colors)):
        ax.plot(epochs, ph[:, i], label=label, color=color, linewidth=1.5)

    # Dashed drift trajectories (true drifting optimum)
    # The "optimal" params shift BY the drift offset, so optimal = base + offset
    if np.any(drift_g2_re != 0):
        ax.plot(
            epochs,
            1.0 + drift_g2_re,
            "--",
            color="#1f77b4",
            alpha=0.5,
            label=r"Re($g_2$) optimal",
            linewidth=2,
        )
    if np.any(drift_g2_im != 0):
        ax.plot(
            epochs,
            drift_g2_im,
            "--",
            color="#ff7f0e",
            alpha=0.5,
            label=r"Im($g_2$) optimal",
            linewidth=2,
        )

    ax.set_ylabel("Parameter value")
    ax.set_title("Parameter Tracking vs Drift Trajectory")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Bottom: Raw drift offsets
    ax = axes[2]
    if np.any(drift_g2_re != 0):
        ax.plot(epochs, drift_g2_re, label=r"$\delta_{g_2,re}$", linewidth=1.2)
    if np.any(drift_g2_im != 0):
        ax.plot(epochs, drift_g2_im, label=r"$\delta_{g_2,im}$", linewidth=1.2)
    if np.any(drift_epsd_re != 0):
        ax.plot(
            epochs, drift_epsd_re, label=r"$\delta_{\varepsilon_d,re}$", linewidth=1.2
        )
    if np.any(drift_detuning != 0):
        ax.plot(epochs, drift_detuning, label=r"$\Delta$ [MHz]", linewidth=1.2)
    if np.any(drift_kerr != 0):
        ax.plot(epochs, drift_kerr, label=r"$K$ [MHz]", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Drift offset")
    ax.set_title("Drift Offsets Over Time")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[plots] {save_path}")
    plt.show()
    plt.close(fig)
