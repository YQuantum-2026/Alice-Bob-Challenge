from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.plotting._style import (
    DRIFT_STYLES,
    OPTIMIZER_COLORS,
)

# ---------------------------------------------------------------------------
# Pareto frontier and convergence analysis
# ---------------------------------------------------------------------------


def plot_pareto_frontier(results, save_path=None):
    """Pareto frontier: T_Z vs T_X for all validated runs.

    Shows the tradeoff between bit-flip and phase-flip lifetimes.
    Iso-bias lines show constant eta = T_Z / T_X.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    drift_markers = {
        "none": "o",
        "amplitude_slow": "s",
        "amplitude_fast": "^",
        "frequency": "D",
        "kerr": "v",
        "snr": "<",
        "multi": ">",
        "white_noise": "P",
        "step": "X",
        "frequency_step": "p",
        "tls": "*",
    }

    for r in results:
        if not r.validation_history:
            continue
        v = r.validation_history[-1]
        color = OPTIMIZER_COLORS.get(r.optimizer_type, "#333")
        marker = drift_markers.get(r.drift_type, "o")
        ax.scatter(
            v["Tz"],
            v["Tx"],
            c=color,
            marker=marker,
            s=80,
            zorder=5,
            edgecolors="black",
            linewidth=0.5,
        )

    # Iso-bias lines
    tz_range = np.linspace(
        0.1, ax.get_xlim()[1] * 1.1 if ax.get_xlim()[1] > 1 else 100, 100
    )
    for eta in [50, 100, 200]:
        ax.plot(tz_range, tz_range / eta, "--", color="gray", alpha=0.3, linewidth=0.8)
        ax.annotate(
            f"eta={eta}",
            xy=(tz_range[-1], tz_range[-1] / eta),
            fontsize=6,
            color="gray",
            alpha=0.6,
        )

    # Legend for optimizers (colors)
    for opt, color in OPTIMIZER_COLORS.items():
        ax.scatter([], [], c=color, label=opt, s=40, edgecolors="black", linewidth=0.5)
    ax.legend(fontsize=7, title="Optimizer")

    ax.set_xlabel(r"$T_Z$ [$\mu$s] (bit-flip)")
    ax.set_ylabel(r"$T_X$ [$\mu$s] (phase-flip)")
    ax.set_title(r"Pareto Frontier: $T_Z$ vs $T_X$")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[plots] {save_path}")
    plt.show()
    plt.close(fig)


def plot_convergence_speed(results, threshold_pct=0.9, save_path=None):
    """Grouped bar chart: epochs to reach threshold % of best final reward.

    Lower = faster convergence.
    """
    # Find global best final reward
    best_final = max(r.reward_history[-1] for r in results if r.reward_history)
    worst_initial = min(r.reward_history[0] for r in results if r.reward_history)
    threshold = worst_initial + threshold_pct * (best_final - worst_initial)

    # Group by optimizer
    data = {}  # {optimizer: {drift: epochs_to_threshold}}
    for r in results:
        if not r.reward_history:
            continue
        rh = np.array(r.reward_history)
        above = np.where(rh >= threshold)[0]
        epoch_at = int(above[0]) if len(above) > 0 else len(rh)
        data.setdefault(r.optimizer_type, {})[r.drift_type] = epoch_at

    if not data:
        return

    optimizers = sorted(data.keys())
    all_drifts = sorted(set(d for opt_data in data.values() for d in opt_data))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(len(optimizers))
    width = 0.8 / max(len(all_drifts), 1)

    drift_colors = plt.cm.Set2(np.linspace(0, 1, len(all_drifts)))
    for i, drift in enumerate(all_drifts):
        vals = [data.get(opt, {}).get(drift, 0) for opt in optimizers]
        ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            label=drift,
            color=drift_colors[i],
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(optimizers)
    ax.set_xlabel("Optimizer")
    ax.set_ylabel(f"Epochs to {int(threshold_pct * 100)}% of best")
    ax.set_title("Convergence Speed (lower = faster)")
    ax.legend(fontsize=7, title="Drift")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[plots] {save_path}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Alpha evolution and efficiency
# ---------------------------------------------------------------------------


def plot_alpha_evolution(results, save_path=None):
    """Line plot of cat size alpha evolution from validation_history."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    for r in results:
        if not r.validation_history:
            continue
        epochs_v = [v["epoch"] for v in r.validation_history]
        alphas = [v.get("alpha", 0) for v in r.validation_history]
        if not any(a > 0 for a in alphas):
            continue

        color = OPTIMIZER_COLORS.get(r.optimizer_type, "#333")
        style = DRIFT_STYLES.get(r.drift_type, "-")
        label = f"{r.optimizer_type}/{r.drift_type}"
        ax.plot(
            epochs_v,
            alphas,
            color=color,
            linestyle=style if isinstance(style, str) else "-",
            label=label,
            linewidth=1.5,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Cat size $|\alpha|$")
    ax.set_title(r"Cat Size $|\alpha|$ Evolution During Optimization")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[plots] {save_path}")
    plt.show()
    plt.close(fig)


def plot_efficiency_scatter(results, save_path=None):
    """Scatter: wall time vs final reward, colored by optimizer."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for r in results:
        if not r.reward_history:
            continue
        color = OPTIMIZER_COLORS.get(r.optimizer_type, "#333")
        final_r = r.reward_history[-1]
        ax.scatter(
            r.wall_time,
            final_r,
            c=color,
            s=60,
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )

    # Legend
    for opt, color in OPTIMIZER_COLORS.items():
        ax.scatter([], [], c=color, label=opt, s=40, edgecolors="black", linewidth=0.5)
    ax.legend(fontsize=7, title="Optimizer")

    ax.set_xlabel("Wall Time [s]")
    ax.set_ylabel("Final Reward")
    ax.set_title("Efficiency: Wall Time vs Final Reward")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[plots] {save_path}")
    plt.show()
    plt.close(fig)
