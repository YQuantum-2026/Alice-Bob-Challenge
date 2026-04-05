from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.plotting._style import (
    DRIFT_STYLES,
    OPTIMIZER_COLORS,
    REWARD_COLORS,
    set_plot_style,
)

# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_reward_convergence(results, group_by="optimizer", save_path=None):
    """Plot reward vs epoch for multiple runs.

    Parameters
    ----------
    results : list[RunResult]
        Benchmark results.
    group_by : str
        "optimizer" or "reward" — determines color grouping.
    save_path : str or None
        If set, save figure to this path.
    """
    set_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for r in results:
        epochs = np.arange(len(r.reward_history))
        rewards = np.array(r.reward_history)
        color_map = OPTIMIZER_COLORS if group_by == "optimizer" else REWARD_COLORS
        key = r.optimizer_type if group_by == "optimizer" else r.reward_type
        color = color_map.get(key, "#333333")
        label = r.label

        ax.plot(epochs, rewards, color=color, label=label, alpha=0.8)

        if r.reward_std_history:
            stds = np.array(r.reward_std_history)
            ax.fill_between(
                epochs, rewards - stds, rewards + stds, color=color, alpha=0.1
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Proxy Reward")
    ax.set_title("Optimization Convergence")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_parameter_tracking(
    results, param_idx=0, param_name=r"Re($g_2$)", save_path=None
):
    """Plot optimizer parameter trajectory vs drift.

    Parameters
    ----------
    results : list[RunResult]
        Benchmark results.
    param_idx : int
        Which control parameter to plot (0-3).
    param_name : str
        LaTeX label for the parameter.
    save_path : str or None
    """
    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top: reward
    ax = axes[0]
    for r in results:
        epochs = np.arange(len(r.reward_history))
        color = OPTIMIZER_COLORS.get(r.optimizer_type, "#333")
        ax.plot(epochs, r.reward_history, color=color, label=r.label, alpha=0.8)
    ax.set_ylabel("Reward")
    ax.set_title("Reward Under Drift")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom: parameter tracking
    ax = axes[1]
    for r in results:
        epochs = np.arange(len(r.param_history))
        params = np.array(r.param_history)
        color = OPTIMIZER_COLORS.get(r.optimizer_type, "#333")
        ax.plot(
            epochs,
            params[:, param_idx],
            color=color,
            label=f"{r.optimizer_type} {param_name}",
            alpha=0.8,
        )

    # Plot drift trajectory if available
    if results and results[0].drift_offset_history:
        r = results[0]
        epochs = np.arange(len(r.drift_offset_history))
        drift_vals = [d.get("g2_re", 0.0) for d in r.drift_offset_history]
        # The "true" optimal g2_re = 1.0 + drift_offset (optimizer should track this)
        ax.plot(
            epochs,
            1.0 + np.array(drift_vals),
            "k--",
            label="Drifting optimum",
            linewidth=2,
            alpha=0.6,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(param_name)
    ax.set_title("Parameter Tracking vs Drift")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_lifetime_comparison(results, save_path=None):
    """Bar chart comparing final T_X, T_Z, bias across runs.

    Uses the last validation measurement from each result.

    Parameters
    ----------
    results : list[RunResult]
    save_path : str or None
    """
    set_plot_style()

    # Filter results that have validation data
    valid = [r for r in results if r.validation_history]
    if not valid:
        print("No validation data available for lifetime comparison.")
        return

    labels = [r.label for r in valid]
    tz_vals = [r.validation_history[-1]["Tz"] for r in valid]
    tx_vals = [r.validation_history[-1]["Tx"] for r in valid]
    bias_vals = [r.validation_history[-1]["bias"] for r in valid]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(labels))

    axes[0].barh(x, tz_vals, color="#2166AC", alpha=0.8)
    axes[0].set_xlabel(r"$T_Z$ [$\mu$s]")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_title(r"Bit-Flip Lifetime $T_Z$")

    axes[1].barh(x, tx_vals, color="#B2182B", alpha=0.8)
    axes[1].set_xlabel(r"$T_X$ [$\mu$s]")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_title(r"Phase-Flip Lifetime $T_X$")

    axes[2].barh(x, bias_vals, color="#4DAF4A", alpha=0.8)
    axes[2].set_xlabel(r"$\eta = T_Z / T_X$")
    axes[2].set_yticks(x)
    axes[2].set_yticklabels(labels, fontsize=7)
    axes[2].set_title(r"Bias Ratio $\eta$")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_reward_type_comparison(results, save_path=None):
    """Compare reward functions for the same optimizer.

    Filters results with the same optimizer, different reward types.
    """
    set_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for r in results:
        epochs = np.arange(len(r.reward_history))
        color = REWARD_COLORS.get(r.reward_type, "#333")
        style = DRIFT_STYLES.get(r.drift_type, "-")
        ax.plot(
            epochs,
            r.reward_history,
            color=color,
            linestyle=style,
            label=f"{r.reward_type} ({r.drift_type})",
            alpha=0.8,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Function Comparison")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_drift_tracking_matrix(results, save_path=None):
    """Multi-panel: rows = optimizers, cols = drift types.

    Each panel shows reward convergence for that combination.
    """
    set_plot_style()

    optimizers = sorted(set(r.optimizer_type for r in results))
    drifts = sorted(set(r.drift_type for r in results))

    if not optimizers or not drifts:
        print("Not enough data for drift tracking matrix.")
        return

    fig, axes = plt.subplots(
        len(optimizers),
        len(drifts),
        figsize=(4 * len(drifts), 3 * len(optimizers)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for i, opt in enumerate(optimizers):
        for j, drift in enumerate(drifts):
            ax = axes[i][j]
            matching = [
                r for r in results if r.optimizer_type == opt and r.drift_type == drift
            ]
            for r in matching:
                epochs = np.arange(len(r.reward_history))
                color = REWARD_COLORS.get(r.reward_type, "#333")
                ax.plot(
                    epochs,
                    r.reward_history,
                    color=color,
                    label=r.reward_type,
                    alpha=0.8,
                )

            if i == 0:
                ax.set_title(drift, fontsize=9)
            if j == 0:
                ax.set_ylabel(opt, fontsize=9)
            ax.grid(True, alpha=0.2)
            if i == 0 and j == 0:
                ax.legend(fontsize=6)

    fig.suptitle("Drift Tracking Matrix: Optimizer × Drift", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_summary_heatmap(results, metric="final_reward", save_path=None):
    """Heatmap of final metric across reward × optimizer combinations.

    Parameters
    ----------
    results : list[RunResult]
    metric : str
        "final_reward" or "final_bias" or "wall_time".
    save_path : str or None
    """
    set_plot_style()

    rewards = sorted(set(r.reward_type for r in results))
    optimizers = sorted(set(r.optimizer_type for r in results))

    matrix = np.full((len(rewards), len(optimizers)), np.nan)

    for r in results:
        i = rewards.index(r.reward_type)
        j = optimizers.index(r.optimizer_type)

        if metric == "final_reward":
            val = r.reward_history[-1] if r.reward_history else np.nan
        elif metric == "final_bias":
            if r.validation_history:
                val = r.validation_history[-1].get("bias", np.nan)
            else:
                val = np.nan
        elif metric == "wall_time":
            val = r.wall_time
        else:
            val = np.nan

        # Average over repeated runs
        if np.isnan(matrix[i, j]):
            matrix[i, j] = val
        else:
            matrix[i, j] = (matrix[i, j] + val) / 2

    fig, ax = plt.subplots(
        1, 1, figsize=(max(6, len(optimizers) * 1.5), max(4, len(rewards) * 1.2))
    )
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")

    ax.set_xticks(np.arange(len(optimizers)))
    ax.set_xticklabels(optimizers, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(rewards)))
    ax.set_yticklabels(rewards)

    # Annotate cells
    for i in range(len(rewards)):
        for j in range(len(optimizers)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val < np.nanmedian(matrix) else "black",
                    fontsize=8,
                )

    metric_labels = {
        "final_reward": "Final Reward",
        "final_bias": r"Final Bias $\eta$",
        "wall_time": "Wall Time [s]",
    }
    ax.set_title(f"Summary: {metric_labels.get(metric, metric)}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)
