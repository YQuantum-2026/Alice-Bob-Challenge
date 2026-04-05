from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.plotting._style import (
    set_plot_style,
)

# ---------------------------------------------------------------------------
# Weight sweep and reward comparison plots
# ---------------------------------------------------------------------------


def plot_weight_sweep_heatmap(
    sweep_results: list,
    x_weight: str,
    y_weight: str,
    metric: str = "tz_times_tx",
    save_path: str | None = None,
) -> None:
    """2D heatmap of a metric over two swept reward weights.

    Parameters
    ----------
    sweep_results : list of (weights_dict, RunResult, ground_truth_dict)
        Output from ``run_weight_sweep``.
    x_weight : str
        Name of the weight to use as the x-axis.
    y_weight : str
        Name of the weight to use as the y-axis.
    metric : str
        One of "tz_times_tx", "tz", "tx", "bias", "final_reward".
    save_path : str or None
        If set, save figure to this path.
    """
    set_plot_style()

    # Extract unique sorted values for x and y axes
    x_vals = sorted(set(w[x_weight] for w, _, _ in sweep_results))
    y_vals = sorted(set(w[y_weight] for w, _, _ in sweep_results))

    # Build matrix
    matrix = np.full((len(y_vals), len(x_vals)), np.nan)
    for weights_dict, result, gt in sweep_results:
        xi = x_vals.index(weights_dict[x_weight])
        yi = y_vals.index(weights_dict[y_weight])

        if metric == "tz_times_tx":
            matrix[yi, xi] = gt["Tz"] * gt["Tx"]
        elif metric == "tz":
            matrix[yi, xi] = gt["Tz"]
        elif metric == "tx":
            matrix[yi, xi] = gt["Tx"]
        elif metric == "bias":
            matrix[yi, xi] = gt["bias"]
        elif metric == "final_reward":
            matrix[yi, xi] = (
                result.reward_history[-1] if result.reward_history else np.nan
            )

    # Choose colormap
    cmap = "RdBu_r" if metric == "bias" else "viridis"

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(max(6, len(x_vals) * 1.2), max(5, len(y_vals) * 1.0)),
    )
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{v:.2g}" for v in x_vals], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels([f"{v:.2g}" for v in y_vals])
    ax.set_xlabel(x_weight)
    ax.set_ylabel(y_weight)

    # Annotate cells
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            val = matrix[i, j]
            if np.isfinite(val):
                text_color = "white" if val < np.nanmedian(matrix) else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )

    metric_labels = {
        "tz_times_tx": r"$T_Z \times T_X$",
        "tz": r"$T_Z$ [$\mu$s]",
        "tx": r"$T_X$ [$\mu$s]",
        "bias": r"Bias $\eta = T_Z / T_X$",
        "final_reward": "Final Reward",
    }
    ax.set_title(f"Weight Sweep: {metric_labels.get(metric, metric)}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[plots] {save_path}")
    plt.show()
    plt.close(fig)


def plot_reward_correlation_matrix(
    rewards_dict: dict[str, list[float]],
    save_path: str | None = None,
) -> None:
    """Pairwise Spearman rank correlation matrix between reward functions.

    Parameters
    ----------
    rewards_dict : dict
        Maps reward_name -> list of scalar reward values, all evaluated at
        the same parameter points. Lists must have equal length.
    save_path : str or None
        If set, save figure to this path.
    """
    from scipy.stats import spearmanr

    set_plot_style()

    names = list(rewards_dict.keys())
    n = len(names)
    values = np.array([rewards_dict[k] for k in names])  # (n_rewards, n_points)

    # Compute pairwise Spearman correlations
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(values[i], values[j])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(max(6, n * 1.2), max(5, n * 1.0)),
    )
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(names)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    ax.set_title("Reward Function Correlation (Spearman)")
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"Spearman $\rho$")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[plots] {save_path}")
    plt.show()
    plt.close(fig)


def plot_enhanced_vs_proxy(
    enhanced_result,
    proxy_result,
    save_path: str | None = None,
) -> None:
    """Two-panel comparison of enhanced_proxy vs proxy reward optimization.

    Top panel: reward convergence curves with std bands.
    Bottom panel: side-by-side T_Z, T_X bar chart from validation_history.

    Parameters
    ----------
    enhanced_result : RunResult
        Result from running CMA-ES with reward_type="enhanced_proxy".
    proxy_result : RunResult
        Result from running CMA-ES with reward_type="proxy".
    save_path : str or None
        If set, save figure to this path.
    """
    set_plot_style()

    proxy_color = "#1f77b4"
    enhanced_color = "#17becf"

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    # --- Top: Convergence curves ---
    ax = axes[0]
    for result, color, label in [
        (proxy_result, proxy_color, "Proxy"),
        (enhanced_result, enhanced_color, "Enhanced Proxy"),
    ]:
        epochs = np.arange(len(result.reward_history))
        rewards = np.array(result.reward_history)
        ax.plot(epochs, rewards, color=color, label=label, linewidth=1.5)
        if result.reward_std_history:
            stds = np.array(result.reward_std_history)
            ax.fill_between(
                epochs,
                rewards - stds,
                rewards + stds,
                color=color,
                alpha=0.15,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Convergence: Enhanced Proxy vs Proxy")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # --- Bottom: Lifetime comparison bars ---
    ax = axes[1]
    has_proxy_val = bool(proxy_result.validation_history)
    has_enhanced_val = bool(enhanced_result.validation_history)

    if has_proxy_val and has_enhanced_val:
        v_proxy = proxy_result.validation_history[-1]
        v_enhanced = enhanced_result.validation_history[-1]

        metrics = ["Tz", "Tx"]
        metric_labels = [r"$T_Z$ [$\mu$s]", r"$T_X$ [$\mu$s]"]
        proxy_vals = [v_proxy[m] for m in metrics]
        enhanced_vals = [v_enhanced[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.3

        bars_p = ax.bar(
            x - width / 2,
            proxy_vals,
            width,
            label="Proxy",
            color=proxy_color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        bars_e = ax.bar(
            x + width / 2,
            enhanced_vals,
            width,
            label="Enhanced Proxy",
            color=enhanced_color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

        # Annotate bars
        for bar, val in zip(bars_p, proxy_vals):
            fmt = ".1f" if val > 1 else ".4f"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:{fmt}}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        for bar, val in zip(bars_e, enhanced_vals):
            fmt = ".1f" if val > 1 else ".4f"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:{fmt}}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel(r"Lifetime [$\mu$s]")
        ax.set_title("Final Lifetime Comparison")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(
            0.5,
            0.5,
            "No validation data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Final Lifetime Comparison (no data)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[plots] {save_path}")
    plt.show()
    plt.close(fig)
