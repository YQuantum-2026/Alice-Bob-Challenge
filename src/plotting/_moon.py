from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.plotting._style import (
    set_plot_style,
)

# ---------------------------------------------------------------------------
# Moon cat comparison plots
# ---------------------------------------------------------------------------


def plot_moon_cat_convergence(
    moon_results: dict,
    save_path: str | None = None,
) -> None:
    """Plot reward convergence for standard cat (4D) vs moon cat (5D).

    Shows mean reward per epoch with shaded standard-deviation bands for
    both the standard and moon cat optimization runs.

    Parameters
    ----------
    moon_results : dict
        Keys "standard" and "moon", values are RunResult dataclass instances
        from ``run_moon_cat_comparison``.
    save_path : str or None
        If set, save figure to this path.
    """
    set_plot_style()

    std_result = moon_results.get("standard")
    moon_result = moon_results.get("moon")

    if std_result is None or moon_result is None:
        print("[warn] Missing standard or moon result for convergence plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Standard cat (4D)
    epochs_std = np.arange(len(std_result.reward_history))
    rewards_std = np.array(std_result.reward_history)
    ax.plot(
        epochs_std,
        rewards_std,
        color="#2166AC",
        label="Standard Cat (4D)",
        linewidth=1.5,
    )
    if std_result.reward_std_history:
        stds_std = np.array(std_result.reward_std_history)
        ax.fill_between(
            epochs_std,
            rewards_std - stds_std,
            rewards_std + stds_std,
            color="#2166AC",
            alpha=0.15,
        )

    # Moon cat (5D)
    epochs_moon = np.arange(len(moon_result.reward_history))
    rewards_moon = np.array(moon_result.reward_history)
    ax.plot(
        epochs_moon,
        rewards_moon,
        color="#ff7f0e",
        label="Moon Cat (5D)",
        linewidth=1.5,
    )
    if moon_result.reward_std_history:
        stds_moon = np.array(moon_result.reward_std_history)
        ax.fill_between(
            epochs_moon,
            rewards_moon - stds_moon,
            rewards_moon + stds_moon,
            color="#ff7f0e",
            alpha=0.15,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Moon Cat vs Standard Cat Convergence")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[plots] {save_path}")
    plt.close(fig)


def plot_moon_cat_lifetimes(
    moon_results: dict,
    save_path: str | None = None,
) -> None:
    """Grouped bar chart comparing T_Z, T_X, and bias for standard vs moon cat.

    Extracts the last validation measurement from each RunResult and plots
    three groups (TZ, TX, Bias) with two bars each (standard, moon).

    Parameters
    ----------
    moon_results : dict
        Keys "standard" and "moon", values are RunResult dataclass instances
        from ``run_moon_cat_comparison``.
    save_path : str or None
        If set, save figure to this path.
    """
    set_plot_style()

    std_result = moon_results.get("standard")
    moon_result = moon_results.get("moon")

    if std_result is None or moon_result is None:
        print("[warn] Missing standard or moon result for lifetime plot.")
        return

    # Check that both have validation data
    if not std_result.validation_history or not moon_result.validation_history:
        print("[warn] No validation data available for moon cat lifetime comparison.")
        return

    v_std = std_result.validation_history[-1]
    v_moon = moon_result.validation_history[-1]

    metrics = ["Tz", "Tx", "bias"]
    metric_labels = [r"$T_Z$ [$\mu$s]", r"$T_X$ [$\mu$s]", r"Bias $\eta$"]
    fmt_specs = [".1f", ".4f", ".0f"]

    std_vals = [v_std[m] for m in metrics]
    moon_vals = [v_moon[m] for m in metrics]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.3

    bars_std = ax.bar(
        x - width / 2,
        std_vals,
        width,
        label="Standard Cat (4D)",
        color="#2166AC",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    bars_moon = ax.bar(
        x + width / 2,
        moon_vals,
        width,
        label="Moon Cat (5D)",
        color="#ff7f0e",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    # Annotate bars with values
    for bar, val, fmt in zip(bars_std, std_vals, fmt_specs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar, val, fmt in zip(bars_moon, moon_vals, fmt_specs):
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
    ax.set_yscale("log")
    ax.set_ylabel("Value (log scale)")
    ax.set_title("Moon Cat Lifetime Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[plots] {save_path}")
    plt.close(fig)
