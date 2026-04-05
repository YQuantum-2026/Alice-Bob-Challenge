"""Dashboard visualizations — optimizer-centric benchmark comparisons.

All plots center on comparing the 5 optimizers (CMA-ES, Bayesian, Hybrid,
REINFORCE, PPO-Clip) across drift scenarios.  The reward type is a
measurement methodology detail; optimizer and drift are the independent
variables.

Ref: Pack et al. (2025), arXiv:2509.08555 — optimizer benchmarking methodology.
Ref: Sivak et al. (2025), arXiv:2511.08493 — drift tracking evaluation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.plotting._style import (
    DRIFT_STYLES,
    OPTIMIZER_COLORS,
    REWARD_COLORS,
    set_plot_style,
)

# Drift scenarios ordered by severity (left-to-right = easy → hard)
DRIFT_SEVERITY_ORDER = [
    "none", "white_noise", "snr", "amplitude_slow", "amplitude_fast",
    "frequency", "frequency_step", "kerr", "step", "multi", "tls",
]

# Drift markers grouped for legibility (not 11 separate shapes)
DRIFT_MARKER_GROUPS = {
    "none": ("o", "No drift"),
    "amplitude_slow": ("s", "Amplitude"),
    "amplitude_fast": ("s", "Amplitude"),
    "frequency": ("D", "Frequency"),
    "frequency_step": ("D", "Frequency"),
    "kerr": ("^", "Kerr/Step"),
    "step": ("^", "Kerr/Step"),
    "snr": ("v", "SNR/Noise"),
    "white_noise": ("v", "SNR/Noise"),
    "multi": ("*", "Multi"),
    "tls": ("P", "TLS"),
}

# Human-readable drift descriptions for plot titles
DRIFT_DESCRIPTIONS = {
    "none": "No Drift (Baseline)",
    "white_noise": "White Noise",
    "snr": "SNR Degradation",
    "amplitude_slow": "Slow Amplitude Drift",
    "amplitude_fast": "Fast Amplitude Drift",
    "frequency": "Frequency Detuning",
    "frequency_step": "Square-Wave Frequency",
    "kerr": "Kerr Nonlinearity",
    "step": "Sudden Step Shift",
    "multi": "Multi-Drift (Combined)",
    "tls": "TLS Defect Coupling",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _filter_results(results, optimizer=None, reward=None, drift=None):
    """Filter RunResult list by any combination of axes."""
    out = results
    if optimizer:
        out = [r for r in out if r.optimizer_type == optimizer]
    if reward:
        out = [r for r in out if r.reward_type == reward]
    if drift:
        out = [r for r in out if r.drift_type == drift]
    return out


def _sort_drifts(drift_names):
    """Sort drift names by severity ordering. Unknown drifts go at the end."""
    known = [d for d in DRIFT_SEVERITY_ORDER if d in drift_names]
    unknown = sorted(drift_names - set(known))
    return known + unknown


# ---------------------------------------------------------------------------
# 1. Summary dashboard — the "big picture" 2×2 figure
# ---------------------------------------------------------------------------


def plot_summary_dashboard(results, save_path=None):
    """2x2 summary comparing all optimizers: reward, T_Z, speed, drift penalty.

    Each panel has one box per optimizer, aggregated across all drift scenarios.
    This is the primary figure for answering "which optimizer is best overall?"
    """
    set_plot_style()

    optimizers = sorted(set(r.optimizer_type for r in results))
    if not optimizers:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = [OPTIMIZER_COLORS.get(o, "#333") for o in optimizers]

    # (a) Final reward by optimizer
    ax = axes[0, 0]
    data_a = [
        [r.reward_history[-1] for r in results
         if r.optimizer_type == o and r.reward_history]
        for o in optimizers
    ]
    bp = ax.boxplot(data_a, labels=optimizers, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Final Reward")
    ax.set_title("(a) Final Reward")
    ax.grid(True, alpha=0.3, axis="y")

    # (b) T_Z by optimizer
    ax = axes[0, 1]
    data_b = []
    for o in optimizers:
        tzs = []
        for r in results:
            if r.optimizer_type == o and r.validation_history:
                v = r.validation_history[-1]
                if "Tz" in v:
                    tzs.append(v["Tz"])
        data_b.append(tzs if tzs else [0])
    bp = ax.boxplot(data_b, labels=optimizers, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel(r"$T_Z$ [$\mu$s]")
    ax.set_title(r"(b) Bit-Flip Lifetime $T_Z$")
    ax.grid(True, alpha=0.3, axis="y")

    # (c) Convergence speed
    ax = axes[1, 0]
    data_c = []
    for o in optimizers:
        speeds = []
        for r in results:
            if r.optimizer_type == o and r.reward_history:
                rh = np.array(r.reward_history)
                best = rh.max()
                threshold = 0.8 * best
                above = np.where(rh >= threshold)[0]
                speeds.append(int(above[0]) if len(above) > 0 else len(rh))
        data_c.append(speeds if speeds else [0])
    bp = ax.boxplot(data_c, labels=optimizers, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Epochs to 80% of Best")
    ax.set_title("(c) Convergence Speed")
    ax.grid(True, alpha=0.3, axis="y")

    # (d) Drift penalty
    ax = axes[1, 1]
    data_d = []
    for o in optimizers:
        no_drift = [
            r.reward_history[-1] for r in results
            if r.optimizer_type == o and r.drift_type == "none" and r.reward_history
        ]
        baseline = np.mean(no_drift) if no_drift else 0
        penalties = []
        for r in results:
            if r.optimizer_type == o and r.drift_type != "none" and r.reward_history:
                penalties.append(baseline - r.reward_history[-1])
        data_d.append(penalties if penalties else [0])
    bp = ax.boxplot(data_d, labels=optimizers, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Reward Drop from No-Drift")
    ax.set_title("(d) Drift Penalty")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Optimizer Benchmark Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Head-to-head optimizer convergence
# ---------------------------------------------------------------------------


def plot_optimizer_convergence(results, drift_type="none", save_path=None):
    """Head-to-head convergence: all optimizers on the same axes.

    This is the primary "which optimizer wins?" figure for a given drift
    scenario.  All optimizers are overlaid with their own color and std bands.

    Ref: Pack et al. (2025), arXiv:2509.08555, Fig. 1 — convergence comparison.
    """
    set_plot_style()

    filtered = _filter_results(results, drift=drift_type)
    if not filtered:
        print(f"[plots] No results for drift={drift_type}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    # Always show all 5 optimizers — plot data if available, legend entry either way
    all_optimizers = list(OPTIMIZER_COLORS.keys())
    for opt in all_optimizers:
        color = OPTIMIZER_COLORS[opt]
        opt_runs = _filter_results(filtered, optimizer=opt)

        if not opt_runs:
            # No data — still add to legend (dimmed)
            ax.plot([], [], color=color, label=f"{opt} (no data)", linewidth=1.8,
                    alpha=0.3, linestyle="--")
            continue

        # Use the run with the best final reward for this optimizer
        best = max(opt_runs, key=lambda r: r.reward_history[-1] if r.reward_history else -1e9)
        epochs = np.arange(len(best.reward_history))
        rh = np.array(best.reward_history)

        ax.plot(epochs, rh, color=color, label=opt, linewidth=1.8, alpha=0.9)

        if best.reward_std_history:
            stds = np.array(best.reward_std_history)
            ax.fill_between(epochs, rh - stds, rh + stds, color=color, alpha=0.12)

        # Annotate final reward on right margin
        final = rh[-1]
        ax.annotate(
            f"{final:.2f}", xy=(epochs[-1], final),
            xytext=(5, 0), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold",
            va="center",
        )

    drift_desc = DRIFT_DESCRIPTIONS.get(drift_type, drift_type)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title(f"Optimizer Convergence — {drift_desc}")
    ax.legend(fontsize=8, title="Optimizer")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


# Backward-compatible alias for run.py imports
plot_optimizer_comparison_panel = plot_optimizer_convergence


# ---------------------------------------------------------------------------
# 3. Per-drift scenario figure (convergence + drift trajectory)
# ---------------------------------------------------------------------------


def plot_drift_scenario(results, drift_type, save_path=None):
    """Two-panel comparison of all optimizers under a specific drift scenario.

    Top: Reward convergence for each optimizer (head-to-head).
    Bottom: Drift offset trajectory showing what the hardware is doing.

    Ref: Sivak et al. (2025), arXiv:2511.08493, Sec. IV — drift tracking.
    """
    set_plot_style()

    filtered = _filter_results(results, drift=drift_type)
    if not filtered:
        print(f"[plots] No results for drift={drift_type}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # --- Top: Optimizer convergence (all 5) ---
    ax = axes[0]
    all_optimizers = list(OPTIMIZER_COLORS.keys())
    for opt in all_optimizers:
        color = OPTIMIZER_COLORS[opt]
        opt_runs = _filter_results(filtered, optimizer=opt)

        if not opt_runs:
            ax.plot([], [], color=color, label=f"{opt} (no data)", linewidth=1.8,
                    alpha=0.3, linestyle="--")
            continue

        best = max(opt_runs, key=lambda r: r.reward_history[-1] if r.reward_history else -1e9)
        epochs = np.arange(len(best.reward_history))
        rh = np.array(best.reward_history)

        ax.plot(epochs, rh, color=color, label=opt, linewidth=1.8, alpha=0.9)
        if best.reward_std_history:
            stds = np.array(best.reward_std_history)
            ax.fill_between(epochs, rh - stds, rh + stds, color=color, alpha=0.12)

    ax.set_ylabel("Reward")
    ax.set_title("Optimizer Convergence")
    ax.legend(fontsize=8, title="Optimizer")
    ax.grid(True, alpha=0.3)

    # --- Bottom: Drift trajectory ---
    ax = axes[1]
    # Use the first result that has drift data to show the trajectory
    ref_run = next((r for r in filtered if r.drift_offset_history), None)
    if ref_run and ref_run.drift_offset_history:
        epochs = np.arange(len(ref_run.drift_offset_history))
        drift_fields = {
            r"$\delta g_{2,\mathrm{re}}$": "g2_re",
            r"$\delta g_{2,\mathrm{im}}$": "g2_im",
            r"$\delta \varepsilon_{d,\mathrm{re}}$": "eps_d_re",
            r"$\Delta$ (detuning)": "detuning",
            r"$K$ (Kerr)": "kerr",
            r"$\sigma_{\mathrm{SNR}}$": "snr_noise",
        }
        plot_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        ci = 0
        for label, key in drift_fields.items():
            vals = np.array([d.get(key, 0.0) for d in ref_run.drift_offset_history])
            if np.any(vals != 0):
                ax.plot(epochs, vals, label=label, color=plot_colors[ci % len(plot_colors)],
                        linewidth=1.3)
                ci += 1

        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, "No drift data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=10)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Drift Offset")
    ax.set_title("Hardware Drift Trajectory")
    ax.grid(True, alpha=0.3)

    drift_desc = DRIFT_DESCRIPTIONS.get(drift_type, drift_type)
    fig.suptitle(f"Drift Scenario: {drift_desc}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Drift robustness (performance vs drift severity)
# ---------------------------------------------------------------------------


def plot_drift_robustness(results, metric="final_reward", reward_type=None,
                          save_path=None):
    """Line plot: performance vs drift severity, one line per optimizer.

    Shows how each optimizer's performance degrades as drift becomes more severe.
    Drift scenarios are ordered left-to-right by increasing difficulty.

    Ref: Pack et al. (2025), arXiv:2509.08555 — robustness evaluation methodology.
    """
    set_plot_style()

    if reward_type:
        results = _filter_results(results, reward=reward_type)

    drifts = _sort_drifts(set(r.drift_type for r in results))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Always show all 5 optimizers
    all_optimizers = list(OPTIMIZER_COLORS.keys())
    for opt in all_optimizers:
        color = OPTIMIZER_COLORS[opt]
        means, errs, x_pos = [], [], []
        for j, drift in enumerate(drifts):
            matching = [
                r for r in results
                if r.optimizer_type == opt and r.drift_type == drift
            ]
            vals = []
            for r in matching:
                if metric == "final_reward" and r.reward_history:
                    vals.append(r.reward_history[-1])
                elif metric in ("Tz", "Tx", "bias") and r.validation_history:
                    v = r.validation_history[-1]
                    if metric in v:
                        vals.append(v[metric])
            if vals:
                means.append(np.mean(vals))
                errs.append(np.std(vals) if len(vals) > 1 else 0)
                x_pos.append(j)

        if means:
            ax.errorbar(
                x_pos, means, yerr=errs, color=color, label=opt,
                marker="o", linewidth=1.8, capsize=3, markersize=5,
            )
        else:
            ax.plot([], [], color=color, label=f"{opt} (no data)",
                    linewidth=1.8, alpha=0.3, linestyle="--")

    # Vertical separator between no-drift and drifted
    if "none" in drifts and len(drifts) > 1:
        ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.4)

    ax.set_xticks(range(len(drifts)))
    ax.set_xticklabels(drifts, rotation=30, ha="right", fontsize=8)
    metric_label = {"final_reward": "Final Reward", "Tz": r"$T_Z$ [$\mu$s]",
                    "Tx": r"$T_X$ [$\mu$s]", "bias": r"Bias $\eta$"}
    ax.set_ylabel(metric_label.get(metric, metric))
    ax.set_title("Drift Robustness: Optimizer Performance Across Scenarios")
    ax.legend(fontsize=8, title="Optimizer")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Lifetime scatter (T_Z vs T_X by optimizer)
# ---------------------------------------------------------------------------


def plot_lifetime_scatter(results, save_path=None):
    """Scatter T_Z vs T_X, colored by optimizer, shaped by drift group.

    Shows the bit-flip / phase-flip tradeoff achieved by each optimizer.
    Iso-bias lines mark constant eta = T_Z / T_X.
    """
    set_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for r in results:
        if not r.validation_history:
            continue
        v = r.validation_history[-1]
        if "Tz" not in v or "Tx" not in v:
            continue
        color = OPTIMIZER_COLORS.get(r.optimizer_type, "#333")
        marker_info = DRIFT_MARKER_GROUPS.get(r.drift_type, ("o", r.drift_type))
        ax.scatter(
            v["Tz"], v["Tx"], c=color, marker=marker_info[0],
            s=80, edgecolors="black", linewidth=0.5, zorder=5,
        )

    # Iso-bias lines
    xlim = ax.get_xlim()
    tz_range = np.linspace(0.1, max(xlim[1], 10), 100)
    for eta in [50, 100, 200]:
        ax.plot(
            tz_range, tz_range / eta, "--",
            color="gray", alpha=0.3, linewidth=0.8,
        )
        ax.annotate(
            f"$\\eta$={eta}",
            xy=(tz_range[-1], tz_range[-1] / eta),
            fontsize=6, color="gray", alpha=0.6,
        )

    # Optimizer legend (colors)
    opt_handles = []
    for opt, c in OPTIMIZER_COLORS.items():
        h = ax.scatter(
            [], [], c=c, s=40, edgecolors="black", linewidth=0.5, label=opt,
        )
        opt_handles.append(h)
    leg1 = ax.legend(
        handles=opt_handles, fontsize=7, title="Optimizer", loc="upper left",
    )
    ax.add_artist(leg1)

    # Drift legend (markers) — deduplicated by group name
    seen = set()
    drift_handles = []
    for _drift_type, (marker, group_name) in DRIFT_MARKER_GROUPS.items():
        if group_name not in seen:
            seen.add(group_name)
            h = ax.scatter(
                [], [], c="gray", marker=marker, s=40,
                edgecolors="black", linewidth=0.5, label=group_name,
            )
            drift_handles.append(h)
    ax.legend(handles=drift_handles, fontsize=7, title="Drift", loc="lower right")
    ax.add_artist(leg1)

    ax.set_xlabel(r"$T_Z$ [$\mu$s] (bit-flip)")
    ax.set_ylabel(r"$T_X$ [$\mu$s] (phase-flip)")
    ax.set_title(r"Lifetime Scatter: $T_Z$ vs $T_X$ by Optimizer")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Per-optimizer detail page — one optimizer across ALL drifts
# ---------------------------------------------------------------------------


def plot_optimizer_detail_page(results, optimizer, save_path=None):
    """2x3 detail page for a single optimizer across all drift conditions.

    (a) Convergence curves colored by drift scenario
    (b) Final reward bar chart by drift (severity order)
    (c) Best-run parameter trajectories + drift overlay
    (d) T_Z by drift scenario
    (e) T_X by drift scenario
    (f) Bias ratio (T_Z / T_X) by drift scenario
    """
    set_plot_style()

    filtered = _filter_results(results, optimizer=optimizer)
    if not filtered:
        print(f"[plots] No results for optimizer={optimizer}")
        return

    drifts = _sort_drifts(set(r.drift_type for r in filtered))
    drift_cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(drifts), 1)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # (a) Convergence across drifts
    ax = axes[0, 0]
    for j, drift in enumerate(drifts):
        for r in _filter_results(filtered, drift=drift):
            epochs = np.arange(len(r.reward_history))
            style = DRIFT_STYLES.get(drift, "-")
            ax.plot(
                epochs, r.reward_history, color=drift_cmap[j],
                linestyle=style if isinstance(style, str) else "-",
                label=drift, alpha=0.8,
            )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title("(a) Convergence by Drift")
    ax.grid(True, alpha=0.3)

    # (b) Final reward by drift scenario
    ax = axes[0, 1]
    drift_rewards = []
    for drift in drifts:
        matching = _filter_results(filtered, drift=drift)
        fr = [r.reward_history[-1] for r in matching if r.reward_history]
        drift_rewards.append(np.mean(fr) if fr else 0)
    x = np.arange(len(drifts))
    ax.bar(x, drift_rewards, color=[drift_cmap[i] for i in range(len(drifts))],
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(drifts, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Final Reward")
    ax.set_title("(b) Final Reward by Drift")
    ax.grid(True, alpha=0.3, axis="y")

    # (c) Parameter trajectories (best run)
    ax = axes[0, 2]
    best_run = max(
        filtered,
        key=lambda r: r.reward_history[-1] if r.reward_history else -1e9,
    )
    if best_run.param_history:
        ph = np.array(best_run.param_history)
        epochs = np.arange(len(ph))
        param_labels = [
            r"Re($g_2$)", r"Im($g_2$)",
            r"Re($\varepsilon_d$)", r"Im($\varepsilon_d$)",
        ]
        param_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, (pl, pc) in enumerate(zip(param_labels, param_colors)):
            ax.plot(epochs, ph[:, i], label=pl, color=pc, linewidth=1.2)
        # Overlay drift trajectory if applicable
        if best_run.drift_offset_history:
            drift_g2_re = np.array([d.get("g2_re", 0) for d in best_run.drift_offset_history])
            if np.any(drift_g2_re != 0):
                ax.plot(epochs[:len(drift_g2_re)], 1.0 + drift_g2_re,
                        "--", color="#1f77b4", alpha=0.4, linewidth=2,
                        label=r"Re($g_2$) optimal")
        ax.legend(fontsize=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter Value")
    ax.set_title(f"(c) Best Run Parameters ({best_run.drift_type})")
    ax.grid(True, alpha=0.3)

    # (d) T_Z across drifts
    ax = axes[1, 0]
    tz_vals, tz_labels = [], []
    for drift in drifts:
        for r in _filter_results(filtered, drift=drift):
            if r.validation_history and "Tz" in r.validation_history[-1]:
                tz_vals.append(r.validation_history[-1]["Tz"])
                tz_labels.append(drift)
    if tz_vals:
        x = np.arange(len(tz_vals))
        ax.bar(
            x, tz_vals,
            color=[drift_cmap[drifts.index(d)] for d in tz_labels],
            edgecolor="black", linewidth=0.3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(tz_labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(r"$T_Z$ [$\mu$s]")
    ax.set_title(r"(d) $T_Z$ by Drift")
    ax.grid(True, alpha=0.3, axis="y")

    # (e) T_X across drifts
    ax = axes[1, 1]
    tx_vals, tx_labels = [], []
    for drift in drifts:
        for r in _filter_results(filtered, drift=drift):
            if r.validation_history and "Tx" in r.validation_history[-1]:
                tx_vals.append(r.validation_history[-1]["Tx"])
                tx_labels.append(drift)
    if tx_vals:
        x = np.arange(len(tx_vals))
        ax.bar(
            x, tx_vals,
            color=[drift_cmap[drifts.index(d)] for d in tx_labels],
            edgecolor="black", linewidth=0.3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(tx_labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(r"$T_X$ [$\mu$s]")
    ax.set_title(r"(e) $T_X$ by Drift")
    ax.grid(True, alpha=0.3, axis="y")

    # (f) Bias ratio (T_Z / T_X) by drift
    ax = axes[1, 2]
    bias_vals, bias_labels = [], []
    for drift in drifts:
        for r in _filter_results(filtered, drift=drift):
            if r.validation_history and "bias" in r.validation_history[-1]:
                bias_vals.append(r.validation_history[-1]["bias"])
                bias_labels.append(drift)
    if bias_vals:
        x = np.arange(len(bias_vals))
        ax.bar(
            x, bias_vals,
            color=[drift_cmap[drifts.index(d)] for d in bias_labels],
            edgecolor="black", linewidth=0.3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(bias_labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(r"Bias $\eta = T_Z / T_X$")
    ax.set_title(r"(f) Bias Ratio by Drift")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Optimizer Detail: {optimizer}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Backward-compatible stubs (keep run.py imports working)
# ---------------------------------------------------------------------------


def plot_reward_effectiveness(results, optimizer="cmaes", save_path=None):
    """Deprecated — redirects to plot_optimizer_convergence."""
    plot_optimizer_convergence(results, drift_type="none", save_path=save_path)
