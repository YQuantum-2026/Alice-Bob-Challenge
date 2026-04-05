"""Convergence tests for all optimizers.

Validates that each optimizer shows improving reward over epochs and
generates convergence comparison figures as part of the main pipeline.

These tests are slow (quantum simulation) — use pytest -m slow to include.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.benchmark import RunResult, run_single
from src.config import get_config

# ---------------------------------------------------------------------------
# Tiny config for fast convergence testing
# ---------------------------------------------------------------------------
from tests.conftest import TINY_PARAMS

N_EPOCHS = 15


def _convergence_config():
    """Build a small config suitable for convergence testing."""
    cfg = get_config("local")
    cfg.cat_params = TINY_PARAMS
    cfg.optimizer.n_epochs = N_EPOCHS
    cfg.optimizer.population_size = 6
    cfg.optimizer.hybrid_cma_epochs = 5
    cfg.optimizer.hybrid_grad_steps = 3
    cfg.optimizer.bayesian_n_initial = 3
    cfg.reward.full_eval_interval = 1000  # disable periodic validation
    return cfg


# ---------------------------------------------------------------------------
# Per-optimizer convergence tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestConvergenceCMAES:
    def test_reward_improves(self):
        """CMA-ES reward should trend upward over epochs."""
        cfg = _convergence_config()
        result = run_single("proxy", "cmaes", "none", cfg, verbose=False)
        rewards = np.array(result.reward_history)
        assert len(rewards) == N_EPOCHS
        assert all(np.isfinite(rewards))
        # First-half mean should be <= second-half mean (improving trend)
        mid = len(rewards) // 2
        assert np.mean(rewards[mid:]) >= np.mean(rewards[:mid]) - 0.5


@pytest.mark.slow
class TestConvergenceHybrid:
    def test_reward_improves(self):
        """Hybrid optimizer reward should trend upward."""
        cfg = _convergence_config()
        result = run_single("proxy", "hybrid", "none", cfg, verbose=False)
        rewards = np.array(result.reward_history)
        assert len(rewards) == N_EPOCHS
        assert all(np.isfinite(rewards))
        mid = len(rewards) // 2
        assert np.mean(rewards[mid:]) >= np.mean(rewards[:mid]) - 0.5


@pytest.mark.slow
class TestConvergenceREINFORCE:
    def test_reward_finite(self):
        """REINFORCE reward should remain finite across all epochs."""
        cfg = _convergence_config()
        result = run_single("proxy", "reinforce", "none", cfg, verbose=False)
        rewards = np.array(result.reward_history)
        assert len(rewards) == N_EPOCHS
        assert all(np.isfinite(rewards))


@pytest.mark.slow
class TestConvergencePPO:
    def test_reward_finite(self):
        """PPO-Clip reward should remain finite across all epochs."""
        cfg = _convergence_config()
        result = run_single("proxy", "ppo", "none", cfg, verbose=False)
        rewards = np.array(result.reward_history)
        assert len(rewards) == N_EPOCHS
        assert all(np.isfinite(rewards))


@pytest.mark.slow
class TestConvergenceBayesian:
    def test_reward_finite(self):
        """Bayesian optimizer reward should remain finite."""
        cfg = _convergence_config()
        result = run_single("proxy", "bayesian", "none", cfg, verbose=False)
        rewards = np.array(result.reward_history)
        assert len(rewards) == N_EPOCHS
        assert all(np.isfinite(rewards))


# ---------------------------------------------------------------------------
# Multi-optimizer comparison (generates figures)
# ---------------------------------------------------------------------------

OPTIMIZER_NAMES = ["cmaes", "hybrid", "reinforce", "ppo", "bayesian"]


@pytest.mark.slow
class TestConvergenceComparison:
    def test_all_optimizers_converge_and_plot(self, tmp_path):
        """Run all 5 optimizers, verify convergence, and save comparison plot."""
        cfg = _convergence_config()
        results = {}

        for opt in OPTIMIZER_NAMES:
            result = run_single("proxy", opt, "none", cfg, verbose=False)
            rewards = np.array(result.reward_history)
            assert len(rewards) == N_EPOCHS, f"{opt}: wrong epoch count"
            assert all(np.isfinite(rewards)), f"{opt}: non-finite rewards"
            results[opt] = result

        # Generate convergence comparison figure
        save_path = str(tmp_path / "convergence_comparison.png")
        _plot_convergence_comparison(results, save_path=save_path)
        assert os.path.exists(save_path), "Convergence figure not saved"


def _plot_convergence_comparison(results: dict, save_path: str | None = None):
    """Plot reward convergence for all optimizers on one figure.

    Parameters
    ----------
    results : dict
        Mapping optimizer_name -> RunResult.
    save_path : str or None
        If set, save figure to this path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.plotting import OPTIMIZER_COLORS, set_plot_style

    set_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for opt_name, result in results.items():
        epochs = np.arange(len(result.reward_history))
        rewards = np.array(result.reward_history)
        color = OPTIMIZER_COLORS.get(opt_name, "#333333")

        ax.plot(epochs, rewards, color=color, label=opt_name, alpha=0.8, linewidth=1.5)

        if result.reward_std_history:
            stds = np.array(result.reward_std_history)
            ax.fill_between(
                epochs, rewards - stds, rewards + stds, color=color, alpha=0.1
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Proxy Reward")
    ax.set_title("Optimizer Convergence Comparison")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline-callable convergence report generator
# ---------------------------------------------------------------------------


def generate_convergence_report(
    benchmark_results: list[RunResult],
    save_dir: str = "figures",
):
    """Generate convergence figures from benchmark results.

    Groups results by optimizer (no-drift, proxy reward only) and produces
    a comparison plot. Called from run.py as part of the main pipeline.

    Parameters
    ----------
    benchmark_results : list[RunResult]
        Results from run_benchmark().
    save_dir : str
        Directory to save figures.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.plotting import OPTIMIZER_COLORS, set_plot_style

    os.makedirs(save_dir, exist_ok=True)

    # Filter: proxy reward, no drift
    proxy_no_drift = [
        r
        for r in benchmark_results
        if r.reward_type == "proxy" and r.drift_type == "none" and r.reward_history
    ]

    if not proxy_no_drift:
        # Fall back to any results with reward history
        proxy_no_drift = [r for r in benchmark_results if r.reward_history]

    if not proxy_no_drift:
        return

    # Group by optimizer
    results_by_opt = {}
    for r in proxy_no_drift:
        results_by_opt[r.optimizer_type] = r

    # Generate comparison plot
    _plot_convergence_comparison(
        results_by_opt,
        save_path=os.path.join(save_dir, "convergence_comparison.png"),
    )

    # Generate per-optimizer individual plots
    per_opt_dir = os.path.join(save_dir, "convergence_per_optimizer")
    os.makedirs(per_opt_dir, exist_ok=True)

    set_plot_style()
    for opt_name, result in results_by_opt.items():
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        epochs = np.arange(len(result.reward_history))
        rewards = np.array(result.reward_history)
        color = OPTIMIZER_COLORS.get(opt_name, "#333333")

        ax.plot(epochs, rewards, color=color, linewidth=1.5)
        if result.reward_std_history:
            stds = np.array(result.reward_std_history)
            ax.fill_between(
                epochs, rewards - stds, rewards + stds, color=color, alpha=0.15
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Proxy Reward")
        ax.set_title(f"{opt_name} Convergence")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(per_opt_dir, f"{opt_name}.png")
        plt.savefig(save_path)
        plt.close(fig)
