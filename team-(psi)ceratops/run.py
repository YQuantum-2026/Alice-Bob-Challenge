#!/usr/bin/env python
"""Run the cat qubit optimization benchmark.

Usage:
  python run.py                          # MEDIUM profile, defaults
  python run.py --profile hpc            # HPC profile
  python run.py --profile local          # fast dev
  python run.py --enable rewards,drift,moon_cat  # select sections
  python run.py --enable all             # everything
  python run.py --sweep                  # full parameter sweep (HPC)

Sections:
  rewards: Reward function comparison (proxy, photon, fidelity, parity)
  optimizers: Optimizer comparison (CMA-ES, hybrid, PPO, Bayesian)
  drift: Drift tracking (amplitude, frequency, Kerr, SNR, multi, step)
  moon_cat: Moon cat extension (Rousseau et al. 2025, arXiv:2502.07892)
  gates: Single-qubit gates (Zeno gate, arXiv:2204.09128)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime

# ============================================================
# CONFIGURATION — Edit these to control what runs
# ============================================================
PROFILE = "local"  # "local", "medium", or "hpc"
SAVE_RESULTS = True
GENERATE_PLOTS = True
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cat qubit optimization benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profile", type=str, default=None, help="Scale profile: local, medium, hpc"
    )
    parser.add_argument(
        "--enable",
        type=str,
        default=None,
        help="Comma-separated sections to enable: rewards,optimizers,drift,moon_cat,gates,all",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Full parameter sweep (all rewards x optimizers x drifts)",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default=None,
        help="Comma-separated optimizer list, e.g. 'cmaes,hybrid,reinforce,ppo,bayesian'",
    )
    parser.add_argument(
        "--drifts",
        type=str,
        default=None,
        help="Comma-separated drift list, e.g. 'none,amplitude_slow,frequency'",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive plot windows (for SLURM/headless)",
    )
    args = parser.parse_args()

    profile = args.profile or PROFILE
    save = not args.no_save and SAVE_RESULTS
    plots = not args.no_plots and GENERATE_PLOTS
    interactive = not args.no_interactive

    return profile, save, plots, args.sweep, interactive, args.enable, args.optimizers, args.drifts


def get_git_hash():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def print_env_info(cfg):
    """Print detailed environment and configuration for debug/SLURM logs."""
    import jax

    print("=" * 70)
    print("  YQuantum 2026 — Cat Qubit Online Optimization Benchmark")
    print("=" * 70)
    print()

    # Environment
    print("[env] Python:      ", sys.version.split()[0])
    print("[env] Platform:    ", platform.platform())
    print("[env] Git hash:    ", get_git_hash())
    print("[env] Timestamp:   ", datetime.now().isoformat())
    print("[env] JAX devices: ", jax.devices())
    print("[env] Working dir: ", os.getcwd())
    print()

    # Library versions
    import cmaes as cmaes_lib
    import dynamiqs

    print("[lib] dynamiqs:    ", dynamiqs.__version__)
    print("[lib] jax:         ", jax.__version__)
    print("[lib] cmaes:       ", cmaes_lib.__version__)
    try:
        import optax

        print("[lib] optax:       ", optax.__version__)
    except Exception:
        print("[lib] optax:        not installed")
    print()

    # Configuration
    p = cfg.cat_params
    o = cfg.optimizer
    r = cfg.reward
    b = cfg.benchmark

    print("[cfg] Profile:       ", cfg.name)
    print(f"[cfg] Precision:      {'float64' if p.use_double else 'float32'}")
    print(f"[cfg] Hilbert space:  na={p.na}, nb={p.nb} (dim={p.na * p.nb})")
    print(f"[cfg] Hardware:       kappa_b={p.kappa_b} MHz, kappa_a={p.kappa_a} MHz")
    print()
    print(
        f"[cfg] CMA-ES:         pop={o.population_size}, epochs={o.n_epochs}, "
        f"sigma0={o.sigma0}, floor={o.sigma_floor}"
    )
    print(f"[cfg] Hybrid lr:      lr={o.learning_rate}")
    print(
        f"[cfg] Hybrid:         cma_epochs={o.hybrid_cma_epochs}, "
        f"grad_steps={o.hybrid_grad_steps}"
    )
    print(
        f"[cfg] Policy (RL):    lr_mu={o.policy_lr_mean}, lr_sigma={o.policy_lr_sigma}, "
        f"beta={o.policy_beta_entropy}"
    )
    print(
        f"[cfg] Bayesian:       n_initial={o.bayesian_n_initial}, "
        f"acq={o.bayesian_acq_func}"
    )
    print()
    print(f"[cfg] Reward probe:   t_z={r.t_probe_z} us, t_x={r.t_probe_x} us")
    print(f"[cfg] Target bias:    eta={r.target_bias}")
    print(f"[cfg] Weights:        w_lifetime={r.w_lifetime}, w_bias={r.w_bias}")
    print(f"[cfg] Photon target:  n_target={r.n_target}")
    print(
        f"[cfg] Validation:     every {r.full_eval_interval} epochs, "
        f"tfinal_z={r.tfinal_z} us, tfinal_x={r.tfinal_x} us"
    )
    print()
    enabled = [
        name
        for name, flag in [
            ("rewards", b.enable_reward_sweep),
            ("optimizers", b.enable_optimizer_sweep),
            ("drift", b.enable_drift_sweep),
            ("moon_cat", b.enable_moon_cat),
            ("gates", b.enable_gates),
        ]
        if flag
    ]
    print(f"[cfg] Enabled:        {enabled}")
    print(f"[cfg] Rewards:        {b.rewards}")
    print(f"[cfg] Optimizers:     {b.optimizers}")
    print(f"[cfg] Drifts:         {b.drifts}")
    n_combos = len(b.rewards) * len(b.optimizers) * len(b.drifts)
    print(
        f"[cfg] Sweep size:     {len(b.rewards)} x {len(b.optimizers)} x "
        f"{len(b.drifts)} = {n_combos} combinations"
    )
    if b.enable_moon_cat:
        mc = cfg.moon_cat
        print(
            f"[cfg] Moon cat:       lambda=[{mc.lambda_min}, {mc.lambda_max}], "
            f"init={mc.lambda_init}"
        )
    if b.enable_gates:
        g = cfg.gate
        print(
            f"[cfg] Gate:           eps_z=[{g.epsilon_z_min}, {g.epsilon_z_max}], "
            f"duration={g.gate_duration} us"
        )
    print()
    print("-" * 70)


def save_results_json(all_results: dict, profile: str, cfg) -> str:
    """Serialize results + config metadata to JSON."""
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/run_{profile}_{timestamp}.json"

    metadata = {
        "profile": profile,
        "timestamp": timestamp,
        "git_hash": get_git_hash(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": {
            "na": cfg.cat_params.na,
            "nb": cfg.cat_params.nb,
            "kappa_b": cfg.cat_params.kappa_b,
            "kappa_a": cfg.cat_params.kappa_a,
            "use_double": cfg.cat_params.use_double,
            "n_epochs": cfg.optimizer.n_epochs,
            "population_size": cfg.optimizer.population_size,
            "enable_reward_sweep": cfg.benchmark.enable_reward_sweep,
            "enable_optimizer_sweep": cfg.benchmark.enable_optimizer_sweep,
            "enable_drift_sweep": cfg.benchmark.enable_drift_sweep,
            "enable_moon_cat": cfg.benchmark.enable_moon_cat,
            "enable_gates": cfg.benchmark.enable_gates,
            "rewards": cfg.benchmark.rewards,
            "optimizers": cfg.benchmark.optimizers,
            "drifts": cfg.benchmark.drifts,
        },
    }

    serializable = {"metadata": metadata}
    for key, results in all_results.items():
        if isinstance(results, list):
            serializable[key] = [
                {
                    "reward_type": r.reward_type,
                    "optimizer_type": r.optimizer_type,
                    "drift_type": r.drift_type,
                    "reward_history": r.reward_history,
                    "reward_std_history": r.reward_std_history,
                    "param_history": [p.tolist() for p in r.param_history],
                    "validation_history": r.validation_history,
                    "drift_offset_history": r.drift_offset_history,
                    "wall_time": r.wall_time,
                    "n_epochs": r.n_epochs,
                }
                for r in results
            ]
        elif isinstance(results, dict):
            serializable[key] = {}
            for sub_key, r in results.items():
                serializable[key][sub_key] = {
                    "reward_type": r.reward_type,
                    "optimizer_type": r.optimizer_type,
                    "drift_type": r.drift_type,
                    "reward_history": r.reward_history,
                    "reward_std_history": r.reward_std_history,
                    "param_history": [p.tolist() for p in r.param_history],
                    "validation_history": r.validation_history,
                    "drift_offset_history": r.drift_offset_history,
                    "wall_time": r.wall_time,
                    "n_epochs": r.n_epochs,
                }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

    size_kb = os.path.getsize(path) / 1024
    print(f"\n[save] {path} ({size_kb:.0f} KB)")
    return path


def generate_plots(all_results: dict, cfg=None):
    """Generate all comparison plots to figures/."""
    os.makedirs("figures", exist_ok=True)
    print("\n[plots] Generating figures...")

    try:
        from src.plotting import (
            plot_alpha_evolution,
            plot_convergence_speed,
            plot_drift_detail,
            plot_drift_robustness,
            plot_drift_scenario,
            plot_drift_tracking_matrix,
            plot_efficiency_scatter,
            plot_lifetime_scatter,
            plot_logical_decay,
            plot_optimizer_convergence,
            plot_optimizer_detail_page,
            plot_parameter_tracking,
            plot_reward_convergence,
            plot_reward_type_comparison,
            plot_run_detail,
            plot_summary_dashboard,
            plot_summary_heatmap,
        )

        if "benchmark" in all_results:
            results = all_results["benchmark"]

            # Convergence by optimizer
            plot_reward_convergence(
                results,
                group_by="optimizer",
                save_path="figures/convergence_by_optimizer.png",
            )
            print("[plots]   figures/convergence_by_optimizer.png")

            # Convergence by reward type
            no_drift = [r for r in results if r.drift_type == "none"]
            if no_drift:
                plot_reward_type_comparison(
                    no_drift,
                    save_path="figures/reward_type_comparison.png",
                )
                print("[plots]   figures/reward_type_comparison.png")

            # Drift tracking
            with_drift = [r for r in results if r.drift_type != "none"]
            if with_drift:
                plot_parameter_tracking(
                    with_drift,
                    save_path="figures/drift_parameter_tracking.png",
                )
                print("[plots]   figures/drift_parameter_tracking.png")

                plot_drift_tracking_matrix(
                    results,
                    save_path="figures/drift_tracking_matrix.png",
                )
                print("[plots]   figures/drift_tracking_matrix.png")

            # Lifetime scatter (replaces flat bar chart)
            validated = [r for r in results if r.validation_history]
            if validated:
                plot_lifetime_scatter(
                    validated,
                    save_path="figures/dashboard_lifetime_scatter.png",
                )
                print("[plots]   figures/dashboard_lifetime_scatter.png")

            # Summary heatmap
            plot_summary_heatmap(
                results,
                metric="final_reward",
                save_path="figures/summary_heatmap_reward.png",
            )
            print("[plots]   figures/summary_heatmap_reward.png")

            # Logical X/Z decay for best params from each optimizer (no drift)
            proxy_nodrift = [
                r
                for r in results
                if r.drift_type == "none"
                and r.reward_type == "proxy"
                and r.param_history
            ]
            if proxy_nodrift:
                decay_params = []
                for r in proxy_nodrift:
                    best = r.param_history[-1]
                    decay_params.append(
                        {
                            "g2_re": float(best[0]),
                            "g2_im": float(best[1]),
                            "eps_d_re": float(best[2]),
                            "eps_d_im": float(best[3]),
                            "label": r.optimizer_type,
                        }
                    )
                # Add default params for comparison
                decay_params.insert(
                    0,
                    {
                        "g2_re": 1.0,
                        "g2_im": 0.0,
                        "eps_d_re": 4.0,
                        "eps_d_im": 0.0,
                        "label": "default",
                    },
                )
                plot_logical_decay(
                    decay_params,
                    cat_params=cfg.cat_params,
                    save_path="figures/logical_decay.png",
                )

            # Per-run detail plots (top 3 by final reward)
            top_runs = sorted(
                results,
                key=lambda r: r.reward_history[-1] if r.reward_history else -1e9,
                reverse=True,
            )[:3]
            for r in top_runs:
                label = f"{r.optimizer_type}_{r.reward_type}_{r.drift_type}"
                if r.drift_type == "none":
                    plot_run_detail(r, save_path=f"figures/detail_{label}.png")
                else:
                    plot_drift_detail(r, save_path=f"figures/drift_detail_{label}.png")

            # Comparison plots
            validated = [r for r in results if r.validation_history]
            if len(validated) >= 2:
                plot_alpha_evolution(validated, save_path="figures/alpha_evolution.png")
                print("[plots]   figures/alpha_evolution.png")

            if len(results) >= 2:
                plot_convergence_speed(
                    results, save_path="figures/convergence_speed.png"
                )
                print("[plots]   figures/convergence_speed.png")

                plot_efficiency_scatter(
                    results, save_path="figures/efficiency_scatter.png"
                )
                print("[plots]   figures/efficiency_scatter.png")

            # --- Dashboard plots ---

            # Summary: 2x2 box plots comparing all optimizers
            plot_summary_dashboard(
                results, save_path="figures/dashboard_summary.png"
            )
            print("[plots]   figures/dashboard_summary.png")

            # Head-to-head optimizer convergence for each drift scenario
            from src.plotting._dashboard import _sort_drifts
            drift_types_present = _sort_drifts(
                set(r.drift_type for r in results)
            )
            for dtype in drift_types_present:
                plot_optimizer_convergence(
                    results, drift_type=dtype,
                    save_path=f"figures/dashboard_convergence_{dtype}.png",
                )
                print(f"[plots]   figures/dashboard_convergence_{dtype}.png")

                # Per-drift detailed figure (convergence + drift trajectory)
                if dtype != "none":
                    plot_drift_scenario(
                        results, drift_type=dtype,
                        save_path=f"figures/dashboard_drift_{dtype}.png",
                    )
                    print(f"[plots]   figures/dashboard_drift_{dtype}.png")

            # Drift robustness ranking (all optimizers across drifts)
            if len(drift_types_present) > 1:
                plot_drift_robustness(
                    results, metric="final_reward",
                    save_path="figures/dashboard_drift_robustness.png",
                )
                print("[plots]   figures/dashboard_drift_robustness.png")

            # Per-optimizer detail pages
            optimizers_present = sorted(set(r.optimizer_type for r in results))
            for opt in optimizers_present:
                plot_optimizer_detail_page(
                    results, optimizer=opt,
                    save_path=f"figures/dashboard_detail_{opt}.png",
                )
                print(f"[plots]   figures/dashboard_detail_{opt}.png")

    except Exception as e:
        print(f"[plots] Error: {e}")

    # --- Convergence comparison report ---
    try:
        from tests.test_convergence import generate_convergence_report

        if "benchmark" in all_results:
            generate_convergence_report(all_results["benchmark"], save_dir="figures")
            print("[plots] Convergence comparison figures saved")
    except Exception as e:
        print(f"[plots] Convergence report skipped: {e}")


def print_summary(all_results: dict):
    """Print detailed results summary."""
    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    if "benchmark" in all_results:
        results = all_results["benchmark"]
        print(f"\n  Benchmark Sweep: {len(results)} runs completed")
        print(
            f"  {'Optimizer':<12} {'Reward':<10} {'Drift':<16} "
            f"{'Final R':>9} {'Best R':>9} {'Time':>7}"
        )
        print("  " + "-" * 67)
        for r in results:
            final = r.reward_history[-1] if r.reward_history else float("nan")
            best = max(r.reward_history) if r.reward_history else float("nan")
            print(
                f"  {r.optimizer_type:<12} {r.reward_type:<10} {r.drift_type:<16} "
                f"{final:>9.4f} {best:>9.4f} {r.wall_time:>6.1f}s"
            )

        # Validation results
        validated = [r for r in results if r.validation_history]
        if validated:
            print(
                f"\n  Lifetime Validation ({len(validated)} runs with full measurement):"
            )
            print(
                f"  {'Label':<40} {'T_Z [us]':>10} {'T_X [us]':>10} {'Bias':>10} {'alpha':>8}"
            )
            print("  " + "-" * 82)
            for r in validated:
                v = r.validation_history[-1]
                label = f"{r.optimizer_type}/{r.reward_type}/{r.drift_type}"
                print(
                    f"  {label:<40} {v['Tz']:>10.2f} {v['Tx']:>10.4f} "
                    f"{v['bias']:>10.1f} {v.get('alpha', 0):>8.3f}"
                )

    if "moon_cat" in all_results:
        mc = all_results["moon_cat"]
        print("\n  Moon Cat Comparison")
        print("  " + "-" * 50)
        for label, r in mc.items():
            final = r.reward_history[-1] if r.reward_history else float("nan")
            v = r.validation_history[-1] if r.validation_history else {}
            print(
                f"  {label:<15} reward={final:>8.4f}  T_Z={v.get('Tz', 'N/A'):>8}  "
                f"T_X={v.get('Tx', 'N/A'):>8}  bias={v.get('bias', 'N/A'):>8}"
            )

    if "gate" in all_results:
        gt = all_results["gate"]
        print("\n  Gate Benchmark")
        print("  " + "-" * 50)
        for label, r in gt.items():
            final = r.reward_history[-1] if r.reward_history else float("nan")
            print(f"  {label:<15} reward={final:>8.4f}  time={r.wall_time:>6.1f}s")

    total_time = sum(
        r.wall_time
        for v in all_results.values()
        for r in (v if isinstance(v, list) else v.values())
    )
    print(f"\n  Total: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)


def apply_sweep_config(cfg):
    """Configure for full parameter sweep (all combos)."""
    cfg.benchmark.rewards = ["proxy", "photon", "fidelity", "parity"]
    cfg.benchmark.optimizers = ["cmaes", "hybrid", "reinforce", "ppo", "bayesian"]
    cfg.benchmark.drifts = [
        "none",
        "amplitude_slow",
        "amplitude_fast",
        "frequency",
        "frequency_step",
        "kerr",
        "step",
        "snr",
        "multi",
        "white_noise",
    ]
    # Note: "tls" is excluded from sweep — gated by enable_tls flag, doubles Hilbert space
    cfg.benchmark.enable_reward_sweep = True
    cfg.benchmark.enable_optimizer_sweep = True
    cfg.benchmark.enable_drift_sweep = True
    cfg.benchmark.enable_moon_cat = True
    cfg.benchmark.enable_gates = True
    return cfg


def main():
    profile, save, plots, sweep, interactive, enable_str, opt_str, drift_str = parse_args()

    # Suppress SparseDIAQArray warnings globally
    warnings.filterwarnings(
        "ignore", message=".*SparseDIAQArray.*converted to a DenseQArray.*"
    )

    # Set matplotlib backend before any imports
    if not interactive:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if interactive:
        plt.ion()

    import dynamiqs as dq

    from src.config import get_config

    cfg = get_config(profile)

    # Apply --enable flag
    if enable_str:
        valid_sections = {"rewards", "optimizers", "drift", "moon_cat", "gates", "all"}
        sections = [s.strip() for s in enable_str.split(",")]
        for s in sections:
            if s not in valid_sections:
                print(
                    f"[error] Unknown --enable section: '{s}'. "
                    f"Valid: {sorted(valid_sections)}"
                )
                sys.exit(1)
        if "all" in sections:
            cfg.benchmark.enable_reward_sweep = True
            cfg.benchmark.enable_optimizer_sweep = True
            cfg.benchmark.enable_drift_sweep = True
            cfg.benchmark.enable_moon_cat = True
            cfg.benchmark.enable_gates = True
        else:
            cfg.benchmark.enable_reward_sweep = "rewards" in sections
            cfg.benchmark.enable_optimizer_sweep = "optimizers" in sections
            cfg.benchmark.enable_drift_sweep = "drift" in sections
            cfg.benchmark.enable_moon_cat = "moon_cat" in sections
            cfg.benchmark.enable_gates = "gates" in sections

    if sweep:
        cfg = apply_sweep_config(cfg)
        print(
            "[sweep] Full parameter sweep enabled — all rewards x optimizers x drifts"
        )

    # Apply --optimizers / --drifts overrides (after --enable and --sweep)
    if opt_str:
        cfg.benchmark.optimizers = [s.strip() for s in opt_str.split(",")]
        cfg.benchmark.enable_optimizer_sweep = True
    if drift_str:
        cfg.benchmark.drifts = [s.strip() for s in drift_str.split(",")]
        cfg.benchmark.enable_drift_sweep = True

    # Precision
    if cfg.cat_params.use_double:
        dq.set_precision("double")

    print_env_info(cfg)

    all_results = {}
    t_start = time.time()

    # --- Benchmark Sweep ---
    if (
        cfg.benchmark.enable_reward_sweep
        or cfg.benchmark.enable_optimizer_sweep
        or cfg.benchmark.enable_drift_sweep
    ):
        print("\n" + "=" * 50)
        n = (
            len(cfg.benchmark.rewards)
            * len(cfg.benchmark.optimizers)
            * len(cfg.benchmark.drifts)
        )
        print(f"  Benchmark Sweep: {n} combinations")
        print("=" * 50)

        from src.benchmark import run_benchmark

        all_results["benchmark"] = run_benchmark(cfg, verbose=True)

    # --- Moon Cat ---
    if cfg.benchmark.enable_moon_cat:
        print("\n" + "=" * 50)
        print("  Moon Cat Extension (5D optimization)")
        print("=" * 50)
        try:
            from src.moon_cat import run_moon_cat_comparison

            all_results["moon_cat"] = run_moon_cat_comparison(cfg)

            # Generate moon cat plots
            try:
                from src.plotting import (
                    plot_moon_cat_convergence,
                    plot_moon_cat_lifetimes,
                )

                plot_moon_cat_convergence(
                    all_results["moon_cat"],
                    save_path="figures/moon_cat_convergence.png",
                )
                plot_moon_cat_lifetimes(
                    all_results["moon_cat"],
                    save_path="figures/moon_cat_lifetime_comparison.png",
                )
                print("[plots] Moon cat comparison plots saved")
            except Exception as e:
                print(f"[warn] Moon cat plots failed: {e}")

            # Generate moon cat Wigner comparison
            try:
                from src.visualization import plot_moon_cat_wigner

                mc_results = all_results["moon_cat"]
                std_best = mc_results["standard"].param_history[-1]
                moon_best = mc_results["moon"].param_history[-1]
                plot_moon_cat_wigner(
                    std_best,
                    moon_best,
                    cfg.cat_params,
                    save_path="figures/moon_cat_wigner_comparison.png",
                )
                print("[plots] Moon cat Wigner comparison saved")
            except Exception as e:
                print(f"[warn] Moon cat Wigner plot failed: {e}")

        except ImportError as e:
            print(f"[skip] {e}")
        except Exception as e:
            print(f"[error] Moon cat failed: {e}")
            import traceback

            traceback.print_exc()

    # --- Single-Qubit Gates ---
    if cfg.benchmark.enable_gates:
        print("\n" + "=" * 50)
        print("  Single-Qubit Gates")
        print("=" * 50)
        try:
            from src.gates import run_gate_benchmark

            all_results["gate"] = run_gate_benchmark(cfg)
        except ImportError as e:
            print(f"[skip] {e}")
        except Exception as e:
            print(f"[error] Gate benchmark failed: {e}")
            import traceback

            traceback.print_exc()

    # --- Summary ---
    total_time = time.time() - t_start
    print_summary(all_results)

    # --- Save ---
    if save and all_results:
        save_results_json(all_results, profile, cfg)

    # --- Plots ---
    if plots and all_results:
        generate_plots(all_results, cfg)

    # --- Animations ---
    if plots and all_results:
        try:
            from src.visualization import generate_all_animations

            t_anim = time.time()
            generate_all_animations(all_results, cfg)
            print(f"[anim] Animation generation: {time.time() - t_anim:.1f}s")
        except Exception as e:
            print(f"[anim] Error generating animations: {e}")

    # --- Interactive display ---
    if interactive and plots and all_results:
        plt.show(block=True)

    print(
        f"\n[done] {datetime.now().isoformat()} — {total_time:.1f}s ({total_time / 60:.1f} min)"
    )


if __name__ == "__main__":
    main()
