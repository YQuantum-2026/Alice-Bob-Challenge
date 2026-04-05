"""Benchmark runner for cat qubit online optimization.

Provides:
  - build_optimizer(): factory for all optimizer types
  - run_single(): run one (reward, optimizer, drift) combination
  - run_benchmark(): sweep over all configured combinations
  - run_weight_sweep(): sweep over reward weight combinations with ground-truth evaluation
  - RunResult: dataclass storing complete optimization history

Usage:
  from src.config import get_config
  from src.benchmark import run_benchmark

  cfg = get_config("medium")
  cfg.benchmark.rewards = ["proxy", "photon", "parity"]
  cfg.benchmark.optimizers = ["cmaes", "hybrid"]
  cfg.benchmark.drifts = ["none", "amplitude_slow", "frequency"]
  results = run_benchmark(cfg)

References:
  Pack et al. "Benchmarking Optimization Algorithms for Automated Calibration
  of Quantum Devices." arXiv:2509.08555 (2025). — Methodology for optimizer comparison.
  Sivak et al. "RL Control of QEC." arXiv:2511.08493 (2025). — Drift tracking evaluation.
"""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass, field
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from src.cat_qubit import measure_lifetimes
from src.config import RunConfig, build_drift_model

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Complete history from a single optimization run.

    Attributes
    ----------
    reward_type : str
        Which reward function was used ("proxy", "photon", "fidelity", "parity").
    optimizer_type : str
        Which optimizer was used ("cmaes", "hybrid", "reinforce", "ppo", "bayesian").
    drift_type : str
        Which drift scenario was active ("none", "amplitude_slow", etc.).
    reward_history : list[float]
        Mean reward per epoch.
    reward_std_history : list[float]
        Reward std per epoch (for population-based optimizers).
    param_history : list[np.ndarray]
        Optimizer mean/best parameters per epoch.
    validation_history : list[dict]
        Periodic full T_X/T_Z/bias measurements. Each dict has keys:
        "epoch", "Tz", "Tx", "bias", "alpha".
    drift_offset_history : list[dict]
        Drift offsets applied at each epoch.
    wall_time : float
        Total wall-clock time [seconds].
    n_epochs : int
        Number of epochs completed.
    config_name : str
        Profile name used.
    seed : int
        Random seed used for this run.
    """

    reward_type: str = ""
    optimizer_type: str = ""
    drift_type: str = ""
    reward_history: list[float] = field(default_factory=list)
    reward_std_history: list[float] = field(default_factory=list)
    param_history: list[np.ndarray] = field(default_factory=list)
    validation_history: list[dict] = field(default_factory=list)
    drift_offset_history: list[dict] = field(default_factory=list)
    wall_time: float = 0.0
    n_epochs: int = 0
    config_name: str = ""
    seed: int = 0

    @property
    def label(self) -> str:
        return f"{self.optimizer_type}/{self.reward_type}/{self.drift_type}"


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def build_optimizer(opt_type: str, reward_fn, cfg: RunConfig, n_params: int = 4):
    """Build an optimizer by type name.

    Parameters
    ----------
    opt_type : str
        One of "cmaes", "hybrid", "reinforce", "ppo", "bayesian".
    reward_fn : callable
        JIT-compiled reward: x (n_params,) → scalar. Needed for hybrid optimizer.
    cfg : RunConfig
        Full config for hyperparameters.
    n_params : int
        Dimensionality of the parameter space. Default: 4 (standard cat).
        Use 5 for moon cat or gate extensions.

    Returns
    -------
    OnlineOptimizer
    """
    oc = cfg.optimizer

    if opt_type == "cmaes":
        from src.optimizers.cmaes_opt import CMAESOptimizer

        return CMAESOptimizer(
            mean0=np.array(oc.init_params),
            population_size=oc.population_size,
            sigma0=oc.sigma0,
            sigma_floor=oc.sigma_floor,
            seed=oc.seed,
        )

    elif opt_type == "hybrid":
        from src.optimizers.hybrid_opt import HybridOptimizer

        return HybridOptimizer(
            reward_fn=reward_fn,
            cma_epochs=oc.hybrid_cma_epochs,
            grad_steps=oc.hybrid_grad_steps,
            learning_rate=oc.learning_rate,
            population_size=oc.population_size,
            sigma0=oc.sigma0,
            init_params=oc.init_params,
            seed=oc.seed,
        )

    elif opt_type == "reinforce":
        from src.optimizers.reinforce_opt import REINFORCEOptimizer

        return REINFORCEOptimizer(
            n_params=n_params,
            population_size=oc.population_size,
            lr_mean=oc.policy_lr_mean,
            lr_sigma=oc.policy_lr_sigma,
            beta_entropy=oc.policy_beta_entropy,
            baseline_decay=oc.policy_baseline_decay,
            seed=oc.seed,
            init_params=oc.init_params,
        )

    elif opt_type == "ppo":
        from src.optimizers.ppo_opt import PPOOptimizer

        return PPOOptimizer(
            n_params=n_params,
            population_size=oc.population_size,
            lr_mean=oc.policy_lr_mean,
            lr_sigma=oc.policy_lr_sigma,
            beta_entropy=oc.policy_beta_entropy,
            baseline_decay=oc.policy_baseline_decay,
            clip_eps=oc.ppo_clip_eps,
            n_epochs=oc.ppo_n_epochs,
            seed=oc.seed,
            init_params=oc.init_params,
        )

    elif opt_type == "bayesian":
        from src.optimizers.bayesian_opt import BayesianOptimizer

        return BayesianOptimizer(
            n_initial=oc.bayesian_n_initial,
            acq_func=oc.bayesian_acq_func,
            seed=oc.seed,
        )

    else:
        raise ValueError(
            f"Unknown optimizer '{opt_type}'. "
            "Options: cmaes, hybrid, reinforce, ppo, bayesian"
        )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_single(
    reward_type: str,
    optimizer_type: str,
    drift_type: str,
    cfg: RunConfig,
    verbose: bool = True,
) -> RunResult:
    """Run one (reward, optimizer, drift) combination.

    Parameters
    ----------
    reward_type : str
        Reward function name.
    optimizer_type : str
        Optimizer name.
    drift_type : str
        Drift scenario name.
    cfg : RunConfig
        Full configuration.
    verbose : bool
        Print progress.

    Returns
    -------
    RunResult
    """
    from src.reward import build_reward

    params = cfg.cat_params
    n_epochs = cfg.optimizer.n_epochs

    # Build reward
    reward_fn, batched_reward_fn = build_reward(reward_type, params, cfg.reward)

    # Build drift model
    drift_model = build_drift_model(drift_type, cfg.drift)

    # Build optimizer
    optimizer = build_optimizer(optimizer_type, reward_fn, cfg)

    if drift_type != "none":
        n_drift = 6
        from src.reward import build_drift_aware_reward

        drift_reward_fn, drift_batched_fn = build_drift_aware_reward(
            reward_type, params, cfg.reward, n_drift_slots=n_drift
        )

        # Compile
        if verbose:
            print(
                f"[{optimizer_type}/{reward_type}/{drift_type}] Compiling drift reward..."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dummy = jnp.zeros(4 + n_drift)
            _ = drift_reward_fn(dummy)
    else:
        n_drift = 0
        drift_reward_fn = reward_fn
        drift_batched_fn = batched_reward_fn

    if verbose:
        print(
            f"[{optimizer_type}/{reward_type}/{drift_type}] Starting {n_epochs} epochs"
        )

    # Logging
    result = RunResult(
        reward_type=reward_type,
        optimizer_type=optimizer_type,
        drift_type=drift_type,
        config_name=cfg.name,
        seed=cfg.optimizer.seed,
    )

    t_start = time.time()

    epoch_iter = tqdm(
        range(n_epochs),
        desc=f"  {optimizer_type}/{reward_type}/{drift_type}",
        disable=not verbose,
        leave=True,
        ncols=100,
    )
    for epoch in epoch_iter:
        # --- Ask optimizer for candidates ---
        # NOTE: batch size varies by optimizer (CMA-ES ≈ 24, Bayesian ≈ 1),
        # so total evaluations differ across the same epoch budget.
        xs_control = optimizer.ask()  # shape (N, 4) or (1, 4)
        n_candidates = xs_control.shape[0]

        if n_drift > 0:
            # --- Get drift state (scale amplitude drift with actual g2) ---
            try:
                best = optimizer.get_best()
            except RuntimeError:
                # Bayesian optimizer raises before first tell(); use current ask
                best = xs_control[0]
            offsets = drift_model.get_control_offsets(
                epoch,
                current_params={"g2_re": float(best[0]), "g2_im": float(best[1])},
            )
            h_terms = drift_model.get_hamiltonian_terms(epoch)
            drift_vec = np.array(
                [
                    offsets["g2_re_offset"],
                    offsets["g2_im_offset"],
                    offsets["eps_d_re_offset"],
                    offsets["eps_d_im_offset"],
                    h_terms["detuning"],
                    h_terms["kerr"],
                ]
            )

            # --- Append drift to candidates ---
            drift_broadcast = jnp.broadcast_to(
                jnp.array(drift_vec), (n_candidates, n_drift)
            )
            xs_full = jnp.concatenate([xs_control, drift_broadcast], axis=1)
        else:
            offsets = {
                "g2_re_offset": 0.0,
                "g2_im_offset": 0.0,
                "eps_d_re_offset": 0.0,
                "eps_d_im_offset": 0.0,
            }
            h_terms = {"detuning": 0.0, "kerr": 0.0}
            xs_full = xs_control

        # --- Evaluate rewards ---
        if n_candidates > 1:
            rewards = drift_batched_fn(xs_full)
        else:
            rewards = jnp.array([drift_reward_fn(xs_full[0])])

        # --- Apply SNR noise ---
        snr_noise_std = drift_model.get_snr_noise(epoch)
        if snr_noise_std > 0:
            key = jax.random.PRNGKey(cfg.optimizer.seed * 10_000 + epoch)
            noise = jax.random.normal(key, shape=rewards.shape) * snr_noise_std
            rewards = rewards + noise

        # --- Tell optimizer (control params only) ---
        optimizer.tell(xs_control, rewards)

        # --- Log ---
        result.reward_history.append(float(jnp.mean(rewards)))
        result.reward_std_history.append(float(jnp.std(rewards)))
        result.param_history.append(np.array(optimizer.get_best()))
        result.drift_offset_history.append(
            {
                "g2_re": float(offsets["g2_re_offset"]),
                "g2_im": float(offsets["g2_im_offset"]),
                "eps_d_re": float(offsets["eps_d_re_offset"]),
                "eps_d_im": float(offsets["eps_d_im_offset"]),
                "detuning": float(h_terms["detuning"]),
                "kerr": float(h_terms["kerr"]),
                "snr_noise": float(snr_noise_std),
            }
        )

        # --- Update tqdm postfix ---
        epoch_iter.set_postfix_str(f"R={float(jnp.mean(rewards)):.3f}")

        # --- Periodic full validation ---
        if epoch % cfg.reward.full_eval_interval == 0 and epoch > 0:
            best = optimizer.get_best()
            try:
                _alpha_free = reward_type in ("vacuum", "parity", "fidelity")
                if _alpha_free:
                    # Use the reward function itself for validation —
                    # matches the measurement protocol the optimizer is using.
                    x_best = jnp.array(
                        [
                            float(best[0]),
                            float(best[1]),
                            float(best[2]),
                            float(best[3]),
                        ]
                    )
                    val_reward = float(reward_fn(x_best))
                    result.validation_history.append(
                        {
                            "epoch": epoch,
                            "reward": val_reward,
                            "method": reward_type,
                        }
                    )
                    if verbose:
                        tqdm.write(
                            f"  Epoch {epoch:4d} | R({reward_type})={val_reward:.3f}"
                        )
                else:
                    # Proxy-based validation with heuristic alpha
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        lt = measure_lifetimes(
                            float(best[0]),
                            float(best[1]),
                            float(best[2]),
                            float(best[3]),
                            tfinal_z=cfg.reward.tfinal_z,
                            tfinal_x=cfg.reward.tfinal_x,
                            params=params,
                        )
                    from src.cat_qubit import compute_alpha

                    alpha = compute_alpha(
                        float(best[0]),
                        float(best[1]),
                        float(best[2]),
                        float(best[3]),
                        params,
                    )
                    result.validation_history.append(
                        {
                            "epoch": epoch,
                            "Tz": lt["Tz"],
                            "Tx": lt["Tx"],
                            "bias": lt["bias"],
                            "alpha": float(alpha),
                            "method": "proxy",
                        }
                    )
                    if verbose:
                        tqdm.write(
                            f"  Epoch {epoch:4d} | T_Z={lt['Tz']:.1f} | "
                            f"T_X={lt['Tx']:.4f} | \u03b7={lt['bias']:.0f}"
                        )
            except Exception as e:
                if verbose:
                    tqdm.write(f"  Epoch {epoch:4d} | Validation failed: {e}")

    result.wall_time = time.time() - t_start
    result.n_epochs = n_epochs

    if verbose:
        print(f"[{result.label}] Done in {result.wall_time:.1f}s")

    return result


# ---------------------------------------------------------------------------
# TLS drift run (extended Hilbert space)
# ---------------------------------------------------------------------------


def run_single_tls(
    reward_type: str,
    optimizer_type: str,
    cfg: RunConfig,
    verbose: bool = True,
) -> RunResult:
    """Run one optimization with TLS drift in the extended Hilbert space.

    When TLS is active, the Hilbert space doubles: storage(na) ⊗ buffer(nb) ⊗ TLS(2).
    All operators, Hamiltonian, and jump operators are rebuilt in the extended space.
    Before TLS onset_epoch, g_tls=0 so the system behaves as standard cat.

    Only supports proxy-type rewards (JIT-compiled) in the TLS space.
    Non-JIT rewards (vacuum, parity, fidelity, spectral) are not supported
    due to the alpha estimation complexity in the extended space.

    Parameters
    ----------
    reward_type : str
        Reward function name (only proxy-based rewards supported).
    optimizer_type : str
        Optimizer name.
    cfg : RunConfig
        Full configuration.
    verbose : bool
        Print progress.

    Returns
    -------
    RunResult
    """
    from src.cat_qubit import (
        build_hamiltonian,
        build_tls_hamiltonian_term,
        build_tls_jump_ops,
        build_tls_operators,
        compute_alpha,
    )
    from src.reward._helpers import _compute_lifetime_score

    params = cfg.cat_params
    n_epochs = cfg.optimizer.n_epochs
    drift_model = build_drift_model("tls", cfg.drift)

    # Build operators in the TLS-extended space
    a_ext, b_ext, sigma_z, sigma_m = build_tls_operators(params)

    # Get TLS parameters (use defaults from drift model)
    _, omega_tls_default, gamma_tls_default = drift_model.get_tls_coupling(epoch=9999)

    # Build jump ops with TLS decay
    jump_ops_ext = build_tls_jump_ops(a_ext, b_ext, sigma_m, params, gamma_tls_default)

    # Build a proxy reward function in the extended space.
    # This is a simplified version of _build_proxy_loss_fn that works with
    # the TLS-extended operators.
    import dynamiqs as dq

    t_probe_z = cfg.reward.t_probe_z
    t_probe_x = cfg.reward.t_probe_x
    target_bias = cfg.reward.target_bias
    w_lifetime = cfg.reward.w_lifetime
    w_bias = cfg.reward.w_bias

    # Alpha estimation (heuristic — same as proxy reward)
    alpha_fn = compute_alpha

    def tls_reward_fn(x):
        """Proxy reward in TLS-extended space."""
        g2_re, g2_im, eps_d_re, eps_d_im = x[0], x[1], x[2], x[3]

        # Get current TLS coupling (may include drift offsets in x[4:])
        if x.shape[0] > 4:
            g_tls_eff = float(x[4])
        else:
            g_tls_eff = 0.0

        # Build Hamiltonian in extended space
        H = build_hamiltonian(a_ext, b_ext, g2_re, g2_im, eps_d_re, eps_d_im)
        if g_tls_eff > 0:
            H = H + build_tls_hamiltonian_term(
                a_ext, sigma_z, sigma_m, g_tls_eff, omega_tls_default
            )

        # Estimate alpha (heuristic)
        alpha = alpha_fn(g2_re, g2_im, eps_d_re, eps_d_im, params)

        if alpha < 0.1:
            return jnp.array(-100.0)

        # Build logical ops in extended space
        # Parity on storage ⊗ I_buffer ⊗ I_TLS
        parity_ext = (1j * jnp.pi * a_ext.dag() @ a_ext).expm()

        # Z operator: coherent state projector difference ⊗ I_buffer ⊗ I_TLS
        plus_alpha = dq.coherent(params.na, alpha)
        minus_alpha = dq.coherent(params.na, -alpha)
        sz_local = plus_alpha @ plus_alpha.dag() - minus_alpha @ minus_alpha.dag()
        sz_ext = dq.tensor(sz_local, dq.eye(params.nb), dq.eye(2))

        # Initial states in extended space (TLS starts in ground state |0⟩)
        cat_even = dq.unit(
            dq.coherent(params.na, alpha) + dq.coherent(params.na, -alpha)
        )
        cat_odd = dq.unit(
            dq.coherent(params.na, alpha) - dq.coherent(params.na, -alpha)
        )
        tls_ground = dq.basis(2, 0)

        psi_z = dq.tensor(cat_even, dq.basis(params.nb, 0), tls_ground)
        psi_x = dq.tensor(cat_odd, dq.basis(params.nb, 0), tls_ground)

        # Measure Z decay
        tsave_z = jnp.array([0.0, float(t_probe_z)])
        result_z = dq.mesolve(H, jump_ops_ext, psi_z, tsave_z, exp_ops=[sz_ext])
        exp_z = jnp.abs(result_z.expects[0, -1])
        T_Z_est = -t_probe_z / jnp.log(jnp.maximum(exp_z, 1e-10))

        # Measure X decay
        tsave_x = jnp.array([0.0, float(t_probe_x)])
        result_x = dq.mesolve(H, jump_ops_ext, psi_x, tsave_x, exp_ops=[parity_ext])
        exp_x = jnp.abs(result_x.expects[0, -1])
        T_X_est = -t_probe_x / jnp.log(jnp.maximum(exp_x, 1e-10))

        return _compute_lifetime_score(
            T_Z_est, T_X_est, target_bias, w_lifetime, w_bias
        )

    # Build optimizer
    optimizer = build_optimizer(optimizer_type, tls_reward_fn, cfg)

    if verbose:
        print(
            f"[{optimizer_type}/{reward_type}/tls] Starting {n_epochs} epochs "
            f"(dim={params.na}×{params.nb}×2={params.na * params.nb * 2})"
        )

    result = RunResult(
        reward_type=reward_type,
        optimizer_type=optimizer_type,
        drift_type="tls",
        config_name=cfg.name,
        seed=cfg.optimizer.seed,
    )

    t_start = time.time()

    epoch_iter = tqdm(
        range(n_epochs),
        desc=f"  {optimizer_type}/{reward_type}/tls",
        disable=not verbose,
        leave=True,
        ncols=100,
    )
    for epoch in epoch_iter:
        xs_control = optimizer.ask()
        n_candidates = xs_control.shape[0]

        # Get TLS coupling at this epoch (0 before onset, g_tls after)
        g_tls_now, _, _ = drift_model.get_tls_coupling(epoch)

        # Append TLS coupling as extra slot
        tls_broadcast = jnp.full((n_candidates, 1), g_tls_now)
        xs_full = jnp.concatenate([xs_control, tls_broadcast], axis=1)

        # Evaluate rewards
        rewards = jnp.array(
            [float(tls_reward_fn(xs_full[i])) for i in range(n_candidates)]
        )

        # Tell optimizer (control params only)
        optimizer.tell(xs_control, rewards)

        # Log
        result.reward_history.append(float(jnp.mean(rewards)))
        result.reward_std_history.append(float(jnp.std(rewards)))
        result.param_history.append(np.array(optimizer.get_best()))
        result.drift_offset_history.append(
            {
                "g2_re": 0.0,
                "g2_im": 0.0,
                "eps_d_re": 0.0,
                "eps_d_im": 0.0,
                "detuning": 0.0,
                "kerr": 0.0,
                "snr_noise": 0.0,
                "tls_g": float(g_tls_now),
            }
        )

        epoch_iter.set_postfix_str(
            f"R={float(jnp.mean(rewards)):.3f} tls={'ON' if g_tls_now > 0 else 'off'}"
        )

    result.wall_time = time.time() - t_start
    result.n_epochs = n_epochs

    if verbose:
        print(f"[{result.label}] Done in {result.wall_time:.1f}s")

    return result


# ---------------------------------------------------------------------------
# Full benchmark sweep
# ---------------------------------------------------------------------------


def run_benchmark(cfg: RunConfig, verbose: bool = True) -> list[RunResult]:
    """Run all (reward × optimizer × drift) combinations from config.

    Iterates over the Cartesian product of:
      cfg.benchmark.rewards × cfg.benchmark.optimizers × cfg.benchmark.drifts

    Parameters
    ----------
    cfg : RunConfig
        Full configuration including BenchmarkConfig.
    verbose : bool
        Print progress.

    Returns
    -------
    list[RunResult]
        One result per combination.
    """
    combos = list(
        product(
            cfg.benchmark.rewards,
            cfg.benchmark.optimizers,
            cfg.benchmark.drifts,
        )
    )
    n_total = len(combos) * cfg.benchmark.n_runs_per_combo

    if verbose:
        print(
            f"=== Benchmark: {len(combos)} combinations × "
            f"{cfg.benchmark.n_runs_per_combo} runs = {n_total} total ==="
        )
        print(f"  Rewards: {cfg.benchmark.rewards}")
        print(f"  Optimizers: {cfg.benchmark.optimizers}")
        print(f"  Drifts: {cfg.benchmark.drifts}")
        print(f"  Profile: {cfg.name}")
        print()

    results = []

    combo_bar = tqdm(
        enumerate(combos),
        total=len(combos),
        desc="Benchmark",
        disable=not verbose,
        ncols=100,
    )
    for _i, (reward_type, opt_type, drift_type) in combo_bar:
        combo_bar.set_description(f"[{opt_type}/{reward_type}/{drift_type}]")

        # TLS drift requires doubled Hilbert space (storage ⊗ buffer ⊗ TLS)
        if drift_type == "tls":
            if not cfg.benchmark.enable_tls:
                if verbose:
                    tqdm.write("  Skipping TLS (enable_tls=False)")
                continue
            for run_idx in range(cfg.benchmark.n_runs_per_combo):
                run_cfg = copy.deepcopy(cfg)
                run_cfg.optimizer.seed = cfg.optimizer.seed + run_idx * 1000
                result = run_single_tls(reward_type, opt_type, run_cfg, verbose)
                results.append(result)
            continue

        for run_idx in range(cfg.benchmark.n_runs_per_combo):
            # Vary seed per run for statistics
            run_cfg = copy.deepcopy(cfg)
            run_cfg.optimizer.seed = cfg.optimizer.seed + run_idx * 1000

            result = run_single(reward_type, opt_type, drift_type, run_cfg, verbose)
            results.append(result)

    if verbose:
        print(f"\n=== Benchmark complete: {len(results)} runs ===")
        total_time = sum(r.wall_time for r in results)
        print(f"Total wall time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    if verbose and results:
        print(f"\n  {'Combo':<45s} {'Time':>8s}")
        print("  " + "-" * 55)
        for r in sorted(results, key=lambda x: x.wall_time, reverse=True):
            print(f"  {r.label:<45s} {r.wall_time:>7.1f}s")
        fastest = min(results, key=lambda x: x.wall_time)
        slowest = max(results, key=lambda x: x.wall_time)
        avg_time = sum(r.wall_time for r in results) / len(results)
        print(f"\n  Fastest: {fastest.label} ({fastest.wall_time:.1f}s)")
        print(f"  Slowest: {slowest.label} ({slowest.wall_time:.1f}s)")
        print(f"  Average: {avg_time:.1f}s per combo")

    return results


def run_weight_sweep(
    weight_grid: dict[str, list[float]],
    cfg: RunConfig,
    reward_type: str = "enhanced_proxy",
    optimizer_type: str = "cmaes",
    drift_type: str = "none",
    verbose: bool = True,
) -> list[tuple[dict, RunResult, dict]]:
    """Sweep over reward weight combinations with ground-truth evaluation.

    For each combination of weights in the grid, runs a full optimization
    and evaluates the final parameters with measure_lifetimes().

    Parameters
    ----------
    weight_grid : dict
        Maps weight name to list of values to sweep.
        Example: {"w_bias": [0.1, 0.5, 1.0], "w_buffer": [0.0, 0.1, 0.5]}
    cfg : RunConfig
        Base configuration (deep-copied and modified per sweep point).
    reward_type : str
        Reward function to use (default: "enhanced_proxy").
    optimizer_type : str
        Optimizer to use (default: "cmaes" for consistency).
    drift_type : str
        Drift scenario (default: "none").
    verbose : bool
        Print progress.

    Returns
    -------
    list of (weights_dict, RunResult, ground_truth_dict) tuples
        weights_dict: the weight values used for this run
        RunResult: full optimization history
        ground_truth_dict: {"Tz": float, "Tx": float, "bias": float} from measure_lifetimes
    """
    weight_names = list(weight_grid.keys())
    weight_values = [weight_grid[k] for k in weight_names]
    combos = list(product(*weight_values))
    n_total = len(combos)

    if verbose:
        print(f"=== Weight Sweep: {n_total} combinations ===")
        for name, vals in weight_grid.items():
            print(f"  {name}: {vals}")
        print()

    sweep_results: list[tuple[dict, RunResult, dict]] = []

    for idx, combo in enumerate(combos):
        weights_dict = dict(zip(weight_names, combo))

        if verbose:
            print(
                f"[{idx + 1}/{n_total}] Weights: "
                + ", ".join(f"{k}={v:.3f}" for k, v in weights_dict.items())
            )

        # Deep-copy config and set weights on reward sub-config
        cfg_copy = copy.deepcopy(cfg)
        for w_name, w_val in weights_dict.items():
            setattr(cfg_copy.reward, w_name, w_val)

        # Run optimization
        result = run_single(
            reward_type, optimizer_type, drift_type, cfg_copy, verbose=verbose
        )

        # Evaluate final params with ground-truth measure_lifetimes
        best = result.param_history[-1] if result.param_history else np.zeros(4)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gt = measure_lifetimes(
                    float(best[0]),
                    float(best[1]),
                    float(best[2]),
                    float(best[3]),
                    tfinal_z=cfg_copy.reward.tfinal_z,
                    tfinal_x=cfg_copy.reward.tfinal_x,
                    params=cfg_copy.cat_params,
                )
        except Exception as e:
            if verbose:
                print(f"  [warn] Ground-truth evaluation failed: {e}")
            gt = {"Tz": float("nan"), "Tx": float("nan"), "bias": float("nan")}

        sweep_results.append((weights_dict, result, gt))

    # Print summary table
    if verbose and sweep_results:
        print(f"\n=== Weight Sweep Summary ({n_total} runs) ===")
        header_weights = "  ".join(f"{k:>10s}" for k in weight_names)
        print(
            f"  {header_weights}  {'T_Z':>8s}  {'T_X':>8s}  "
            f"{'Bias':>8s}  {'Final R':>8s}"
        )
        print("  " + "-" * (12 * len(weight_names) + 40))
        for weights_dict, result, gt in sweep_results:
            w_str = "  ".join(f"{weights_dict[k]:>10.3f}" for k in weight_names)
            final_r = (
                result.reward_history[-1] if result.reward_history else float("nan")
            )
            print(
                f"  {w_str}  {gt['Tz']:>8.1f}  {gt['Tx']:>8.4f}  "
                f"{gt['bias']:>8.0f}  {final_r:>8.3f}"
            )

        # Highlight best by bias
        valid_results = [
            (w, r, g) for w, r, g in sweep_results if np.isfinite(g["bias"])
        ]
        if valid_results:
            best_by_bias = max(valid_results, key=lambda x: x[2]["bias"])
            print(
                "\n  Best by bias: "
                + ", ".join(f"{k}={v:.3f}" for k, v in best_by_bias[0].items())
                + f" -> eta={best_by_bias[2]['bias']:.0f}"
            )

    return sweep_results
