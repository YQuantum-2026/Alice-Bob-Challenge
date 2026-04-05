"""Scalable configuration for cat qubit optimization.

Provides:
  - Preset profiles: LOCAL (fast dev), MEDIUM (local production), HPC (Palmetto full-scale)
  - BenchmarkConfig: controls which reward × optimizer × drift combos to sweep
  - All simulation parameters in one place for easy scaling
  - Factory functions: build_drift_model() for named drift scenarios

Usage:
  from src.config import get_config

  cfg = get_config("medium")
  cfg.benchmark.rewards = ["proxy", "photon", "fidelity", "parity"]
  cfg.benchmark.optimizers = ["cmaes", "hybrid", "reinforce", "ppo", "bayesian"]
  cfg.benchmark.drifts = ["none", "amplitude_slow", "frequency", "snr"]
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from src.cat_qubit import CatQubitParams

# ---------------------------------------------------------------------------
# Optimizer config
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    """Hyperparameters for all optimizers."""

    # --- CMA-ES ---
    population_size: int = 24
    n_epochs: int = 200
    sigma0: float = 0.5
    sigma_floor: float | None = 0.05

    # --- Hybrid CMA-ES + Gradient ---
    learning_rate: float = 0.01
    hybrid_cma_epochs: int = 30  # CMA-ES phase length
    hybrid_grad_steps: int = 10  # gradient refinement steps per phase

    # --- REINFORCE / PPO policy gradient (shared hyperparams) ---
    # Ref: Sivak et al. (2025), arXiv:2511.08493, Sec. II (REINFORCE)
    # Ref: Schulman et al. (2017), arXiv:1707.06347 (PPO-Clip)
    policy_lr_mean: float = 0.05  # learning rate for policy mean
    policy_lr_sigma: float = 0.01  # learning rate for policy std
    policy_beta_entropy: float = 0.02  # entropy regularization weight
    policy_baseline_decay: float = 0.9  # exponential moving average for baseline
    ppo_clip_eps: float = 0.2  # PPO clipping epsilon (PPO-Clip only)
    ppo_n_epochs: int = 3  # update epochs per batch (PPO-Clip only)

    # --- Bayesian optimization ---
    bayesian_n_initial: int = 10  # random exploration before GP kicks in
    bayesian_acq_func: str = "gp_hedge"  # acquisition function

    # --- Shared ---
    seed: int = 420
    init_params: list[float] = field(default_factory=lambda: [1.0, 0.0, 4.0, 0.0])
    """Initial parameter mean [g2_re, g2_im, eps_d_re, eps_d_im]."""


# ---------------------------------------------------------------------------
# Reward config
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """Configuration for all reward functions."""

    # --- Proxy reward ---
    t_probe_z: float = 50.0  # probe time for T_Z [us]
    t_probe_x: float = 0.3  # probe time for T_X [us]
    target_bias: float = 100.0  # target η = T_Z / T_X
    w_lifetime: float = 1.0  # weight on lifetime maximization
    w_bias: float = 0.5  # weight on bias targeting

    # --- Full measurement (validation) ---
    tfinal_z: float = 200.0  # T_Z simulation window [us]
    tfinal_x: float = 1.0  # T_X simulation window [us]
    npoints: int = (
        100  # time points for full measurement (validation only, via measure_lifetimes)
    )
    full_eval_interval: int = 20  # run full measurement every N epochs

    # --- Photon number proxy ---
    # Ref: measures ⟨a†a⟩ as proxy for |α|²
    n_target: float = 4.0  # target photon number ≈ |α_target|²
    t_steady: float = 5.0  # time to reach steady state [us]

    # --- Fidelity-based ---
    # Uses same t_steady as photon reward

    # --- Parity decay ---
    # Uses same t_probe_x and t_probe_z as proxy reward
    # Parity P = exp(iπ a†a) is α-independent

    # --- Log-derivative lifetime estimation ---
    use_log_derivative: bool = False  # use d(log⟨O⟩)/dt slope for T estimation
    n_log_deriv_points: int = 5  # number of time points for regression

    # --- Vacuum-based (alpha-free) reward ---
    # Physically correct: start from vacuum, let system find its own cat states.
    # No compute_alpha needed. Matches Alice & Bob experimental protocol.
    # Ref: Réglade et al. Nature 629, 778-783 (2024). arXiv:2307.06617.
    t_settle: float = 15.0  # settling time for vacuum → cat/well [μs]
    t_measure_z: float = 200.0  # T_Z measurement window [μs]
    t_measure_x: float = 1.0  # T_X measurement window [μs]

    # --- Enhanced proxy (physics-augmented) ---
    w_buffer: float = 0.0  # weight on buffer occupation penalty
    w_confinement: float = 0.0  # weight on code space leakage penalty
    w_margin: float = 0.0  # weight on alpha stability margin penalty
    margin_threshold: float = 1.0  # margin below this is penalized


# ---------------------------------------------------------------------------
# Drift config
# ---------------------------------------------------------------------------


@dataclass
class DriftConfig:
    """Configuration for all drift models."""

    # --- Amplitude drift ---
    amplitude_drift_A: float = 0.3
    amplitude_drift_freq: float = 0.005

    # --- Frequency drift ---
    frequency_drift_A: float = 0.5
    frequency_drift_freq: float = 0.005

    # --- Kerr drift ---
    kerr_drift_A: float = 0.01
    kerr_drift_freq: float = 0.003

    # --- TLS coupling ---
    # Ref: Challenge notebook, "Drift and Noise Modeling" section
    tls_g: float = 0.1  # TLS-storage coupling [MHz]
    tls_omega: float = 0.0  # TLS detuning (0 = resonant)
    tls_gamma: float = 0.5  # TLS decay rate [MHz]
    tls_onset_epoch: int = 50  # epoch when TLS appears

    # --- White noise ---
    white_noise_sigma_g2: float = 0.05  # Gaussian noise std on g2 [MHz]
    white_noise_sigma_epsd: float = 0.0  # Gaussian noise std on eps_d [MHz]

    # --- SNR degradation ---
    snr_base_noise: float = 0.01
    snr_growth_rate: float = 0.001


# ---------------------------------------------------------------------------
# Moon cat config
# ---------------------------------------------------------------------------


@dataclass
class MoonCatConfig:
    """Configuration for moon cat extension.

    Ref: Rousseau et al. (2025), arXiv:2502.07892
    Adds squeezing term g₂·λ·a†a·b to the Hamiltonian.
    """

    lambda_min: float = 0.0  # lower bound for λ
    lambda_max: float = 1.0  # upper bound for λ
    lambda_init: float = 0.3  # initial λ for optimizer
    compare_with_standard: bool = True  # run standard cat in parallel for comparison


# ---------------------------------------------------------------------------
# Gate config
# ---------------------------------------------------------------------------


@dataclass
class GateConfig:
    """Configuration for single-qubit gate extension.

    Ref: iQuHack-2025 — Zeno gate H = ε_Z(a† + a)
    """

    epsilon_z_min: float = 0.01  # lower bound for gate strength
    epsilon_z_max: float = 1.0  # upper bound for gate strength
    epsilon_z_init: float = 0.2  # initial gate strength [MHz]
    gate_duration: float = 0.5  # gate time [μs]
    w_gate_fidelity: float = 1.0  # weight on gate fidelity in reward
    w_stabilization: float = 1.0  # weight on stabilization quality in reward


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Controls which (reward × optimizer × drift) combinations to sweep.

    Set these lists to control what gets benchmarked. The benchmark runner
    iterates over the Cartesian product of rewards × optimizers × drifts.
    """

    enable_reward_sweep: bool = True
    """Run reward function comparison (proxy, photon, fidelity, parity)."""

    enable_optimizer_sweep: bool = True
    """Run optimizer comparison (CMA-ES, hybrid, PPO, Bayesian)."""

    enable_drift_sweep: bool = True
    """Run drift tracking scenarios."""

    enable_moon_cat: bool = False
    """Run moon cat extension (5D with squeezing parameter λ)."""

    enable_gates: bool = False
    """Run single-qubit gate extension (Zeno gate)."""

    rewards: list[str] = field(default_factory=lambda: ["proxy"])
    """Which reward functions to benchmark.
    Options: "proxy", "photon", "fidelity", "parity", "multipoint", "spectral",
    "enhanced_proxy", "vacuum"
    ("full" is validation-only, not used as optimization target)"""

    optimizers: list[str] = field(default_factory=lambda: ["cmaes"])
    """Which optimizers to benchmark.
    Options: "cmaes", "hybrid", "reinforce", "ppo", "bayesian" """

    drifts: list[str] = field(default_factory=lambda: ["none"])
    """Which drift scenarios to benchmark.
    Options: "none", "amplitude_slow", "amplitude_fast", "frequency",
             "kerr", "step", "snr", "multi", "white_noise", "tls" """

    n_runs_per_combo: int = 1
    """Repeated runs per combination for statistics."""

    enable_tls: bool = False
    """Enable TLS drift (experimental — doubles Hilbert space to na*nb*2)."""


# ---------------------------------------------------------------------------
# Top-level run config
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Complete configuration for optimization runs.

    Attributes
    ----------
    name : str
        Profile name ("local", "medium", "hpc").
    cat_params : CatQubitParams
        Hilbert space dimensions and hardware parameters.
    optimizer : OptimizerConfig
        Optimizer hyperparameters.
    reward : RewardConfig
        Reward function settings.
    drift : DriftConfig
        Drift model settings.
    benchmark : BenchmarkConfig
        Which combinations to sweep.
    moon_cat : MoonCatConfig
        Moon cat extension settings.
    gate : GateConfig
        Single-qubit gate settings.
    """

    name: str = "local"
    cat_params: CatQubitParams = field(default_factory=CatQubitParams)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    moon_cat: MoonCatConfig = field(default_factory=MoonCatConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    default_reward: str = "vacuum"
    """Default reward function for single runs. Vacuum is the alpha-free primary."""
    default_optimizer: str = "cmaes"
    """Default optimizer for single runs."""
    default_drift: str = "none"
    """Default drift scenario for single runs."""

    def summary(self) -> str:
        """Human-readable summary of the configuration."""
        p = self.cat_params
        o = self.optimizer
        r = self.reward
        b = self.benchmark
        enabled = []
        if b.enable_reward_sweep:
            enabled.append("rewards")
        if b.enable_optimizer_sweep:
            enabled.append("optimizers")
        if b.enable_drift_sweep:
            enabled.append("drift")
        if b.enable_moon_cat:
            enabled.append("moon_cat")
        if b.enable_gates:
            enabled.append("gates")
        return (
            f"=== RunConfig: {self.name} ===\n"
            f"  Hilbert space: na={p.na}, nb={p.nb} (dim={p.na * p.nb})\n"
            f"  Hardware: kappa_b={p.kappa_b}, kappa_a={p.kappa_a}\n"
            f"  Enabled: {', '.join(enabled)}\n"
            f"  CMA-ES: pop={o.population_size}, epochs={o.n_epochs}, sigma0={o.sigma0}\n"
            f"  Hybrid: lr={o.learning_rate}\n"
            f"  Reward: t_z={r.t_probe_z}, t_x={r.t_probe_x}, target_bias={r.target_bias}\n"
            f"  Full eval every {r.full_eval_interval} epochs\n"
            f"  Benchmark: {len(b.rewards)} rewards × {len(b.optimizers)} optimizers × {len(b.drifts)} drifts\n"
            f"    Rewards: {b.rewards}\n"
            f"    Optimizers: {b.optimizers}\n"
            f"    Drifts: {b.drifts}\n"
        )


# ---------------------------------------------------------------------------
# Drift model factory
# ---------------------------------------------------------------------------


def build_drift_model(drift_name: str, drift_cfg: DriftConfig):
    """Build a DriftModel from a named scenario.

    Parameters
    ----------
    drift_name : str
        One of: "none", "amplitude_slow", "amplitude_fast", "frequency",
        "kerr", "step", "snr", "multi", "white_noise", "tls".
    drift_cfg : DriftConfig
        Drift parameters.

    Returns
    -------
    DriftModel
    """
    from src.drift import (
        AmplitudeDrift,
        DriftModel,
        FrequencyDrift,
        KerrDrift,
        SNRDrift,
        StepDrift,
        WhiteNoiseDrift,
    )

    if drift_name == "none":
        return DriftModel()
    elif drift_name == "amplitude_slow":
        return DriftModel(
            amplitude_drifts=[
                AmplitudeDrift(
                    amplitude=drift_cfg.amplitude_drift_A,
                    frequency=drift_cfg.amplitude_drift_freq,
                )
            ]
        )
    elif drift_name == "amplitude_fast":
        return DriftModel(
            amplitude_drifts=[
                AmplitudeDrift(amplitude=drift_cfg.amplitude_drift_A, frequency=0.02)
            ]
        )
    elif drift_name == "frequency":
        return DriftModel(
            frequency_drifts=[
                FrequencyDrift(
                    amplitude=drift_cfg.frequency_drift_A,
                    frequency=drift_cfg.frequency_drift_freq,
                )
            ]
        )
    elif drift_name == "kerr":
        return DriftModel(
            kerr_drifts=[
                KerrDrift(
                    amplitude=drift_cfg.kerr_drift_A,
                    frequency=drift_cfg.kerr_drift_freq,
                )
            ]
        )
    elif drift_name == "frequency_step":
        from src.drift import SquareWaveFrequencyDrift

        return DriftModel(
            square_wave_drifts=[
                SquareWaveFrequencyDrift(
                    amplitude=drift_cfg.frequency_drift_A,
                    frequency=drift_cfg.frequency_drift_freq,
                )
            ]
        )
    elif drift_name == "step":
        return DriftModel(step_drifts=[StepDrift(step_epoch=100, g2_re_shift=0.3)])
    elif drift_name == "snr":
        return DriftModel(
            snr_drifts=[
                SNRDrift(
                    base_noise=drift_cfg.snr_base_noise,
                    growth_rate=drift_cfg.snr_growth_rate,
                )
            ]
        )
    elif drift_name == "multi":
        return DriftModel(
            amplitude_drifts=[AmplitudeDrift(amplitude=0.2, frequency=0.005)],
            frequency_drifts=[FrequencyDrift(amplitude=0.3, frequency=0.003)],
            kerr_drifts=[KerrDrift(amplitude=0.005, frequency=0.002)],
        )
    elif drift_name == "white_noise":
        return DriftModel(
            white_noise_drifts=[
                WhiteNoiseDrift(
                    sigma_g2=drift_cfg.white_noise_sigma_g2,
                    sigma_epsd=drift_cfg.white_noise_sigma_epsd,
                )
            ]
        )
    elif drift_name == "tls":
        from src.drift import TLSDrift

        return DriftModel(
            tls_drifts=[
                TLSDrift(
                    g_tls=drift_cfg.tls_g,
                    omega_tls=drift_cfg.tls_omega,
                    gamma_tls=drift_cfg.tls_gamma,
                    onset_epoch=drift_cfg.tls_onset_epoch,
                )
            ]
        )
    else:
        raise ValueError(
            f"Unknown drift '{drift_name}'. Options: none, amplitude_slow, "
            "amplitude_fast, frequency, kerr, frequency_step, step, snr, multi, "
            "white_noise, tls"
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

LOCAL = RunConfig(
    name="local",
    cat_params=CatQubitParams(na=10, nb=4, kappa_b=10.0, kappa_a=1.0),
    optimizer=OptimizerConfig(
        population_size=8,
        n_epochs=50,
        sigma0=0.5,
        sigma_floor=0.05,
        learning_rate=0.01,
        hybrid_cma_epochs=15,
        hybrid_grad_steps=5,
        bayesian_n_initial=5,
    ),
    reward=RewardConfig(
        t_probe_z=30.0,
        t_probe_x=0.2,
        npoints=50,
        full_eval_interval=10,
        n_target=3.0,
        t_steady=3.0,
    ),
    benchmark=BenchmarkConfig(
        rewards=["proxy", "enhanced_proxy"],
        optimizers=["cmaes", "reinforce"],
        drifts=["kerr", "tls", "snr"],
    ),
)
"""Fast local preset. Small Hilbert space (dim=40), 50 epochs.
2 rewards x 2 optimizers x 3 drifts = 12 combos. ~5-10 min on laptop."""

MEDIUM = RunConfig(
    name="medium",
    cat_params=CatQubitParams(na=15, nb=5, kappa_b=10.0, kappa_a=1.0),
    optimizer=OptimizerConfig(
        population_size=8,
        n_epochs=60,
        sigma0=0.5,
        sigma_floor=0.05,
        learning_rate=0.01,
        hybrid_cma_epochs=15,
        hybrid_grad_steps=5,
        bayesian_n_initial=5,
    ),
    reward=RewardConfig(
        t_probe_z=50.0,
        t_probe_x=0.3,
        npoints=100,
        full_eval_interval=15,
        n_target=4.0,
        t_steady=5.0,
        use_log_derivative=True,
        n_log_deriv_points=5,
    ),
    benchmark=BenchmarkConfig(
        rewards=["enhanced_proxy"],
        optimizers=["cmaes", "reinforce"],
        drifts=["kerr", "tls", "snr"],
    ),
)
"""Palmetto V100 preset. Full physics (dim=75), 60 epochs, pop=8.
2 optimizers x 1 reward x 3 drifts = 6 combos."""

HPC = RunConfig(
    name="hpc",
    cat_params=CatQubitParams(na=20, nb=6, kappa_b=10.0, kappa_a=1.0),
    optimizer=OptimizerConfig(
        population_size=48,
        n_epochs=500,
        sigma0=0.5,
        sigma_floor=0.03,
        learning_rate=0.005,
        hybrid_cma_epochs=50,
        hybrid_grad_steps=20,
        bayesian_n_initial=15,
    ),
    reward=RewardConfig(
        t_probe_z=80.0,
        t_probe_x=0.3,
        tfinal_z=500.0,
        tfinal_x=2.0,
        npoints=200,
        full_eval_interval=25,
        n_target=4.0,
        t_steady=5.0,
    ),
    benchmark=BenchmarkConfig(
        rewards=["proxy", "photon", "fidelity", "parity", "vacuum"],
        optimizers=["cmaes", "hybrid", "reinforce", "ppo", "bayesian"],
        drifts=[
            "none",
            "amplitude_slow",
            "amplitude_fast",
            "frequency",
            "frequency_step",
            "step",
            "kerr",
            "snr",
            "multi",
            "white_noise",
        ],
    ),
)
"""Palmetto HPC preset. Large Hilbert space (dim=120), many epochs, big populations.
Run on GPU node for best performance. ~1-4 hours per optimization run.
Submit via: sbatch slurm/run_optimization.sh"""


EXPERIMENTAL = RunConfig(
    name="experimental",
    cat_params=CatQubitParams(na=15, nb=5, kappa_b=10.0, kappa_a=0.05, use_double=True),
    optimizer=OptimizerConfig(
        population_size=16,
        n_epochs=150,
        sigma0=0.5,
        sigma_floor=0.05,
        learning_rate=0.01,
    ),
    reward=RewardConfig(
        t_probe_z=200.0,
        t_probe_x=1.0,
        tfinal_z=1000.0,
        tfinal_x=5.0,
        npoints=100,
        full_eval_interval=20,
        n_target=4.0,
        t_steady=5.0,
    ),
    benchmark=BenchmarkConfig(
        rewards=["proxy", "spectral"],
        optimizers=["cmaes"],
        drifts=["none", "amplitude_slow"],
    ),
)
"""Experimental preset matching Berdou et al. 2022 parameters.
kappa_a=0.05 MHz (vs 1.0 default) gives larger alpha, longer lifetimes.
Uses float64 for numerical accuracy. Ref: arXiv:2204.09128."""


def get_config(profile: str = "local") -> RunConfig:
    """Get a run configuration by profile name.

    Returns a deep copy so mutations don't affect the preset.

    Parameters
    ----------
    profile : str
        One of "local", "medium", "hpc", "experimental".

    Returns
    -------
    RunConfig
    """
    profiles = {
        "local": LOCAL,
        "medium": MEDIUM,
        "hpc": HPC,
        "experimental": EXPERIMENTAL,
    }
    if profile not in profiles:
        raise ValueError(
            f"Unknown profile '{profile}'. Choose from: {list(profiles.keys())}"
        )
    return copy.deepcopy(profiles[profile])
