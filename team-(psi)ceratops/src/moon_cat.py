"""Moon cat (squeezed cat) extension.

Adds squeezing interaction g2*lam*a.dag()@a@b to the standard cat Hamiltonian,
deforming circular cat blobs into crescent-shaped "moon" states in phase space.

Physics:
  H_moon = H_standard + g2 * lam * a.dag() @ a @ b

  The squeezing parameter lam increases phase-space separation between basis
  states for the same mean photon number, enhancing bit-flip protection.

  Expected improvement: 160x in T_X at same n_bar (Rousseau et al. 2025).
  Scaling exponent: gamma = 4.3 (vs ~1 for standard cat).

Reference:
  Rousseau et al. "Enhancing dissipative cat qubit protection by squeezing."
  arXiv:2502.07892 (2025).

Control knobs (5 real parameters):
  x = [Re(g2), Im(g2), Re(eps_d), Im(eps_d), lam]
"""

from __future__ import annotations

import time
import warnings

# Lazy import to avoid circular dependency — used in type hints only
from typing import TYPE_CHECKING

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from src.benchmark import RunResult
from src.cat_qubit import (
    DEFAULT_PARAMS,
    CatQubitParams,
    build_hamiltonian,
    build_initial_states,
    build_jump_ops,
    build_logical_ops,
    build_operators,
    compute_alpha,
    measure_lifetimes,
    robust_exp_fit,
)
from src.optimizers.cmaes_opt import CMAESOptimizer
from src.reward import _estimate_T_from_log_derivative, build_reward

if TYPE_CHECKING:
    from src.config import RewardConfig, RunConfig

# ---------------------------------------------------------------------------
# Moon cat bounds and initial mean (5D: standard 4 + lambda)
# ---------------------------------------------------------------------------

MOON_CAT_BOUNDS = np.array(
    [
        [0.1, 5.0],  # g2_re
        [-2.0, 2.0],  # g2_im
        [0.5, 20.0],  # eps_d_re
        [-5.0, 5.0],  # eps_d_im
        [0.0, 1.0],  # lambda (squeezing parameter)
    ]
)

MOON_CAT_MEAN = np.array([1.0, 0.0, 4.0, 0.0, 0.3])


# ---------------------------------------------------------------------------
# Moon cat Hamiltonian
# ---------------------------------------------------------------------------


def build_moon_hamiltonian(
    a, b, g2_re: float, g2_im: float, eps_d_re: float, eps_d_im: float, lam: float
):
    """Build the moon cat Hamiltonian: standard cat + squeezing interaction.

    H_moon = H_standard + g2 * lam * a.dag() @ a @ b

    The additional term introduces a squeezing interaction that deforms the
    cat state phase-space distribution from circular blobs into crescents,
    increasing the effective distance between code words.

    Parameters
    ----------
    a, b : QArray
        Storage and buffer annihilation operators.
    g2_re, g2_im : float
        Real and imaginary parts of the two-photon coupling g2.
    eps_d_re, eps_d_im : float
        Real and imaginary parts of the buffer drive amplitude eps_d.
    lam : float
        Squeezing parameter lambda in [0, 1]. Controls moon deformation strength.

    Returns
    -------
    H_moon : QArray
        Moon cat Hamiltonian.

    Reference
    ---------
    Rousseau et al. (2025), arXiv:2502.07892, Eq. (3) — squeezing interaction.
    """
    # Standard two-photon exchange + buffer drive Hamiltonian
    H_standard = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)

    # Squeezing (moon) interaction: Hermitian form
    # H_squeeze = g2 * lam * a†a * b + conj(g2) * lam * a†a * b†
    # Ref: Rousseau et al. (2025), arXiv:2502.07892
    g2 = g2_re + 1j * g2_im
    n_a = a.dag() @ a
    H_squeeze = g2 * lam * n_a @ b + jnp.conj(g2) * lam * n_a @ b.dag()
    H_moon = H_standard + H_squeeze

    return H_moon


# ---------------------------------------------------------------------------
# Moon cat lifetime measurement (uses moon Hamiltonian, not standard)
# ---------------------------------------------------------------------------


def measure_moon_lifetimes(
    g2_re: float,
    g2_im: float,
    eps_d_re: float,
    eps_d_im: float,
    lam: float,
    tfinal_z: float = 200.0,
    tfinal_x: float = 1.0,
    npoints: int = 100,
    params: CatQubitParams = DEFAULT_PARAMS,
) -> dict:
    """Measure T_Z and T_X using the moon cat Hamiltonian (with squeezing).

    Unlike measure_lifetimes() which uses the standard Hamiltonian, this
    includes the squeezing term g2*lam*a†a*b so that lifetimes reflect
    the actual moon cat physics.

    Parameters
    ----------
    g2_re, g2_im, eps_d_re, eps_d_im : float
        Control knobs (same as standard cat).
    lam : float
        Squeezing parameter lambda.
    tfinal_z, tfinal_x : float
        Simulation durations for T_Z and T_X [us].
    npoints : int
        Number of save points for the simulation.
    params : CatQubitParams
        Fixed hardware parameters.

    Returns
    -------
    dict
        "Tz": bit-flip lifetime, "Tx": phase-flip lifetime, "bias": Tz/Tx.
    """
    a, b = build_operators(params)
    H = build_moon_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im, lam)
    jump_ops = build_jump_ops(a, b, params)

    # --- Vacuum-based alpha estimation (NOT heuristic compute_alpha) ---
    # Settle from vacuum to get data-driven |α| from ⟨a†a⟩, matching the
    # experimental protocol (Réglade et al. 2024). This is critical for the
    # moon cat because compute_alpha ignores the squeezing parameter λ.
    n_a_op = a.dag() @ a
    psi_vacuum = dq.tensor(dq.fock(params.na, 0), dq.fock(params.nb, 0))
    t_settle = 15.0  # ~5-10× κ₂⁻¹
    res_settle = dq.mesolve(
        H,
        jump_ops,
        psi_vacuum,
        jnp.array([0.0, t_settle]),
        exp_ops=[n_a_op],
        options=dq.Options(progress_meter=False),
    )
    n_a_settled = float(res_settle.expects[0, -1].real)
    alpha_mag = np.sqrt(max(n_a_settled, 0.01))

    # Phase from drive (exact, not heuristic — only magnitude is approximate)
    g2 = g2_re + 1j * g2_im
    eps_d = eps_d_re + 1j * eps_d_im
    theta = float(np.angle(g2 * eps_d)) / 2.0
    alpha_est = alpha_mag * np.exp(1j * theta)

    sx, sz = build_logical_ops(a, b, alpha_est, params)
    init_states = build_initial_states(alpha_est, params)

    # --- T_Z: fit <Z_L> decay from |+z> ---
    psi_z = init_states["+z"]
    tsave_z = jnp.linspace(0, tfinal_z, npoints)
    res_z = dq.mesolve(
        H,
        jump_ops,
        psi_z,
        tsave_z,
        exp_ops=[sx, sz],
        options=dq.Options(progress_meter=False),
    )
    exp_z = res_z.expects[1, :].real
    fit_z = robust_exp_fit(tsave_z, exp_z)
    Tz = fit_z["tau"]

    # --- T_X: fit <X_L> decay from |+x> ---
    psi_x = init_states["+x"]
    tsave_x = jnp.linspace(0, tfinal_x, npoints)
    res_x = dq.mesolve(
        H,
        jump_ops,
        psi_x,
        tsave_x,
        exp_ops=[sx, sz],
        options=dq.Options(progress_meter=False),
    )
    exp_x = res_x.expects[0, :].real
    fit_x = robust_exp_fit(tsave_x, exp_x)
    Tx = fit_x["tau"]

    if np.isnan(Tz):
        warnings.warn("measure_moon_lifetimes: T_Z fit returned NaN", stacklevel=2)
    if np.isnan(Tx):
        warnings.warn("measure_moon_lifetimes: T_X fit returned NaN", stacklevel=2)
    if np.isnan(Tz) or np.isnan(Tx):
        return {
            "Tz": Tz,
            "Tx": Tx,
            "bias": float("nan"),
            "alpha_vacuum": float(np.abs(alpha_est)),
        }
    Tx_safe = max(Tx, 1e-6)
    return {
        "Tz": Tz,
        "Tx": Tx,
        "bias": Tz / Tx_safe,
        "alpha_vacuum": float(np.abs(alpha_est)),
    }


# ---------------------------------------------------------------------------
# Moon cat proxy reward (5D, JIT-compatible)
# ---------------------------------------------------------------------------


def _build_moon_proxy_loss_fn(
    params: CatQubitParams = DEFAULT_PARAMS,
    t_probe_z: float = 50.0,
    t_probe_x: float = 0.3,
    target_bias: float = 100.0,
    w_lifetime: float = 1.0,
    w_bias: float = 0.5,
):
    """Build a JIT-compiled proxy reward for the 5D moon cat parameter space.

    Same proxy strategy as _build_proxy_loss_fn (single-point expectations),
    but uses the moon cat Hamiltonian with the additional squeezing parameter.

    WARNING: This proxy uses the heuristic compute_alpha which does NOT
    account for the squeezing parameter lambda. The optimizer still receives
    gradient signal for lambda through the dynamics (mesolve uses the full
    moon Hamiltonian H(λ)), but the initial states and logical measurement
    operators are built with a λ-independent α, making the reward signal
    suboptimal for λ. For physically correct alpha estimation, use
    build_moon_reward("vacuum", ...) instead.

    Parameters
    ----------
    params : CatQubitParams
        Fixed hardware parameters.
    t_probe_z : float
        Probe time for T_Z measurement [us].
    t_probe_x : float
        Probe time for T_X measurement [us].
    target_bias : float
        Target eta = T_Z / T_X.
    w_lifetime : float
        Weight on lifetime maximization component.
    w_bias : float
        Weight on bias targeting component.

    Returns
    -------
    callable
        JIT-compiled reward function: x (shape (5,)) -> scalar.
    """
    # Build operators OUTSIDE @jit closure (static, expensive to construct)
    a, b = build_operators(params)
    jump_ops = build_jump_ops(a, b, params)

    # Parity operator: exp(i*pi*a†a) — alpha-independent logical X
    sx = (1j * jnp.pi * a.dag() @ a).expm()

    def moon_proxy_reward(x):
        g2_re, g2_im, eps_d_re, eps_d_im, lam = x[0], x[1], x[2], x[3], x[4]

        # Build moon cat Hamiltonian
        H = build_moon_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im, lam)

        # Estimate alpha and build logical Z operator
        alpha = compute_alpha(g2_re, g2_im, eps_d_re, eps_d_im, params)
        _, sz = build_logical_ops(a, b, alpha, params)

        # --- Measure <Z_L> at t_probe_z from |+z> ---
        g_state = dq.coherent(params.na, alpha)
        psi_z = dq.tensor(g_state, dq.fock(params.nb, 0))

        tsave_z = jnp.array([0.0, t_probe_z])
        res_z = dq.mesolve(
            H,
            jump_ops,
            psi_z,
            tsave_z,
            exp_ops=[sx, sz],
            options=dq.Options(progress_meter=False),
        )
        ez = res_z.expects[1, -1].real  # <Z_L> at t_probe_z

        # --- Measure <X_L> at t_probe_x from |+x> ---
        e_state = dq.coherent(params.na, -alpha)
        cat_plus = dq.unit(g_state + e_state)
        psi_x = dq.tensor(cat_plus, dq.fock(params.nb, 0))

        tsave_x = jnp.array([0.0, t_probe_x])
        res_x = dq.mesolve(
            H,
            jump_ops,
            psi_x,
            tsave_x,
            exp_ops=[sx, sz],
            options=dq.Options(progress_meter=False),
        )
        ex = res_x.expects[0, -1].real  # <X_L> at t_probe_x

        # --- Composite reward ---
        # Clamp to avoid log(0)
        ez_safe = jnp.maximum(ez, 1e-6)
        ex_safe = jnp.maximum(jnp.abs(ex), 1e-6)

        # Lifetime component: higher expectations -> longer lifetimes
        lifetime_score = w_lifetime * (jnp.log(ez_safe) + jnp.log(ex_safe))

        # Bias proxy: ratio of log-expectations relates to lifetime ratio
        log_ez = jnp.minimum(jnp.log(ez_safe), -1e-6)
        log_ex = jnp.minimum(jnp.log(ex_safe), -1e-6)
        bias_proxy = (t_probe_z / t_probe_x) * (log_ex / log_ez)

        # Use same relative-error formula as other rewards for consistency
        bias_penalty = -w_bias * ((bias_proxy - target_bias) / target_bias) ** 2

        return lifetime_score + bias_penalty

    return moon_proxy_reward


# ---------------------------------------------------------------------------
# Moon cat vacuum reward (alpha-free, physically correct)
# ---------------------------------------------------------------------------


def _build_moon_vacuum_reward_fn(
    params: CatQubitParams = DEFAULT_PARAMS,
    t_settle: float = 15.0,
    t_measure_z: float = 200.0,
    t_measure_x: float = 1.0,
    n_measure_points: int = 5,
    target_bias: float = 100.0,
    w_lifetime: float = 1.0,
    w_bias: float = 0.5,
):
    """Build an alpha-free vacuum reward for the 5D moon cat parameter space.

    Same approach as _build_vacuum_reward_fn (src/reward.py) but uses the moon
    cat Hamiltonian with the squeezing parameter lambda. NOT JIT-compatible
    (dynamic alpha computation between mesolve calls). Batched via Python loop.

    This is the physically correct reward for moon cat optimization because:
    1. compute_alpha ignores the squeezing parameter lambda entirely
    2. Vacuum-based alpha matches the experimental protocol (Réglade et al. 2024)

    Parameters
    ----------
    params : CatQubitParams
        Fixed hardware parameters.
    t_settle : float
        Settling time for vacuum → cat [us].
    t_measure_z : float
        T_Z measurement window after settling [us].
    t_measure_x : float
        T_X measurement window after settling [us].
    n_measure_points : int
        Number of time points for log-derivative fit.
    target_bias : float
        Target η = T_Z / T_X.
    w_lifetime, w_bias : float
        Reward weights.

    Returns
    -------
    callable
        Reward function: x (shape (5,)) -> scalar. NOT JIT-compiled.
    """
    a, b = build_operators(params)
    jump_ops = build_jump_ops(a, b, params)

    # Parity operator: P = exp(iπ a†a) — α-independent logical X
    parity_op = (1j * jnp.pi * a.dag() @ a).expm()
    n_a_op = a.dag() @ a
    psi_vacuum = dq.tensor(dq.fock(params.na, 0), dq.fock(params.nb, 0))

    # T_X time grid
    fracs_x = jnp.linspace(1.0 / n_measure_points, 1.0, n_measure_points)
    t_measure_pts_x = t_settle + fracs_x * t_measure_x
    tsave_x = jnp.concatenate(
        [
            jnp.array([0.0, t_settle]),
            t_measure_pts_x,
        ]
    )
    t_rel_x = fracs_x * t_measure_x

    # T_Z time grid
    fracs_z = jnp.linspace(1.0 / n_measure_points, 1.0, n_measure_points)
    t_measure_pts_z = fracs_z * t_measure_z
    tsave_z = jnp.concatenate([jnp.array([0.0]), t_measure_pts_z])
    t_rel_z = fracs_z * t_measure_z

    def moon_vacuum_reward(x):
        g2_re, g2_im, eps_d_re, eps_d_im, lam = x[0], x[1], x[2], x[3], x[4]

        H = build_moon_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im, lam)

        # Well orientation from drive phase (exact)
        g2 = g2_re + 1j * g2_im
        eps_d = eps_d_re + 1j * eps_d_im
        theta = jnp.angle(g2 * eps_d) / 2.0

        # --- T_X: parity decay from vacuum + data-driven α ---
        res_x = dq.mesolve(
            H,
            jump_ops,
            psi_vacuum,
            tsave_x,
            exp_ops=[parity_op, n_a_op],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )
        parity_vals = jnp.abs(res_x.expects[0, 2:].real)
        T_X_est = _estimate_T_from_log_derivative(t_rel_x, parity_vals)

        # Data-driven α from ⟨a†a⟩ at settling time
        n_a_settled = res_x.expects[1, 1].real
        alpha_mag = jnp.sqrt(jnp.maximum(n_a_settled, 0.01))
        alpha_est = alpha_mag * jnp.exp(1j * theta)

        # --- T_Z: quadrature decay from |α_est⟩ ---
        Q_theta = a * jnp.exp(-1j * theta) + a.dag() * jnp.exp(1j * theta)
        psi_alpha = dq.tensor(dq.coherent(params.na, alpha_est), dq.fock(params.nb, 0))
        res_z = dq.mesolve(
            H,
            jump_ops,
            psi_alpha,
            tsave_z,
            exp_ops=[Q_theta],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )
        q_vals = jnp.abs(res_z.expects[0, 1:].real)
        T_Z_est = _estimate_T_from_log_derivative(t_rel_z, q_vals)

        # Reward: same formula as vacuum reward
        lifetime_score = w_lifetime * (
            jnp.log(jnp.maximum(T_Z_est, 1e-6)) + jnp.log(jnp.maximum(T_X_est, 1e-6))
        )
        bias = T_Z_est / jnp.maximum(T_X_est, 1e-6)
        bias_penalty = -w_bias * ((bias - target_bias) / target_bias) ** 2

        return lifetime_score + bias_penalty

    return moon_vacuum_reward


# ---------------------------------------------------------------------------
# Moon cat reward factory
# ---------------------------------------------------------------------------


def build_moon_reward(
    reward_type: str,
    params: CatQubitParams,
    reward_cfg: RewardConfig | None = None,
):
    """Build moon cat reward function and its batched version.

    Like build_reward but for 5D vectors x = [g2_re, g2_im, eps_d_re, eps_d_im, lam].

    Reward types:
      - "proxy": JIT-compiled, fast, but uses heuristic alpha (ignores lambda).
      - "vacuum": Alpha-free, physically correct. NOT JIT-compiled (~N× slower).

    Parameters
    ----------
    reward_type : str
        One of "proxy" or "vacuum".
    params : CatQubitParams
        Fixed hardware parameters.
    reward_cfg : RewardConfig or None
        If None, use default kwargs.

    Returns
    -------
    reward_fn : callable
        Reward function: x (shape (5,)) -> scalar reward.
    batched_reward_fn : callable
        Batched version: xs (shape (N, 5)) -> (N,) rewards.

    Raises
    ------
    ValueError
        If reward_type is not recognized.
    """
    if reward_cfg is not None:
        cfg = reward_cfg
    else:
        from src.config import RewardConfig

        cfg = RewardConfig()

    if reward_type == "proxy":
        warnings.warn(
            "Moon cat proxy reward uses heuristic compute_alpha which ignores "
            "the squeezing parameter lambda. The optimizer still receives "
            "gradient signal for lambda through the Hamiltonian dynamics, but "
            "initial states and measurement operators use a lambda-independent "
            "alpha — making the reward suboptimal for lambda tuning. Use "
            "reward_type='vacuum' for physically correct moon cat optimization.",
            stacklevel=2,
        )
        reward_fn = _build_moon_proxy_loss_fn(
            params,
            t_probe_z=cfg.t_probe_z,
            t_probe_x=cfg.t_probe_x,
            target_bias=cfg.target_bias,
            w_lifetime=cfg.w_lifetime,
            w_bias=cfg.w_bias,
        )
        batched_reward_fn = jit(vmap(reward_fn))
        return reward_fn, batched_reward_fn

    elif reward_type == "vacuum":
        reward_fn = _build_moon_vacuum_reward_fn(
            params,
            t_settle=cfg.t_settle,
            t_measure_z=cfg.t_measure_z,
            t_measure_x=cfg.t_measure_x,
            n_measure_points=cfg.n_log_deriv_points,
            target_bias=cfg.target_bias,
            w_lifetime=cfg.w_lifetime,
            w_bias=cfg.w_bias,
        )

        # Non-JIT: batch via Python loop
        def _batched_loop(xs):
            results = []
            for i in range(xs.shape[0]):
                try:
                    results.append(float(reward_fn(xs[i])))
                except Exception as e:
                    warnings.warn(
                        f"Moon cat reward eval failed for sample {i}: {e}", stacklevel=2
                    )
                    results.append(float("-inf"))
            return jnp.array(results)

        return reward_fn, _batched_loop

    else:
        raise ValueError(
            f"Unknown reward_type '{reward_type}' for moon cat. "
            "Options: 'proxy', 'vacuum'."
        )


# ---------------------------------------------------------------------------
# Moon cat vs standard cat comparison
# ---------------------------------------------------------------------------


def run_moon_cat_comparison(cfg: RunConfig, verbose: bool = True) -> dict:
    """Run CMA-ES optimization for standard cat (4D) AND moon cat (5D).

    Uses the same config (epochs, population, reward settings) for both,
    enabling a direct comparison of optimized lifetimes and bias.

    Parameters
    ----------
    cfg : RunConfig
        Full run configuration. Uses cfg.optimizer for CMA-ES hyperparams,
        cfg.reward for proxy reward settings, cfg.cat_params for hardware,
        and cfg.moon_cat for lambda bounds/initial value.
    verbose : bool
        Print progress and comparison summary.

    Returns
    -------
    dict
        {"standard": RunResult, "moon": RunResult} with full optimization
        histories and final validated lifetimes.
    """
    params = cfg.cat_params
    n_epochs = cfg.optimizer.n_epochs
    pop_size = cfg.optimizer.population_size

    # ------------------------------------------------------------------
    # Standard cat (4D)
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("  TASK 4: Moon Cat Comparison")
        print("=" * 60)
        print("\n[standard] Building 4D reward and optimizer...")

    std_reward_fn, std_batched_fn = build_reward("proxy", params, cfg.reward)
    std_optimizer = CMAESOptimizer(
        population_size=pop_size,
        sigma0=cfg.optimizer.sigma0,
        sigma_floor=cfg.optimizer.sigma_floor,
        seed=cfg.optimizer.seed,
    )

    # Compile
    if verbose:
        print("[standard] Compiling reward function...")
    dummy_4d = jnp.zeros(4)
    _ = std_reward_fn(dummy_4d)

    std_result = RunResult(
        reward_type="proxy",
        optimizer_type="cmaes",
        drift_type="none",
        config_name=cfg.name,
    )

    if verbose:
        print(f"[standard] Starting {n_epochs} epochs (4D)...")

    t_start = time.time()
    for epoch in range(n_epochs):
        xs = std_optimizer.ask()
        rewards = std_batched_fn(xs)
        std_optimizer.tell(xs, rewards)

        std_result.reward_history.append(float(jnp.mean(rewards)))
        std_result.reward_std_history.append(float(jnp.std(rewards)))
        std_result.param_history.append(np.array(std_optimizer.get_best()))

        if verbose and epoch % max(1, n_epochs // 5) == 0:
            print(
                f"  [standard] Epoch {epoch:4d}/{n_epochs} | "
                f"reward={float(jnp.mean(rewards)):.4f}"
            )

    std_result.wall_time = time.time() - t_start
    std_result.n_epochs = n_epochs

    # Validate standard cat
    best_std = std_optimizer.get_best()
    try:
        lt_std = measure_lifetimes(
            float(best_std[0]),
            float(best_std[1]),
            float(best_std[2]),
            float(best_std[3]),
            tfinal_z=cfg.reward.tfinal_z,
            tfinal_x=cfg.reward.tfinal_x,
            params=params,
        )
        alpha_std = compute_alpha(
            float(best_std[0]),
            float(best_std[1]),
            float(best_std[2]),
            float(best_std[3]),
            params,
        )
        std_result.validation_history.append(
            {
                "epoch": n_epochs,
                "Tz": lt_std["Tz"],
                "Tx": lt_std["Tx"],
                "bias": lt_std["bias"],
                "alpha": float(alpha_std),
            }
        )
    except Exception as e:
        lt_std = {"Tz": float("nan"), "Tx": float("nan"), "bias": float("nan")}
        if verbose:
            print(f"  [standard] Validation failed: {e}")

    # ------------------------------------------------------------------
    # Moon cat (5D)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[moon] Building 5D reward and optimizer...")

    # Deliberately use "vacuum" (not "proxy") for moon cat: vacuum reward estimates
    # alpha from actual ⟨a†a⟩ so it correctly captures the effect of lambda on the
    # steady state, unlike the proxy which uses heuristic compute_alpha(ignores λ).
    moon_reward_fn, moon_batched_fn = build_moon_reward("vacuum", params, cfg.reward)

    # Build bounds and mean from config (override module-level defaults)
    mc = cfg.moon_cat
    moon_bounds = np.array(
        [
            [0.1, 5.0],  # g2_re
            [-2.0, 2.0],  # g2_im
            [0.5, 20.0],  # eps_d_re
            [-5.0, 5.0],  # eps_d_im
            [mc.lambda_min, mc.lambda_max],  # lambda
        ]
    )
    moon_mean = np.array(
        [
            cfg.optimizer.init_params[0],
            cfg.optimizer.init_params[1],
            cfg.optimizer.init_params[2],
            cfg.optimizer.init_params[3],
            mc.lambda_init,
        ]
    )

    moon_optimizer = CMAESOptimizer(
        mean0=moon_mean,
        sigma0=cfg.optimizer.sigma0,
        bounds=moon_bounds,
        population_size=pop_size,
        sigma_floor=cfg.optimizer.sigma_floor,
        seed=cfg.optimizer.seed,
    )

    # Compile
    if verbose:
        print("[moon] Compiling reward function...")
    dummy_5d = jnp.zeros(5)
    _ = moon_reward_fn(dummy_5d)

    moon_result = RunResult(
        reward_type="proxy_moon",
        optimizer_type="cmaes",
        drift_type="none",
        config_name=cfg.name,
    )

    if verbose:
        print(f"[moon] Starting {n_epochs} epochs (5D)...")

    t_start = time.time()
    for epoch in range(n_epochs):
        xs = moon_optimizer.ask()
        rewards = moon_batched_fn(xs)
        moon_optimizer.tell(xs, rewards)

        moon_result.reward_history.append(float(jnp.mean(rewards)))
        moon_result.reward_std_history.append(float(jnp.std(rewards)))
        moon_result.param_history.append(np.array(moon_optimizer.get_best()))

        if verbose and epoch % max(1, n_epochs // 5) == 0:
            print(
                f"  [moon] Epoch {epoch:4d}/{n_epochs} | "
                f"reward={float(jnp.mean(rewards)):.4f}"
            )

    moon_result.wall_time = time.time() - t_start
    moon_result.n_epochs = n_epochs

    # Validate moon cat using the MOON Hamiltonian (includes squeezing term)
    best_moon = moon_optimizer.get_best()
    try:
        lt_moon = measure_moon_lifetimes(
            float(best_moon[0]),
            float(best_moon[1]),
            float(best_moon[2]),
            float(best_moon[3]),
            float(best_moon[4]),
            tfinal_z=cfg.reward.tfinal_z,
            tfinal_x=cfg.reward.tfinal_x,
            params=params,
        )
        moon_result.validation_history.append(
            {
                "epoch": n_epochs,
                "Tz": lt_moon["Tz"],
                "Tx": lt_moon["Tx"],
                "bias": lt_moon["bias"],
                "alpha_vacuum": lt_moon.get("alpha_vacuum", float("nan")),
                "lambda": float(best_moon[4]),
            }
        )
    except Exception as e:
        lt_moon = {"Tz": float("nan"), "Tx": float("nan"), "bias": float("nan")}
        if verbose:
            print(f"  [moon] Validation failed: {e}")

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("  COMPARISON: Standard Cat vs Moon Cat")
        print("=" * 60)
        print(f"  {'Metric':<20s} {'Standard (4D)':>15s} {'Moon (5D)':>15s}")
        print("  " + "-" * 50)
        print(f"  {'T_Z [us]':<20s} {lt_std['Tz']:>15.2f} {lt_moon['Tz']:>15.2f}")
        print(f"  {'T_X [us]':<20s} {lt_std['Tx']:>15.4f} {lt_moon['Tx']:>15.4f}")
        print(
            f"  {'Bias (T_Z/T_X)':<20s} {lt_std['bias']:>15.1f} {lt_moon['bias']:>15.1f}"
        )
        print(f"  {'Best params':<20s} {'(4D)':>15s} {'(5D)':>15s}")
        print(
            f"  {'  g2_re':<20s} {float(best_std[0]):>15.4f} {float(best_moon[0]):>15.4f}"
        )
        print(
            f"  {'  g2_im':<20s} {float(best_std[1]):>15.4f} {float(best_moon[1]):>15.4f}"
        )
        print(
            f"  {'  eps_d_re':<20s} {float(best_std[2]):>15.4f} {float(best_moon[2]):>15.4f}"
        )
        print(
            f"  {'  eps_d_im':<20s} {float(best_std[3]):>15.4f} {float(best_moon[3]):>15.4f}"
        )
        print(f"  {'  lambda':<20s} {'N/A':>15s} {float(best_moon[4]):>15.4f}")
        print(
            f"\n  Wall time: standard={std_result.wall_time:.1f}s, "
            f"moon={moon_result.wall_time:.1f}s"
        )
        print("=" * 60)

    return {"standard": std_result, "moon": moon_result}
