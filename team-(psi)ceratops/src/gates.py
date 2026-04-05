"""Single-qubit gate extension.

Adds a Zeno gate Hamiltonian H_gate = epsilon_Z * (a_dag + a) to the cat qubit
system during active gate operations. The optimizer must maintain
stabilization quality (bias, lifetimes) while the gate is active.

Physics:
  H_total = H_standard + epsilon_Z * (a_dag + a)

  The single-photon drive competes with two-photon stabilization.
  The quantum Zeno effect suppresses transitions out of the cat manifold
  when the stabilization is strong relative to the gate drive.

Reference:
  iQuHack-2025 Task 1.3 -- Zeno gate on dissipative cat qubit.
  epsilon_Z = 0.2 MHz (default from iQuHack-2025).

Control knobs (5 real parameters):
  x = [Re(g2), Im(g2), Re(eps_d), Im(eps_d), epsilon_Z]
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from src.cat_qubit import (
    CatQubitParams,
    build_hamiltonian,
    build_jump_ops,
    build_logical_ops,
    build_operators,
)
from src.optimizers.cmaes_opt import CMAESOptimizer

if TYPE_CHECKING:
    from src.config import GateConfig, RewardConfig, RunConfig

from src.benchmark import RunResult

# ---------------------------------------------------------------------------
# 5D bounds and initial mean
# ---------------------------------------------------------------------------

GATE_BOUNDS = np.array(
    [
        [0.1, 5.0],  # g2_re
        [-2.0, 2.0],  # g2_im
        [0.5, 20.0],  # eps_d_re
        [-5.0, 5.0],  # eps_d_im
        [0.01, 1.0],  # epsilon_z
    ]
)
"""Parameter bounds for the 5D gate optimization.

Rows correspond to [g2_re, g2_im, eps_d_re, eps_d_im, epsilon_z].
The first 4 rows match DEFAULT_BOUNDS from cmaes_opt; the 5th row
constrains the Zeno gate drive strength.
"""

GATE_MEAN = np.array([1.0, 0.0, 4.0, 0.0, 0.2])
"""Initial mean for the 5D gate optimization.

[g2_re=1.0, g2_im=0.0, eps_d_re=4.0, eps_d_im=0.0, epsilon_z=0.2]
The first 4 entries match DEFAULT_MEAN from cmaes_opt; epsilon_z=0.2
is the default gate strength from iQuHack-2025 Task 1.3.
"""


# ---------------------------------------------------------------------------
# Gate Hamiltonian
# ---------------------------------------------------------------------------


def build_gate_hamiltonian(
    a,
    b,
    g2_re: float,
    g2_im: float,
    eps_d_re: float,
    eps_d_im: float,
    epsilon_z: float,
):
    """Build the combined stabilization + Zeno gate Hamiltonian.

    H_total = H_standard + epsilon_z * (a_dag + a)

    The standard two-photon exchange Hamiltonian provides cat qubit
    stabilization, while the single-photon drive epsilon_z * (a_dag + a)
    implements the Zeno gate that rotates the logical qubit.

    Ref: iQuHack-2025 Task 1.3 -- Zeno gate on dissipative cat qubit.

    Parameters
    ----------
    a : QArray
        Storage mode annihilation operator (na*nb x na*nb).
    b : QArray
        Buffer mode annihilation operator (na*nb x na*nb).
    g2_re : float
        Real part of two-photon coupling g2.
    g2_im : float
        Imaginary part of two-photon coupling g2.
    eps_d_re : float
        Real part of buffer drive amplitude eps_d.
    eps_d_im : float
        Imaginary part of buffer drive amplitude eps_d.
    epsilon_z : float
        Zeno gate drive strength [MHz]. Controls the single-photon
        displacement that drives transitions within the cat manifold.

    Returns
    -------
    H_gate : QArray
        Total Hamiltonian: stabilization + Zeno gate.
    """
    # Standard two-photon exchange + buffer drive Hamiltonian
    H_standard = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)

    # Zeno gate: single-photon drive on the storage mode
    # Ref: iQuHack-2025 Task 1.3, H_gate = epsilon_z * (a† + a)
    H_gate = H_standard + epsilon_z * (a.dag() + a)

    return H_gate


# ---------------------------------------------------------------------------
# Gate-aware proxy reward
# ---------------------------------------------------------------------------


def _build_gate_proxy_reward_fn(
    params: CatQubitParams,
    reward_cfg: RewardConfig,
    gate_cfg: GateConfig,
):
    """Build a JIT-compiled proxy reward for 5D gate optimization.

    The reward balances two competing objectives:
      1. Stabilization quality: the cat qubit should maintain long T_Z
         (bit-flip lifetime) even while the gate drive is active.
      2. Gate effect: the single-photon drive should visibly affect the
         logical X expectation, indicating the gate is performing a rotation.

    Strategy:
      - Simulate with the full gate Hamiltonian (stabilization + Zeno drive).
      - Measure <Z_L>(t_probe_z) starting from |+z> -- should still decay
        slowly if stabilization works despite the gate perturbation.
      - Measure <X_L>(t_probe_x) starting from |+x> -- the gate disrupts
        parity, so |<X_L>| should differ from the no-gate case.
      - R = w_stab * log(<Z_L>) + w_gate * log(|<X_L>|)

    Operators are built OUTSIDE the @jit closure for compatibility.

    Parameters
    ----------
    params : CatQubitParams
        Fixed hardware parameters (Hilbert space dims, loss rates).
    reward_cfg : RewardConfig
        Probe times and weight parameters for the base reward.
    gate_cfg : GateConfig
        Gate-specific parameters: duration, weights, bounds.

    Returns
    -------
    reward_fn : callable
        JIT-compiled: x (shape (5,)) -> scalar reward.
    """
    # Build operators outside JIT boundary
    a, b = build_operators(params)
    jump_ops = build_jump_ops(a, b, params)

    # Extract config values as plain floats for JIT closure
    t_probe_z = float(reward_cfg.t_probe_z)
    gate_duration = float(gate_cfg.gate_duration)
    w_stab = float(gate_cfg.w_stabilization)
    w_gate = float(gate_cfg.w_gate_fidelity)

    # Vacuum-based alpha estimation operators (built outside JIT)
    n_a_op = a.dag() @ a
    psi_vacuum = dq.tensor(dq.fock(params.na, 0), dq.fock(params.nb, 0))
    t_settle = float(reward_cfg.t_settle)

    @jit
    def gate_proxy_reward(x):
        """Evaluate 5D gate-aware proxy reward.

        Uses vacuum-based alpha estimation instead of heuristic compute_alpha,
        matching the experimental protocol (Réglade et al. 2024).

        Parameters
        ----------
        x : array, shape (5,)
            [g2_re, g2_im, eps_d_re, eps_d_im, epsilon_z].

        Returns
        -------
        float
            Composite reward balancing stabilization and gate effect.
        """
        g2_re = x[0]
        g2_im = x[1]
        eps_d_re = x[2]
        eps_d_im = x[3]
        epsilon_z = x[4]

        # Build stabilization-only Hamiltonian for vacuum settling
        H_stab = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)

        # Vacuum settle to get data-driven alpha (stabilization-only, no gate)
        res_settle = dq.mesolve(
            H_stab,
            jump_ops,
            psi_vacuum,
            jnp.array([0.0, t_settle]),
            exp_ops=[n_a_op],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )
        n_a_settled = res_settle.expects[0, -1].real
        alpha_mag = jnp.sqrt(jnp.maximum(n_a_settled, 0.01))

        # Phase from drive (exact)
        g2 = g2_re + 1j * g2_im
        eps_d = eps_d_re + 1j * eps_d_im
        theta = jnp.angle(g2 * eps_d) / 2.0
        alpha = alpha_mag * jnp.exp(1j * theta)

        # Build gate Hamiltonian: H_standard + epsilon_z * (a† + a)
        H = build_gate_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im, epsilon_z)

        sx, sz = build_logical_ops(a, b, alpha, params)

        # --- Stabilization component: <Z_L>(t_probe_z) from |+z> ---
        # During gate operation, T_Z should remain long (cat is still stabilized)
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

        # --- Gate component: <X_L>(t_probe_x) from |+x> ---
        # The gate drive disrupts parity; measure how much X changes
        e_state = dq.coherent(params.na, -alpha)
        cat_plus = dq.unit(g_state + e_state)  # even cat |+x>
        psi_x = dq.tensor(cat_plus, dq.fock(params.nb, 0))

        # Use gate_duration as the simulation time for the X measurement
        # to capture the gate's effect over its active period
        tsave_x = jnp.array([0.0, gate_duration])
        res_x = dq.mesolve(
            H,
            jump_ops,
            psi_x,
            tsave_x,
            exp_ops=[sx, sz],
            options=dq.Options(progress_meter=False),
        )
        ex = res_x.expects[0, -1].real  # <X_L> at gate_duration

        # --- Composite reward ---
        # Stabilization: higher <Z_L> at t_probe_z -> better cat protection
        ez_safe = jnp.maximum(ez, 1e-6)
        # Gate effect: |<X_L>| should be measurably different from 1.0
        # A good gate causes parity to change -> |<X_L>| < 1
        ex_safe = jnp.maximum(jnp.abs(ex), 1e-6)

        # R = w_stab * log(<Z_L>) + w_gate * log(|<X_L>|)
        # log(<Z_L>) rewards maintaining stabilization during gate
        # log(|<X_L>|) is negative when gate disrupts parity (desired)
        # The optimizer must find epsilon_z that maximizes total reward:
        #   - Too small epsilon_z: no gate effect (log|<X_L>| ~ 0, no gate)
        #   - Too large epsilon_z: destabilizes cat (log<Z_L> drops)
        #   - Optimal: epsilon_z that rotates while preserving cat manifold
        reward = w_stab * jnp.log(ez_safe) + w_gate * jnp.log(ex_safe)

        return reward

    return gate_proxy_reward


# ---------------------------------------------------------------------------
# Gate reward factory (public interface)
# ---------------------------------------------------------------------------


def build_gate_reward(
    reward_type: str,
    params: CatQubitParams,
    reward_cfg: RewardConfig,
    gate_cfg: GateConfig,
):
    """Build gate-aware reward function and its batched version.

    Uses vacuum-based alpha estimation (data-driven from settling simulation)
    instead of the heuristic compute_alpha formula.

    Parameters
    ----------
    reward_type : str
        Currently only "proxy" is supported for the gate extension.
    params : CatQubitParams
        Fixed hardware parameters.
    reward_cfg : RewardConfig
        Probe times and weight parameters.
    gate_cfg : GateConfig
        Gate-specific parameters (duration, weights, bounds).

    Returns
    -------
    reward_fn : callable
        JIT-compiled: x (shape (5,)) -> scalar reward.
    batched_reward_fn : callable
        JIT-compiled: xs (shape (N, 5)) -> (N,) rewards.

    Raises
    ------
    ValueError
        If reward_type is not "proxy".
    """
    if reward_type != "proxy":
        raise ValueError(
            f"Gate reward only supports 'proxy' type, got '{reward_type}'. "
            "The gate extension requires the fast JIT-compatible proxy reward."
        )

    reward_fn = _build_gate_proxy_reward_fn(params, reward_cfg, gate_cfg)
    batched_reward_fn = jit(vmap(reward_fn))
    return reward_fn, batched_reward_fn


# ---------------------------------------------------------------------------
# Gate benchmark runner
# ---------------------------------------------------------------------------


def run_gate_benchmark(cfg: RunConfig, verbose: bool = True) -> dict[str, RunResult]:
    """Run CMA-ES optimization with and without the Zeno gate for comparison.

    Performs two optimization runs:
      1. "no_gate": standard 4D optimization (baseline, no gate drive)
      2. "with_gate": 5D optimization including epsilon_z gate strength

    Both runs use the proxy reward and CMA-ES optimizer. The gate run
    maintains stabilization quality while the gate Hamiltonian is active.

    Parameters
    ----------
    cfg : RunConfig
        Full configuration including gate, reward, and optimizer settings.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict[str, RunResult]
        {"no_gate": RunResult, "with_gate": RunResult}
    """
    from src.reward import build_reward

    params = cfg.cat_params
    gate_cfg = cfg.gate
    reward_cfg = cfg.reward
    opt_cfg = cfg.optimizer

    results = {}

    # -----------------------------------------------------------------
    # Run 1: No gate (standard 4D baseline)
    # -----------------------------------------------------------------
    if verbose:
        print("[gate-benchmark] Run 1/2: No gate (4D baseline)")

    reward_fn_4d, batched_fn_4d = build_reward("proxy", params, reward_cfg)

    optimizer_4d = CMAESOptimizer(
        population_size=opt_cfg.population_size,
        sigma0=opt_cfg.sigma0,
        sigma_floor=opt_cfg.sigma_floor,
        seed=opt_cfg.seed,
    )

    # Compile
    dummy_4d = jnp.zeros(4)
    _ = reward_fn_4d(dummy_4d)

    result_no_gate = RunResult(
        reward_type="proxy",
        optimizer_type="cmaes",
        drift_type="none",
        config_name=cfg.name,
    )

    t_start = time.time()
    n_epochs = opt_cfg.n_epochs

    for epoch in range(n_epochs):
        xs = optimizer_4d.ask()
        rewards = batched_fn_4d(xs)
        optimizer_4d.tell(xs, rewards)

        result_no_gate.reward_history.append(float(jnp.mean(rewards)))
        result_no_gate.reward_std_history.append(float(jnp.std(rewards)))
        result_no_gate.param_history.append(np.array(optimizer_4d.get_best()))

        if verbose and epoch % max(1, n_epochs // 5) == 0:
            print(
                f"  Epoch {epoch:4d}/{n_epochs} | reward={float(jnp.mean(rewards)):.4f}"
            )

    result_no_gate.wall_time = time.time() - t_start
    result_no_gate.n_epochs = n_epochs
    results["no_gate"] = result_no_gate

    if verbose:
        print(
            f"[gate-benchmark] No gate done in {result_no_gate.wall_time:.1f}s | "
            f"best_reward={float(optimizer_4d.best_reward):.4f}"
        )

    # -----------------------------------------------------------------
    # Run 2: With gate (5D optimization)
    # -----------------------------------------------------------------
    if verbose:
        print("\n[gate-benchmark] Run 2/2: With Zeno gate (5D)")

    reward_fn_5d, batched_fn_5d = build_gate_reward(
        "proxy", params, reward_cfg, gate_cfg
    )

    # Build bounds and mean from config (override module-level defaults)
    gc = cfg.gate
    gate_bounds = np.array(
        [
            [0.1, 5.0],
            [-2.0, 2.0],
            [0.5, 20.0],
            [-5.0, 5.0],
            [gc.epsilon_z_min, gc.epsilon_z_max],
        ]
    )
    gate_mean = np.array(
        [
            cfg.optimizer.init_params[0],
            cfg.optimizer.init_params[1],
            cfg.optimizer.init_params[2],
            cfg.optimizer.init_params[3],
            gc.epsilon_z_init,
        ]
    )

    optimizer_5d = CMAESOptimizer(
        mean0=gate_mean,
        sigma0=opt_cfg.sigma0,
        bounds=gate_bounds,
        population_size=opt_cfg.population_size,
        sigma_floor=opt_cfg.sigma_floor,
        seed=opt_cfg.seed,
    )

    # Compile
    dummy_5d = jnp.zeros(5)
    _ = reward_fn_5d(dummy_5d)

    result_with_gate = RunResult(
        reward_type="proxy_gate",
        optimizer_type="cmaes",
        drift_type="none",
        config_name=cfg.name,
    )

    t_start = time.time()

    for epoch in range(n_epochs):
        xs = optimizer_5d.ask()
        rewards = batched_fn_5d(xs)
        optimizer_5d.tell(xs, rewards)

        result_with_gate.reward_history.append(float(jnp.mean(rewards)))
        result_with_gate.reward_std_history.append(float(jnp.std(rewards)))
        result_with_gate.param_history.append(np.array(optimizer_5d.get_best()))

        if verbose and epoch % max(1, n_epochs // 5) == 0:
            best_5d = optimizer_5d.get_best()
            print(
                f"  Epoch {epoch:4d}/{n_epochs} | "
                f"reward={float(jnp.mean(rewards)):.4f} | "
                f"epsilon_z={float(best_5d[4]):.4f}"
            )

    result_with_gate.wall_time = time.time() - t_start
    result_with_gate.n_epochs = n_epochs
    results["with_gate"] = result_with_gate

    if verbose:
        best = optimizer_5d.get_best()
        print(
            f"[gate-benchmark] With gate done in {result_with_gate.wall_time:.1f}s | "
            f"best_reward={float(optimizer_5d.best_reward):.4f}"
        )
        print(
            f"  Optimal params: g2={float(best[0]):.3f}+{float(best[1]):.3f}j, "
            f"eps_d={float(best[2]):.3f}+{float(best[3]):.3f}j, "
            f"epsilon_z={float(best[4]):.4f}"
        )
        print(
            f"\n[gate-benchmark] Summary: "
            f"no_gate={float(results['no_gate'].reward_history[-1]):.4f} vs "
            f"with_gate={float(results['with_gate'].reward_history[-1]):.4f}"
        )

    return results
