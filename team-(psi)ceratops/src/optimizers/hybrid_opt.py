"""Hybrid CMA-ES + Gradient optimizer for cat qubit control.

Hybrid optimizer: global exploration via CMA-ES + local refinement via gradient descent.

Alternates between CMA-ES phases (population-based global search) and Adam
gradient descent phases (local refinement from the CMA-ES mean). After each
gradient phase, the refined point is injected back into CMA-ES as the new mean.

This combines the global exploration strength of CMA-ES with the fast local
convergence of first-order gradient methods, bridging the gap between
black-box and differentiable optimization.

Reference:
  Loshchilov, I. "CMA-ES with Restarts and Increasing Population Size."
  Congress on Evolutionary Computation (2005).
  — Restart strategy inspiration; here we use gradient refinement instead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.optimizers.base import OnlineOptimizer
from src.optimizers.cmaes_opt import DEFAULT_BOUNDS, CMAESOptimizer


class HybridOptimizer(OnlineOptimizer):
    """Hybrid CMA-ES + Adam optimizer.

    Alternates phases: cma_epochs of CMA-ES exploration followed by
    grad_steps of Adam refinement, then repeats. After the gradient phase,
    the CMA-ES mean is set to the refined point and sigma is partially reset.

    Parameters
    ----------
    reward_fn : callable
        Differentiable reward function: x (shape (4,)) -> scalar.
        Must be JIT-compatible for gradient computation.
    cma_epochs : int
        Number of CMA-ES generations per phase. Default: 30.
    grad_steps : int
        Number of Adam gradient steps per phase. Default: 10.
    learning_rate : float
        Adam learning rate for gradient phase. Default: 0.01.
    population_size : int
        CMA-ES population size. Default: 24.
    sigma0 : float
        Initial CMA-ES step size. Default: 0.5.
    bounds : array-like, shape (n_params, 2)
        Parameter bounds [[lo, hi], ...].
    seed : int
        Random seed.
    """

    def __init__(
        self,
        reward_fn,
        cma_epochs: int = 30,
        grad_steps: int = 10,
        learning_rate: float = 0.01,
        population_size: int = 24,
        sigma0: float = 0.5,
        bounds=DEFAULT_BOUNDS,
        init_params=None,
        seed: int = 420,
    ):
        self._reward_fn = reward_fn
        self._cma_epochs = cma_epochs
        self._grad_steps = grad_steps
        self._learning_rate = learning_rate
        self._population_size = population_size
        self._sigma0 = sigma0
        self._bounds = np.array(bounds, dtype=np.float64)
        self._seed = seed

        # --- CMA-ES sub-optimizer ---
        self._cma = CMAESOptimizer(
            mean0=np.array(init_params) if init_params is not None else None,
            sigma0=sigma0,
            bounds=bounds,
            population_size=population_size,
            seed=seed,
        )

        # --- Adam sub-optimizer (optax) ---
        init_params = jnp.array(self._cma.mean)
        self._adam = optax.adam(learning_rate)
        self._adam_state = self._adam.init(init_params)
        self._grad_params = init_params

        # Gradient function: negate reward for minimization
        def neg_reward(x: jnp.ndarray) -> jnp.ndarray:
            return -reward_fn(x)

        self._grad_fn = jax.jit(jax.grad(neg_reward))
        self._key = jax.random.PRNGKey(seed)

        # --- Phase tracking ---
        self._epoch_counter = 0  # counts within current phase
        self._in_gradient_phase = False
        self._total_steps = 0

        # --- Best tracking ---
        self._best_params = init_params
        self._best_reward = -jnp.inf

    # ------------------------------------------------------------------
    # OnlineOptimizer interface
    # ------------------------------------------------------------------

    def ask(self, n_samples: int | None = None) -> jnp.ndarray:
        """Generate candidates.

        During CMA phase, returns a population from CMA-ES.
        During gradient phase, returns the single current gradient point.

        Parameters
        ----------
        n_samples : int or None
            Number of candidates (used in CMA phase only).

        Returns
        -------
        jnp.ndarray, shape (n, n_params)
        """
        if self._in_gradient_phase:
            # Gradient phase: return the single current point
            return self._grad_params.reshape(1, -1)
        else:
            # CMA-ES phase: return population
            return self._cma.ask(n_samples)

    def tell(self, params: jnp.ndarray, rewards: jnp.ndarray):
        """Update optimizer state.

        During CMA phase, updates CMA-ES with the population.
        During gradient phase, runs one Adam step (ignores provided params/rewards).

        Parameters
        ----------
        params : jnp.ndarray, shape (n, n_params)
            Evaluated parameter vectors.
        rewards : jnp.ndarray, shape (n,)
            Corresponding reward values (higher = better).
        """
        if self._in_gradient_phase:
            # --- Gradient phase: one Adam step ---
            grads = self._grad_fn(self._grad_params)

            updates, self._adam_state = self._adam.update(grads, self._adam_state)
            self._grad_params = optax.apply_updates(self._grad_params, updates)

            # Clip to bounds
            lo = jnp.array(self._bounds[:, 0])
            hi = jnp.array(self._bounds[:, 1])
            self._grad_params = jnp.clip(self._grad_params, lo, hi)

            # Evaluate and track best
            reward = float(self._reward_fn(self._grad_params))
            if reward > float(self._best_reward):
                self._best_reward = jnp.array(reward)
                self._best_params = self._grad_params.copy()

            self._epoch_counter += 1

            # Check phase transition: gradient -> CMA-ES
            if self._epoch_counter >= self._grad_steps:
                self._transition_to_cma_phase()

        else:
            # --- CMA-ES phase: normal ask/tell ---
            self._cma.tell(params, rewards)

            # Track best from CMA-ES
            rewards_np = np.asarray(rewards)
            best_idx = np.argmax(rewards_np)
            if float(rewards_np[best_idx]) > float(self._best_reward):
                self._best_reward = jnp.array(float(rewards_np[best_idx]))
                self._best_params = jnp.array(params[best_idx])

            self._epoch_counter += 1

            # Check phase transition: CMA-ES -> gradient
            if self._epoch_counter >= self._cma_epochs:
                self._transition_to_gradient_phase()

        self._total_steps += 1

    def get_best(self) -> jnp.ndarray:
        """Return current best parameter estimate.

        Returns
        -------
        jnp.ndarray, shape (n_params,)
        """
        return self._best_params

    @property
    def name(self) -> str:
        """Human-readable optimizer name."""
        return "Hybrid (CMA-ES + Adam)"

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def _transition_to_gradient_phase(self):
        """Switch from CMA-ES to gradient refinement."""
        self._in_gradient_phase = True
        self._epoch_counter = 0

        # Start gradient descent from CMA-ES mean
        self._grad_params = jnp.array(self._cma.mean)
        self._adam_state = self._adam.init(self._grad_params)

    def _transition_to_cma_phase(self):
        """Switch from gradient refinement back to CMA-ES."""
        self._in_gradient_phase = False
        self._epoch_counter = 0

        # Inject refined point as new CMA-ES mean via public API
        refined = np.asarray(self._grad_params, dtype=np.float64)
        self._cma.set_mean(refined)

        # Partially reset sigma to restore exploration (70% of initial)
        self._cma.set_sigma(self._sigma0 * 0.7)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def phase(self) -> str:
        """Current phase: 'cma' or 'gradient'."""
        return "gradient" if self._in_gradient_phase else "cma"

    @property
    def total_steps(self) -> int:
        """Total tell() calls across all phases."""
        return self._total_steps
