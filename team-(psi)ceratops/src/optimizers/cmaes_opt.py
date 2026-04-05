"""CMA-ES optimizer for cat qubit control.

Wraps the cmaes library (SepCMA / CMA) in the OnlineOptimizer interface.
Adapted from the challenge notebook pi-pulse CMA-ES example (cells 17/23).

Reference:
  Pack et al. "Benchmarking Optimization Algorithms for Automated Calibration
  of Quantum Devices." arXiv:2509.08555 (2025). — CMA-ES dominates.

NOTE: This module accesses private attributes (_sigma, _mean) of the cmaes
library's SepCMA/CMA classes for sigma floor enforcement, inflation, and
mean injection. These are not part of the public API and may break on
library updates. The cmaes version MUST be pinned exactly in requirements.txt.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from cmaes import CMA, SepCMA

from src.optimizers.base import OnlineOptimizer

# Default bounds for [g2_re, g2_im, eps_d_re, eps_d_im]
DEFAULT_BOUNDS = np.array(
    [
        [0.1, 5.0],  # g2_re
        [-2.0, 2.0],  # g2_im
        [0.5, 20.0],  # eps_d_re
        [-5.0, 5.0],  # eps_d_im
    ]
)


class CMAESOptimizer(OnlineOptimizer):
    """CMA-ES optimizer using SepCMA (diagonal covariance).

    Parameters
    ----------
    mean0 : array-like, shape (n_params,) or None
        Initial mean. Default: [1.0, 0.0, 4.0, 0.0].
    sigma0 : float
        Initial step size. Default: 0.5.
    bounds : array-like, shape (n_params, 2)
        Parameter bounds [[lo, hi], ...].
    population_size : int
        Number of candidates per generation. Default: 24.
    seed : int
        Random seed.
    sigma_floor : float or None
        If set, prevents sigma from collapsing below this value.
        Essential for tracking drift in non-stationary problems.
    use_full_cma : bool
        If True, use full CMA (learns correlations). Default False (SepCMA).
    """

    def __init__(
        self,
        mean0=None,
        sigma0: float = 0.5,
        bounds=DEFAULT_BOUNDS,
        population_size: int = 24,
        seed: int = 420,
        sigma_floor: float | None = None,
        use_full_cma: bool = False,
    ):
        if mean0 is None:
            mean0 = np.array([1.0, 0.0, 4.0, 0.0])
        self.population_size = population_size
        self.sigma_floor = sigma_floor
        self._use_full = use_full_cma

        cls = CMA if use_full_cma else SepCMA
        self._optimizer = cls(
            mean=np.array(mean0, dtype=np.float64),
            sigma=sigma0,
            bounds=np.array(bounds, dtype=np.float64),
            population_size=population_size,
            seed=seed,
        )

        self._best_params = np.array(mean0, dtype=np.float64)
        self._best_reward = -np.inf
        self._generation = 0

    # --- Private API wrappers (cmaes==0.11.1 internal access) ---
    # All private attribute access is isolated here. If cmaes changes its
    # internal representation, only these 3 methods need updating.

    def _get_sigma(self) -> float:
        """Read CMA-ES step size (wraps private API access)."""
        try:
            return float(self._optimizer._sigma)
        except AttributeError as err:
            raise RuntimeError(
                "cmaes private API changed. This code requires cmaes==0.11.1."
            ) from err

    def _set_sigma(self, value: float):
        """Write CMA-ES step size (wraps private API access)."""
        try:
            self._optimizer._sigma = float(value)
        except AttributeError as err:
            raise RuntimeError(
                "cmaes private API changed. This code requires cmaes==0.11.1."
            ) from err

    def _set_mean(self, value: np.ndarray):
        """Write CMA-ES distribution mean (wraps private API access)."""
        try:
            self._optimizer._mean = np.asarray(value, dtype=np.float64)
        except AttributeError as err:
            raise RuntimeError(
                "cmaes private API changed. This code requires cmaes==0.11.1."
            ) from err

    def ask(self, n_samples: int | None = None) -> jnp.ndarray:
        n = n_samples or self.population_size
        candidates = [self._optimizer.ask() for _ in range(n)]
        return jnp.array(candidates)

    def tell(self, params: jnp.ndarray, rewards: jnp.ndarray):
        # CMA-ES MINIMIZES, so negate rewards
        params_np = np.asarray(params)
        rewards_np = np.asarray(rewards)

        solutions = [
            (params_np[i], -float(rewards_np[i])) for i in range(len(params_np))
        ]
        self._optimizer.tell(solutions)

        # Track best
        best_idx = np.argmax(rewards_np)
        if rewards_np[best_idx] > self._best_reward:
            self._best_reward = float(rewards_np[best_idx])
            self._best_params = params_np[best_idx].copy()

        # Enforce sigma floor for drift tracking
        if self.sigma_floor is not None:
            self._set_sigma(max(self._get_sigma(), self.sigma_floor))

        self._generation += 1

    def get_best(self) -> jnp.ndarray:
        return jnp.array(self._best_params)

    @property
    def mean(self) -> jnp.ndarray:
        """Current distribution mean."""
        return jnp.array(self._optimizer.mean)

    @property
    def sigma(self) -> float:
        """Current step size."""
        return self._get_sigma()

    @property
    def generation(self) -> int:
        return self._generation

    def should_stop(self) -> bool:
        """Check CMA-ES convergence criteria."""
        return self._optimizer.should_stop()

    def inflate_sigma(self, factor: float = 2.0):
        """Manually inflate sigma to restore exploration (for drift tracking)."""
        self._set_sigma(self._get_sigma() * factor)

    def set_mean(self, new_mean: np.ndarray):
        """Set the distribution mean (e.g. after gradient refinement)."""
        self._set_mean(new_mean)

    def set_sigma(self, new_sigma: float):
        """Set the step size directly."""
        self._set_sigma(new_sigma)

    @property
    def name(self) -> str:
        variant = "CMA" if self._use_full else "SepCMA"
        return f"CMA-ES ({variant})"
