"""Bayesian optimization for cat qubit control.

Bayesian optimizer using GP surrogate + acquisition function. O(n^3) per step but most sample-efficient.

Uses a Gaussian Process surrogate model to build a posterior over the reward
landscape, then selects candidates by optimizing an acquisition function
(GP-Hedge by default, which adaptively selects among LCB, EI, and PI).

Best suited for settings where each reward evaluation is expensive (e.g.,
real hardware) and we want to minimize the total number of evaluations.

Reference:
  Snoek, J., Larochelle, H., Adams, R. P. "Practical Bayesian Optimization
  of Machine Learning Algorithms." NeurIPS (2012).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from src.optimizers.base import OnlineOptimizer
from src.optimizers.cmaes_opt import DEFAULT_BOUNDS

try:
    from skopt import Optimizer as SkoptOptimizer
    from skopt.space import Real
except ImportError as e:
    raise ImportError(
        "BayesianOptimizer requires scikit-optimize. "
        "Install it with: pip install scikit-optimize"
    ) from e


class BayesianOptimizer(OnlineOptimizer):
    """Bayesian optimizer using Gaussian Process surrogate (via scikit-optimize).

    Parameters
    ----------
    bounds : array-like, shape (n_params, 2)
        Parameter bounds [[lo, hi], ...].
    n_initial : int
        Number of initial random evaluations before GP model kicks in. Default: 10.
    acq_func : str
        Acquisition function. Options: 'gp_hedge' (default, adaptive),
        'LCB', 'EI', 'PI'.
    noise_level : float
        Estimated noise level in the reward. Default: 0.1.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        bounds=DEFAULT_BOUNDS,
        n_initial: int = 10,
        acq_func: str = "gp_hedge",
        noise_level: float = 0.1,
        seed: int = 420,
        batch_size: int = 1,
    ):
        bounds_np = np.array(bounds, dtype=np.float64)
        self._n_params = bounds_np.shape[0]
        # batch_size=1 is intentional: sequential BO evaluates one candidate,
        # updates the GP surrogate, then picks the next candidate. This is standard
        # practice and optimal for expensive black-box functions. The benchmark loop
        # handles batch_size=1 correctly (evaluates as a batch of 1).
        self.batch_size = batch_size

        # Convert bounds to skopt Real dimensions
        dimensions = [
            Real(float(bounds_np[i, 0]), float(bounds_np[i, 1]))
            for i in range(self._n_params)
        ]

        self._skopt = SkoptOptimizer(
            dimensions=dimensions,
            acq_func=acq_func,
            n_initial_points=n_initial,
            random_state=seed,
            acq_func_kwargs={"noise_level": noise_level},
        )

        # --- History for best tracking ---
        self._best_params: np.ndarray | None = None
        self._best_reward = -np.inf
        self._step_count = 0

    # ------------------------------------------------------------------
    # OnlineOptimizer interface
    # ------------------------------------------------------------------

    def ask(self, n_samples: int | None = None) -> jnp.ndarray:
        """Generate candidates from the GP surrogate / acquisition function.

        Since skopt.Optimizer.ask() returns one point at a time, we call it
        n_samples times to build a batch.

        Parameters
        ----------
        n_samples : int or None
            Number of candidates. Default: ``self.batch_size`` (1 by default,
            since sequential BO is standard for expensive evaluations).

        Returns
        -------
        jnp.ndarray, shape (n_samples, n_params)
            Candidate parameter vectors.
        """
        n = n_samples or self.batch_size
        if n == 1:
            candidates = [self._skopt.ask()]
        else:
            # ask(n_points=N) uses kriging-believer strategy for diverse batch
            candidates = self._skopt.ask(n_points=n)
        return jnp.array(candidates)

    def tell(self, params: jnp.ndarray, rewards: jnp.ndarray):
        """Feed evaluated (params, rewards) pairs to the GP model.

        skopt minimizes, so we negate rewards before telling.

        Parameters
        ----------
        params : jnp.ndarray, shape (n, n_params)
            Evaluated parameter vectors.
        rewards : jnp.ndarray, shape (n,)
            Corresponding reward values (higher = better).
        """
        params_np = np.asarray(params)
        rewards_np = np.asarray(rewards, dtype=np.float64)

        # Feed each (point, negated_reward) pair to skopt
        for i in range(len(params_np)):
            x = params_np[i].tolist()
            y = -float(rewards_np[i])  # skopt minimizes
            self._skopt.tell(x, y)

        # Track best observed
        best_idx = np.argmax(rewards_np)
        if rewards_np[best_idx] > self._best_reward:
            self._best_reward = float(rewards_np[best_idx])
            self._best_params = params_np[best_idx].copy()

        self._step_count += 1

    def get_best(self) -> jnp.ndarray:
        """Return the point with highest observed reward.

        Returns
        -------
        jnp.ndarray, shape (n_params,)
        """
        if self._best_params is None:
            raise RuntimeError("No observations yet. Call tell() first.")
        return jnp.array(self._best_params)

    @property
    def name(self) -> str:
        """Human-readable optimizer name."""
        return "Bayesian (GP-UCB)"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        """Total number of observations fed to the GP."""
        return len(self._skopt.yi) if hasattr(self._skopt, "yi") else 0

    @property
    def step_count(self) -> int:
        """Total tell() calls."""
        return self._step_count
