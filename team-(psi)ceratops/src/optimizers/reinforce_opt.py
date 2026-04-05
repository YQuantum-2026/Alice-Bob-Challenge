"""REINFORCE policy gradient with factorized Gaussian policy for cat qubit control.

Ref: Sivak et al. (2025), arXiv:2511.08493, Sec. II — policy gradient on factorized Gaussian.

Implements a factorized Gaussian policy pi(x) = N(mu, diag(sigma^2)) where both
the mean (mu) and log-standard-deviation (log_sigma) are learnable parameters.
Candidates are sampled from the policy; rewards are used to compute policy gradient
estimates with a baseline for variance reduction and an entropy bonus
for exploration.

This is a model-free approach: the optimizer treats the reward function as a
black box and does not require differentiability through the simulation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from src.optimizers.base import OnlineOptimizer
from src.optimizers.cmaes_opt import DEFAULT_BOUNDS


class REINFORCEOptimizer(OnlineOptimizer):
    """REINFORCE optimizer with factorized Gaussian policy.

    Parameters
    ----------
    n_params : int
        Dimensionality of parameter space. Default: 4.
    population_size : int
        Number of candidates sampled per ask(). Default: 24.
    lr_mean : float
        Learning rate for the policy mean. Default: 0.05.
    lr_sigma : float
        Learning rate for the log-standard-deviation. Default: 0.01.
    beta_entropy : float
        Entropy bonus coefficient to prevent premature collapse. Default: 0.02.
    baseline_decay : float
        Exponential moving average decay for the reward baseline. Default: 0.9.
    sigma_init : float
        Initial standard deviation for all dimensions. Default: 0.5.
    sigma_min : float
        Minimum allowed sigma (prevents collapse). Default: 0.01.
    sigma_max : float
        Maximum allowed sigma (prevents explosion). Default: 2.0.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_params: int = 4,
        population_size: int = 24,
        lr_mean: float = 0.05,
        lr_sigma: float = 0.01,
        beta_entropy: float = 0.02,
        baseline_decay: float = 0.9,
        sigma_init: float = 0.5,
        sigma_min: float = 0.01,
        sigma_max: float = 2.0,
        seed: int = 420,
        init_params=None,
        bounds=None,
    ):
        self._n_params = n_params
        self._population_size = population_size
        self._lr_mean = lr_mean
        self._lr_sigma = lr_sigma
        self._beta_entropy = beta_entropy
        self._baseline_decay = baseline_decay
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

        self._rng = np.random.default_rng(seed)
        self._bounds = (
            np.array(bounds, dtype=np.float64)
            if bounds is not None
            else np.array(DEFAULT_BOUNDS, dtype=np.float64)
        )

        # --- Policy parameters ---
        if init_params is not None:
            self._mu = np.array(init_params, dtype=np.float64)
        else:
            self._mu = (self._bounds[:, 0] + self._bounds[:, 1]) / 2.0
        self._log_sigma = np.full(n_params, np.log(sigma_init), dtype=np.float64)

        # --- Baseline for variance reduction ---
        self._baseline = 0.0
        self._baseline_initialized = False

        # --- Best tracking ---
        self._best_params = self._mu.copy()
        self._best_reward = -np.inf

        self._step_count = 0

    # ------------------------------------------------------------------
    # OnlineOptimizer interface
    # ------------------------------------------------------------------

    def ask(self, n_samples: int | None = None) -> jnp.ndarray:
        """Sample N candidates from the current factorized Gaussian policy.

        Parameters
        ----------
        n_samples : int or None
            Number of candidates. If None, uses population_size.

        Returns
        -------
        jnp.ndarray, shape (n, n_params)
            Sampled candidate parameter vectors.
        """
        n = n_samples or self._population_size
        sigma = np.exp(self._log_sigma)  # shape (n_params,)

        # Sample from N(mu, diag(sigma^2))
        noise = self._rng.standard_normal((n, self._n_params))
        candidates = self._mu[np.newaxis, :] + sigma[np.newaxis, :] * noise
        candidates = np.clip(candidates, self._bounds[:, 0], self._bounds[:, 1])

        return jnp.array(candidates)

    def tell(self, params: jnp.ndarray, rewards: jnp.ndarray):
        """Update policy parameters using REINFORCE gradient estimate.

        Computes advantage = reward - baseline, then updates mu and log_sigma
        using the score function (grad log pi) weighted by advantages.

        Parameters
        ----------
        params : jnp.ndarray, shape (n, n_params)
            Evaluated parameter vectors (the samples from ask()).
        rewards : jnp.ndarray, shape (n,)
            Corresponding reward values (higher = better).
        """
        params_np = np.asarray(params)
        rewards_np = np.asarray(rewards, dtype=np.float64)

        sigma = np.exp(self._log_sigma)  # shape (n_params,)
        sigma2 = sigma**2  # shape (n_params,)

        # --- Update baseline (exponential moving average) ---
        mean_reward = float(np.mean(rewards_np))
        if not self._baseline_initialized:
            self._baseline = mean_reward
            self._baseline_initialized = True
        else:
            self._baseline = (
                self._baseline_decay * self._baseline
                + (1.0 - self._baseline_decay) * mean_reward
            )

        # --- Compute advantages ---
        advantages = rewards_np - self._baseline  # shape (n,)

        # --- Score function gradients (grad log pi) ---
        diff = params_np - self._mu[np.newaxis, :]  # shape (n, n_params)

        # d/d(mu) log N(x | mu, sigma^2) = (x - mu) / sigma^2
        grad_log_pi_mean = diff / sigma2[np.newaxis, :]  # shape (n, n_params)

        # d/d(log_sigma) log N(x | mu, sigma^2) = (x - mu)^2 / sigma^2 - 1
        grad_log_pi_log_sigma = (diff**2) / sigma2[
            np.newaxis, :
        ] - 1.0  # shape (n, n_params)

        # --- REINFORCE policy gradient updates ---
        # Weighted average of score * advantage
        weighted_mean = np.mean(
            advantages[:, np.newaxis] * grad_log_pi_mean, axis=0
        )  # shape (n_params,)
        weighted_sigma = np.mean(
            advantages[:, np.newaxis] * grad_log_pi_log_sigma, axis=0
        )  # shape (n_params,)

        # Gradient ascent on expected reward
        self._mu += self._lr_mean * weighted_mean
        self._mu = np.clip(self._mu, self._bounds[:, 0], self._bounds[:, 1])
        self._log_sigma += self._lr_sigma * weighted_sigma

        # Entropy bonus: encourages larger sigma (entropy of Gaussian ~ log(sigma))
        # d/d(log_sigma) H = 1, so the bonus simply pushes log_sigma up
        self._log_sigma += self._lr_sigma * self._beta_entropy

        # --- Clamp sigma ---
        self._log_sigma = np.clip(
            self._log_sigma,
            np.log(self._sigma_min),
            np.log(self._sigma_max),
        )

        # --- Track best observed ---
        best_idx = np.argmax(rewards_np)
        if rewards_np[best_idx] > self._best_reward:
            self._best_reward = float(rewards_np[best_idx])
            self._best_params = params_np[best_idx].copy()

        self._step_count += 1

    def get_best(self) -> jnp.ndarray:
        """Return the best observed parameter vector.

        Returns
        -------
        jnp.ndarray, shape (n_params,)
        """
        return jnp.array(self._best_params)

    @property
    def name(self) -> str:
        """Human-readable optimizer name."""
        return "REINFORCE (Policy Gradient)"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def mu(self) -> np.ndarray:
        """Current policy mean."""
        return self._mu.copy()

    @property
    def sigma(self) -> np.ndarray:
        """Current policy standard deviation."""
        return np.exp(self._log_sigma)

    @property
    def step_count(self) -> int:
        """Total tell() calls."""
        return self._step_count
