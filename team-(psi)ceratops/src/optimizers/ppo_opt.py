"""PPO-Clip policy gradient optimizer for cat qubit control.

Ref: Schulman et al. "Proximal Policy Optimization Algorithms."
     arXiv:1707.06347 (2017).

Implements PPO-Clip with a factorized Gaussian policy pi(x) = N(mu, diag(sigma^2)).
Key differences from vanilla REINFORCE:
  - Stores old policy (mu_old, sigma_old) before each update
  - Computes importance sampling ratio r = pi_new(x) / pi_old(x)
  - Clips the ratio: L^CLIP = min(r*A, clip(r, 1-eps, 1+eps)*A)
  - Multiple gradient ascent epochs on the same batch (safe due to clipping)

This is a model-free approach: the optimizer treats the reward function as a
black box and does not require differentiability through the simulation.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from src.optimizers.base import OnlineOptimizer
from src.optimizers.cmaes_opt import DEFAULT_BOUNDS


def _log_prob_gaussian(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Compute log pi(x | mu, sigma) for a factorized Gaussian.

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Samples.
    mu : np.ndarray, shape (d,)
        Policy mean.
    sigma : np.ndarray, shape (d,)
        Policy standard deviation (must be > 0).

    Returns
    -------
    np.ndarray, shape (n,)
        Log-probability of each sample under the factorized Gaussian.
        log pi(x | mu, sigma) = sum_d [ -0.5*((x_d - mu_d)/sigma_d)^2
                                         - log(sigma_d) - 0.5*log(2*pi) ]
    """
    # shape (n, d)
    z = (x - mu[np.newaxis, :]) / sigma[np.newaxis, :]
    log_p_per_dim = (
        -0.5 * z**2 - np.log(sigma[np.newaxis, :]) - 0.5 * np.log(2.0 * np.pi)
    )
    return np.sum(log_p_per_dim, axis=1)  # shape (n,)


class PPOOptimizer(OnlineOptimizer):
    """PPO-Clip optimizer with factorized Gaussian policy.

    Implements the clipped surrogate objective from Schulman et al. (2017).
    At each tell() call, the optimizer runs multiple gradient ascent epochs
    on the same batch, using the clipped importance sampling ratio to prevent
    destructively large policy updates.

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
    clip_eps : float
        PPO clipping parameter epsilon. Default: 0.2.
    n_epochs : int
        Number of gradient ascent epochs per tell() call. Default: 3.
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
        clip_eps: float = 0.2,
        n_epochs: int = 3,
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
        self._clip_eps = clip_eps
        self._n_epochs = n_epochs

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
            b = np.array(DEFAULT_BOUNDS, dtype=np.float64)
            self._mu = (b[:, 0] + b[:, 1]) / 2.0
        self._log_sigma = np.full(n_params, np.log(sigma_init), dtype=np.float64)

        # --- Old policy snapshot (for importance sampling ratio) ---
        self._mu_old = self._mu.copy()
        self._log_sigma_old = self._log_sigma.copy()
        self._pending_ask = False

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

        Before sampling, snapshots the current policy as the "old" policy
        for subsequent importance sampling ratio computation in tell().

        Parameters
        ----------
        n_samples : int or None
            Number of candidates. If None, uses population_size.

        Returns
        -------
        jnp.ndarray, shape (n, n_params)
            Sampled candidate parameter vectors.
        """
        # Snapshot current policy as old before sampling
        if self._pending_ask:
            warnings.warn(
                "PPO ask() called again before tell() — overwriting old policy snapshot",
                stacklevel=2,
            )
        self._mu_old = self._mu.copy()
        self._log_sigma_old = self._log_sigma.copy()
        self._pending_ask = True

        n = n_samples or self._population_size
        sigma = np.exp(self._log_sigma)  # shape (n_params,)

        # Sample from N(mu, diag(sigma^2))
        noise = self._rng.standard_normal((n, self._n_params))
        candidates = self._mu[np.newaxis, :] + sigma[np.newaxis, :] * noise
        candidates = np.clip(candidates, self._bounds[:, 0], self._bounds[:, 1])

        return jnp.array(candidates)

    def tell(self, params: jnp.ndarray, rewards: jnp.ndarray):
        """Update policy using PPO-Clip surrogate objective.

        For each of n_epochs:
          1. Compute log pi_old(x) and log pi_new(x) under factorized Gaussian
          2. Importance sampling ratio r = exp(log pi_new - log pi_old)
          3. Clipped objective: L = min(r*A, clip(r, 1-eps, 1+eps)*A)
          4. Gradient ascent on L w.r.t. mu and log_sigma

        Parameters
        ----------
        params : jnp.ndarray, shape (n, n_params)
            Evaluated parameter vectors (the samples from ask()).
        rewards : jnp.ndarray, shape (n,)
            Corresponding reward values (higher = better).
        """
        params_np = np.asarray(params)
        rewards_np = np.asarray(rewards, dtype=np.float64)

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

        # --- Old policy log-probs (fixed reference) ---
        sigma_old = np.exp(self._log_sigma_old)
        log_pi_old = _log_prob_gaussian(
            params_np, self._mu_old, sigma_old
        )  # shape (n,)

        eps = self._clip_eps

        # --- Multiple PPO epochs on the same batch ---
        for _epoch in range(self._n_epochs):
            sigma = np.exp(self._log_sigma)  # shape (n_params,)
            sigma2 = sigma**2  # shape (n_params,)

            # Current policy log-probs
            log_pi_new = _log_prob_gaussian(params_np, self._mu, sigma)  # shape (n,)

            # Importance sampling ratio
            log_ratio = log_pi_new - log_pi_old  # shape (n,)
            ratio = np.exp(
                np.clip(log_ratio, -20.0, 20.0)
            )  # shape (n,), clip for numerical stability

            # Clipped ratio
            ratio_clipped = np.clip(ratio, 1.0 - eps, 1.0 + eps)  # shape (n,)

            # PPO-Clip surrogate: L = min(r*A, clip(r)*A)
            surr1 = ratio * advantages  # shape (n,)
            surr2 = ratio_clipped * advantages  # shape (n,)

            # When the clipped branch is active (surr2 < surr1), gradient
            # through the ratio is killed by the clip. Mask those samples
            # out entirely so only the unclipped branch contributes gradient.
            # Ref: Schulman et al. (2017), Eq. (7) -- gradient of min(·)
            # PPO-Clip gradient: mask-based approximation where clipped samples
            # contribute zero gradient (dropped), rather than the standard
            # formulation where they contribute the clipped surrogate gradient.
            # This is simpler and used in practice (e.g., CleanRL continuous PPO).
            # The effect is slightly more conservative updates.
            mask = (surr1 <= surr2).astype(
                np.float64
            )  # 1 where unclipped, 0 where clipped

            # --- Score function gradients w.r.t. current policy ---
            diff = params_np - self._mu[np.newaxis, :]  # shape (n, n_params)

            # d/d(mu) log N(x | mu, sigma^2) = (x - mu) / sigma^2
            grad_log_pi_mean = diff / sigma2[np.newaxis, :]  # shape (n, n_params)

            # d/d(log_sigma) log N(x | mu, sigma^2) = (x - mu)^2 / sigma^2 - 1
            grad_log_pi_log_sigma = (diff**2) / sigma2[
                np.newaxis, :
            ] - 1.0  # shape (n, n_params)

            # --- PPO-Clip policy gradient (Schulman et al. 2017, Eq. 7) ---
            # For unclipped samples: ∇_θ [r(θ)·A] = r(θ)·A·∇_θ log π_θ
            # For clipped samples: gradient is zero (clip creates a constant).
            # The ratio r(θ) = π_new/π_old is required because the data was
            # sampled from π_old, not π_new — it corrects for the distribution
            # mismatch that grows across multi-epoch updates within one tell().
            weights = mask * ratio * advantages  # shape (n,)

            weighted_mean = np.mean(
                weights[:, np.newaxis] * grad_log_pi_mean, axis=0
            )  # shape (n_params,)
            weighted_sigma = np.mean(
                weights[:, np.newaxis] * grad_log_pi_log_sigma, axis=0
            )  # shape (n_params,)

            # Gradient ascent
            self._mu += self._lr_mean * weighted_mean
            self._mu = np.clip(self._mu, self._bounds[:, 0], self._bounds[:, 1])
            self._log_sigma += self._lr_sigma * weighted_sigma

            # Clamp sigma
            self._log_sigma = np.clip(
                self._log_sigma,
                np.log(self._sigma_min),
                np.log(self._sigma_max),
            )

        # Entropy bonus (once per tell(), not per epoch — consistent with REINFORCE)
        self._log_sigma += self._lr_sigma * self._beta_entropy
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
        self._pending_ask = False

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
        return "PPO-Clip"

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
