"""Abstract base class for online optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp


class OnlineOptimizer(ABC):
    """Common interface for online optimizers.

    All optimizers follow the ask-tell pattern:
      1. ask() → candidate parameter vectors
      2. evaluate rewards externally
      3. tell(params, rewards) → update optimizer state
    """

    @abstractmethod
    def ask(self, n_samples: int | None = None) -> jnp.ndarray:
        """Generate candidate parameter vectors.

        Parameters
        ----------
        n_samples : int or None
            Number of candidates. If None, use optimizer's default.

        Returns
        -------
        jnp.ndarray, shape (n_samples, n_params)
            Candidate parameter vectors.
        """

    @abstractmethod
    def tell(self, params: jnp.ndarray, rewards: jnp.ndarray):
        """Update optimizer with evaluated (params, rewards) pairs.

        Parameters
        ----------
        params : jnp.ndarray, shape (n, n_params)
            Evaluated parameter vectors.
        rewards : jnp.ndarray, shape (n,)
            Corresponding reward values (higher = better).
        """

    @abstractmethod
    def get_best(self) -> jnp.ndarray:
        """Return current best parameter estimate.

        Returns
        -------
        jnp.ndarray, shape (n_params,)
        """

    @property
    def best_reward(self) -> float:
        """Best reward observed so far."""
        return float(getattr(self, "_best_reward", float("-inf")))

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable optimizer name."""
