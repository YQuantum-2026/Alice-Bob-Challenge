"""Tests for src/optimizers/ — all optimizer approaches.

Validates ask/tell interface, convergence behavior, and special features
for each optimizer type.

References:
  CMA-ES: Pack et al. (2025), arXiv:2509.08555
  Hybrid: CMA-ES exploration + gradient refinement
  REINFORCE: Sivak et al. (2025), arXiv:2511.08493
  PPO-Clip: Schulman et al. (2017), arXiv:1707.06347
  Bayesian: GP surrogate + acquisition function
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src.reward import build_reward
from tests.conftest import FAST_PARAMS


def _get_reward_fn():
    """Build a fast proxy reward for testing."""
    fn, batched = build_reward("proxy", FAST_PARAMS)
    # Warm up JIT
    _ = fn(jnp.array([1.0, 0.0, 4.0, 0.0]))
    return fn, batched


class TestCMAES:
    """CMA-ES optimizer via SepCMA."""

    def test_ask_shape(self):
        from src.optimizers.cmaes_opt import CMAESOptimizer

        opt = CMAESOptimizer(population_size=6, seed=0)
        xs = opt.ask()
        assert xs.shape == (6, 4)

    def test_ask_tell_cycle(self):
        from src.optimizers.cmaes_opt import CMAESOptimizer

        opt = CMAESOptimizer(population_size=6, seed=0)
        xs = opt.ask()
        rewards = jnp.ones(6)  # dummy rewards
        opt.tell(xs, rewards)
        assert opt.generation == 1

    def test_mean_updates(self):
        from src.optimizers.cmaes_opt import CMAESOptimizer

        opt = CMAESOptimizer(population_size=6, seed=0)
        mean_before = np.array(opt.mean)
        xs = opt.ask()
        rewards = jnp.array([float(i) for i in range(6)])  # varied rewards
        opt.tell(xs, rewards)
        mean_after = np.array(opt.mean)
        assert not np.allclose(mean_before, mean_after), "Mean should update after tell"

    def test_name(self):
        from src.optimizers.cmaes_opt import CMAESOptimizer

        opt = CMAESOptimizer()
        assert "CMA-ES" in opt.name


class TestHybrid:
    """Hybrid CMA-ES + gradient refinement."""

    def test_ask_tell_cycle(self):
        fn, _ = _get_reward_fn()
        from src.optimizers.hybrid_opt import HybridOptimizer

        opt = HybridOptimizer(
            reward_fn=fn, cma_epochs=3, grad_steps=2, population_size=4, seed=0
        )
        xs = opt.ask()
        assert xs.shape[1] == 4
        rewards = jnp.zeros(xs.shape[0])
        opt.tell(xs, rewards)

    def test_name(self):
        fn, _ = _get_reward_fn()
        from src.optimizers.hybrid_opt import HybridOptimizer

        opt = HybridOptimizer(reward_fn=fn)
        assert "hybrid" in opt.name.lower() or "Hybrid" in opt.name


class TestREINFORCE:
    """REINFORCE policy gradient optimizer (formerly mislabeled as PPO).
    Ref: Sivak et al. (2025), arXiv:2511.08493."""

    def test_ask_shape(self):
        from src.optimizers.reinforce_opt import REINFORCEOptimizer

        opt = REINFORCEOptimizer(n_params=4, population_size=8, seed=0)
        xs = opt.ask()
        assert xs.shape == (8, 4)

    def test_ask_tell_cycle(self):
        from src.optimizers.reinforce_opt import REINFORCEOptimizer

        rng = np.random.default_rng(42)
        opt = REINFORCEOptimizer(n_params=4, population_size=8, seed=0)
        xs = opt.ask()
        rewards = jnp.array(rng.standard_normal(8))
        opt.tell(xs, rewards)
        # Sigma should stay positive
        assert np.all(np.exp(opt._log_sigma) > 0)

    def test_sigma_stays_positive(self):
        from src.optimizers.reinforce_opt import REINFORCEOptimizer

        rng = np.random.default_rng(42)
        opt = REINFORCEOptimizer(n_params=4, population_size=8, seed=0)
        for _ in range(10):
            xs = opt.ask()
            rewards = jnp.array(rng.standard_normal(8))
            opt.tell(xs, rewards)
        sigma = np.exp(opt._log_sigma)
        assert np.all(sigma > 0), f"Sigma went non-positive: {sigma}"

    def test_name(self):
        from src.optimizers.reinforce_opt import REINFORCEOptimizer

        opt = REINFORCEOptimizer()
        assert "REINFORCE" in opt.name


class TestPPO:
    """PPO-Clip optimizer with importance sampling and clipping.
    Ref: Schulman et al. (2017), arXiv:1707.06347."""

    def test_ask_shape(self):
        from src.optimizers.ppo_opt import PPOOptimizer

        opt = PPOOptimizer(n_params=4, population_size=8, seed=0)
        xs = opt.ask()
        assert xs.shape == (8, 4)

    def test_ask_tell_cycle(self):
        from src.optimizers.ppo_opt import PPOOptimizer

        rng = np.random.default_rng(42)
        opt = PPOOptimizer(n_params=4, population_size=8, seed=0)
        xs = opt.ask()
        rewards = jnp.array(rng.standard_normal(8))
        opt.tell(xs, rewards)
        # Sigma should stay positive
        assert np.all(np.exp(opt._log_sigma) > 0)

    def test_sigma_stays_positive(self):
        from src.optimizers.ppo_opt import PPOOptimizer

        rng = np.random.default_rng(42)
        opt = PPOOptimizer(n_params=4, population_size=8, seed=0)
        for _ in range(10):
            xs = opt.ask()
            rewards = jnp.array(rng.standard_normal(8))
            opt.tell(xs, rewards)
        sigma = np.exp(opt._log_sigma)
        assert np.all(sigma > 0), f"Sigma went non-positive: {sigma}"

    def test_name(self):
        from src.optimizers.ppo_opt import PPOOptimizer

        opt = PPOOptimizer()
        assert "PPO" in opt.name

    def test_multiple_epochs(self):
        """PPO should run multiple update epochs per tell() call."""
        from src.optimizers.ppo_opt import PPOOptimizer

        rng = np.random.default_rng(42)
        # n_epochs=1 vs n_epochs=5 should produce different updates
        opt1 = PPOOptimizer(n_params=4, population_size=8, n_epochs=1, seed=0)
        opt5 = PPOOptimizer(n_params=4, population_size=8, n_epochs=5, seed=0)

        xs1 = opt1.ask()
        xs5 = opt5.ask()
        # Same seed, same samples
        rewards = jnp.array(rng.standard_normal(8))
        opt1.tell(xs1, rewards)
        opt5.tell(xs5, rewards)

        # After tell, the means should differ because more epochs = more updates
        assert not np.allclose(opt1.mu, opt5.mu), (
            "Different n_epochs should produce different mu updates"
        )

    def test_clipping_bounds(self):
        """PPO clip_eps should limit the policy ratio."""
        from src.optimizers.ppo_opt import PPOOptimizer

        # Use very small clip_eps to maximize clipping effect
        opt_tight = PPOOptimizer(n_params=4, population_size=16, clip_eps=0.01, seed=0)
        opt_loose = PPOOptimizer(n_params=4, population_size=16, clip_eps=10.0, seed=0)

        rng = np.random.default_rng(42)
        xs_t = opt_tight.ask()
        xs_l = opt_loose.ask()
        rewards = jnp.array(rng.standard_normal(16))
        opt_tight.tell(xs_t, rewards)
        opt_loose.tell(xs_l, rewards)

        # With very tight clipping, mu should change less than with loose clipping
        init_mu = np.array([2.55, 0.0, 10.25, 0.0])
        mu_change_tight = np.linalg.norm(opt_tight.mu - init_mu)
        mu_change_loose = np.linalg.norm(opt_loose.mu - init_mu)
        assert mu_change_tight <= mu_change_loose, (
            f"Tighter clip_eps should produce smaller updates: "
            f"tight={mu_change_tight:.6f}, loose={mu_change_loose:.6f}"
        )


class TestBayesian:
    """Bayesian optimization (GP surrogate)."""

    def test_ask_tell_cycle(self):
        try:
            from src.optimizers.bayesian_opt import BayesianOptimizer
        except ImportError:
            pytest.skip("scikit-optimize not installed")

        opt = BayesianOptimizer(n_initial=3, seed=0)
        xs = opt.ask(3)
        assert xs.shape == (3, 4)
        rewards = jnp.array([0.1, 0.2, 0.3])
        opt.tell(xs, rewards)

    def test_name(self):
        try:
            from src.optimizers.bayesian_opt import BayesianOptimizer
        except ImportError:
            pytest.skip("scikit-optimize not installed")
        opt = BayesianOptimizer()
        assert "Bayesian" in opt.name or "GP" in opt.name


class TestREINFORCEBoundsClipping:
    """REINFORCE ask() must clip candidates to physical bounds."""

    def test_candidates_within_bounds(self):
        from src.optimizers.cmaes_opt import DEFAULT_BOUNDS
        from src.optimizers.reinforce_opt import REINFORCEOptimizer

        opt = REINFORCEOptimizer(n_params=4, sigma_init=10.0, seed=0)
        candidates = opt.ask(100)
        for i in range(4):
            assert float(jnp.min(candidates[:, i])) >= DEFAULT_BOUNDS[i, 0] - 1e-6
            assert float(jnp.max(candidates[:, i])) <= DEFAULT_BOUNDS[i, 1] + 1e-6


class TestPPODoubleAsk:
    """PPO should warn if ask() called twice without tell()."""

    def test_double_ask_warns(self):
        from src.optimizers.ppo_opt import PPOOptimizer

        opt = PPOOptimizer(n_params=4, population_size=8, seed=420)
        opt.ask()
        with pytest.warns(UserWarning, match="overwriting old policy"):
            opt.ask()

    def test_single_ask_no_warning(self):
        import warnings

        from src.optimizers.ppo_opt import PPOOptimizer

        opt = PPOOptimizer(n_params=4, population_size=8, seed=420)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            opt.ask()  # First ask should NOT warn


class TestPPOBoundsClipping:
    """PPO-Clip ask() must clip candidates to physical bounds."""

    def test_candidates_within_bounds(self):
        from src.optimizers.cmaes_opt import DEFAULT_BOUNDS
        from src.optimizers.ppo_opt import PPOOptimizer

        opt = PPOOptimizer(n_params=4, sigma_init=10.0, seed=0)
        candidates = opt.ask(100)
        for i in range(4):
            assert float(jnp.min(candidates[:, i])) >= DEFAULT_BOUNDS[i, 0] - 1e-6
            assert float(jnp.max(candidates[:, i])) <= DEFAULT_BOUNDS[i, 1] + 1e-6


class TestPPOvsREINFORCE:
    """PPO-Clip and REINFORCE should produce different updates given same input.

    PPO uses importance sampling ratio clipping and multiple epochs;
    REINFORCE uses a single vanilla score-function gradient step.
    """

    def test_different_updates(self):
        from src.optimizers.ppo_opt import PPOOptimizer
        from src.optimizers.reinforce_opt import REINFORCEOptimizer

        rng = np.random.default_rng(123)

        reinforce = REINFORCEOptimizer(
            n_params=4, population_size=16, lr_mean=0.05, lr_sigma=0.01, seed=0
        )
        ppo = PPOOptimizer(
            n_params=4,
            population_size=16,
            lr_mean=0.05,
            lr_sigma=0.01,
            clip_eps=0.2,
            n_epochs=3,
            seed=0,
        )

        # Both start from same initial state
        assert np.allclose(reinforce.mu, ppo.mu), "Initial mu should match"
        assert np.allclose(reinforce.sigma, ppo.sigma), "Initial sigma should match"

        # Same samples and rewards for both
        xs_r = reinforce.ask()
        xs_p = ppo.ask()
        # Same seed -> same samples
        assert np.allclose(xs_r, xs_p, atol=1e-10), (
            "Samples should match with same seed"
        )

        rewards = jnp.array(rng.standard_normal(16))
        reinforce.tell(xs_r, rewards)
        ppo.tell(xs_p, rewards)

        # After one tell(), PPO (3 epochs + clipping) and REINFORCE (1 epoch, no clipping)
        # should produce different policy parameters
        assert not np.allclose(reinforce.mu, ppo.mu, atol=1e-10), (
            "PPO and REINFORCE should produce different mu after tell()"
        )

    def test_different_names(self):
        from src.optimizers.ppo_opt import PPOOptimizer
        from src.optimizers.reinforce_opt import REINFORCEOptimizer

        assert REINFORCEOptimizer().name != PPOOptimizer().name
