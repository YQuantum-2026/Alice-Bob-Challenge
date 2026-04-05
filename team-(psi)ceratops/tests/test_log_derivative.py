"""Tests for log-derivative lifetime estimation feature.

Validates:
  - _estimate_T_from_log_derivative recovers known T from exponential decay
  - Edge cases: small expectations, positive slope, all-equal values
  - Integration: build_reward with use_log_derivative=True returns finite results
    for proxy, parity, and enhanced_proxy reward types
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src.config import RewardConfig
from src.reward import _estimate_T_from_log_derivative, build_reward
from tests.conftest import FAST_PARAMS, X_DEFAULT

# ---------------------------------------------------------------------------
# Unit tests for _estimate_T_from_log_derivative
# ---------------------------------------------------------------------------


class TestEstimateTFromLogDerivative:
    """Unit tests for the log-derivative T estimation function."""

    def test_recovers_known_lifetime(self):
        """Perfect exponential decay A*exp(-t/T) should recover T."""
        T_true = 50.0
        A = 0.9
        t_points = jnp.array([10.0, 25.0, 50.0])
        expectations = A * jnp.exp(-t_points / T_true)

        T_est = _estimate_T_from_log_derivative(t_points, expectations)
        np.testing.assert_allclose(float(T_est), T_true, rtol=1e-4)

    def test_amplitude_invariance(self):
        """Different amplitudes should give the same T estimate."""
        T_true = 30.0
        t_points = jnp.array([5.0, 15.0, 30.0])

        T_a1 = _estimate_T_from_log_derivative(
            t_points, 1.0 * jnp.exp(-t_points / T_true)
        )
        T_a05 = _estimate_T_from_log_derivative(
            t_points, 0.5 * jnp.exp(-t_points / T_true)
        )
        T_a01 = _estimate_T_from_log_derivative(
            t_points, 0.1 * jnp.exp(-t_points / T_true)
        )

        np.testing.assert_allclose(float(T_a1), float(T_a05), rtol=1e-4)
        np.testing.assert_allclose(float(T_a1), float(T_a01), rtol=1e-4)

    def test_very_small_expectations(self):
        """Near-zero expectations should not produce NaN/Inf."""
        t_points = jnp.array([1.0, 2.0, 3.0])
        expectations = jnp.array([1e-10, 1e-12, 1e-14])

        T_est = _estimate_T_from_log_derivative(t_points, expectations)
        assert jnp.isfinite(T_est), f"Got {T_est} for tiny expectations"

    def test_positive_slope_clamps(self):
        """Growing expectations (positive slope) should still return finite T."""
        t_points = jnp.array([1.0, 2.0, 3.0])
        expectations = jnp.array([0.1, 0.2, 0.3])  # growing, not decaying

        T_est = _estimate_T_from_log_derivative(t_points, expectations)
        assert jnp.isfinite(T_est), f"Got {T_est} for positive slope"
        # Should be clamped to max (1e6) since slope is positive -> clamped negative
        assert float(T_est) > 0

    def test_all_equal_expectations(self):
        """Constant expectations (slope=0) should return finite T."""
        t_points = jnp.array([1.0, 2.0, 3.0])
        expectations = jnp.array([0.5, 0.5, 0.5])

        T_est = _estimate_T_from_log_derivative(t_points, expectations)
        assert jnp.isfinite(T_est), f"Got {T_est} for constant expectations"

    def test_negative_expectations_handled(self):
        """Negative expectations (e.g., parity) should use abs() safely."""
        T_true = 20.0
        t_points = jnp.array([5.0, 10.0, 20.0])
        expectations = -0.8 * jnp.exp(-t_points / T_true)  # negative values

        T_est = _estimate_T_from_log_derivative(t_points, expectations)
        assert jnp.isfinite(T_est)
        np.testing.assert_allclose(float(T_est), T_true, rtol=1e-3)

    def test_two_points_minimum(self):
        """With only 2 points, the fit is exact (no residual DOF)."""
        T_true = 100.0
        t_points = jnp.array([10.0, 50.0])
        expectations = jnp.exp(-t_points / T_true)

        T_est = _estimate_T_from_log_derivative(t_points, expectations)
        np.testing.assert_allclose(float(T_est), T_true, rtol=1e-4)

    def test_five_points_noisy(self):
        """With 5 noisy points, estimate should be reasonable."""
        rng = np.random.default_rng(42)
        T_true = 40.0
        t_points = jnp.array([8.0, 16.0, 24.0, 32.0, 40.0])
        clean = jnp.exp(-t_points / T_true)
        noise = 1.0 + 0.02 * jnp.array(rng.standard_normal(5))
        expectations = clean * noise

        T_est = _estimate_T_from_log_derivative(t_points, expectations)
        np.testing.assert_allclose(float(T_est), T_true, rtol=0.1)


# ---------------------------------------------------------------------------
# Integration tests: build_reward with use_log_derivative=True
# ---------------------------------------------------------------------------


class TestLogDerivativeIntegration:
    """Integration tests for log-derivative mode in reward functions."""

    @pytest.fixture
    def log_deriv_cfg(self):
        return RewardConfig(use_log_derivative=True, n_log_deriv_points=5)

    @pytest.mark.parametrize("reward_type", ["proxy", "parity", "enhanced_proxy"])
    def test_finite_output(self, reward_type, log_deriv_cfg):
        fn, _ = build_reward(reward_type, FAST_PARAMS, log_deriv_cfg)
        r = fn(X_DEFAULT)
        assert jnp.isfinite(r), f"{reward_type} with log_derivative returned {r}"

    @pytest.mark.parametrize("reward_type", ["proxy", "parity", "enhanced_proxy"])
    def test_vmap(self, reward_type, log_deriv_cfg):
        _, batched = build_reward(reward_type, FAST_PARAMS, log_deriv_cfg)
        xs = jnp.stack([X_DEFAULT, X_DEFAULT * 0.8])
        results = batched(xs)
        assert results.shape == (2,)
        assert jnp.all(jnp.isfinite(results))

    def test_log_deriv_vs_single_point_proxy(self):
        """Log-derivative and single-point should give different but both finite results."""
        cfg_single = RewardConfig(use_log_derivative=False)
        cfg_log = RewardConfig(use_log_derivative=True, n_log_deriv_points=5)

        fn_single, _ = build_reward("proxy", FAST_PARAMS, cfg_single)
        fn_log, _ = build_reward("proxy", FAST_PARAMS, cfg_log)

        r_single = fn_single(X_DEFAULT)
        r_log = fn_log(X_DEFAULT)

        assert jnp.isfinite(r_single)
        assert jnp.isfinite(r_log)
