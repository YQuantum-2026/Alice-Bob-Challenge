"""Integration tests for the full optimization pipeline.

Validates that each optimizer runs end-to-end with and without drift,
producing well-formed RunResult objects with finite rewards.

These tests exercise the complete stack: config -> reward -> drift -> optimizer
-> run_single() -> RunResult. They use a TINY Hilbert space (na=6, nb=3,
dim=18) and minimal epoch count to keep wall-clock time manageable while
still exercising all code paths.

Tier: 2 (integration) — each test runs a full optimization loop.
Mark: @pytest.mark.slow — skip with `pytest -m 'not slow'`.
"""

import numpy as np
import pytest

from src.benchmark import RunResult, run_single
from src.cat_qubit import CatQubitParams
from src.config import get_config

# ---------------------------------------------------------------------------
# Shared tiny configuration
# ---------------------------------------------------------------------------


def _tiny_config():
    """Build a minimal RunConfig for fast integration testing.

    Hilbert space: na=6, nb=3 (dim=18).
    Optimizer: 10 epochs, population 4.
    Validation: disabled (full_eval_interval > n_epochs).

    Returns
    -------
    RunConfig
        Deep-copied LOCAL preset with overrides for speed.
    """
    cfg = get_config("local")
    cfg.cat_params = CatQubitParams(na=6, nb=3, kappa_b=10.0, kappa_a=1.0)
    cfg.optimizer.n_epochs = 10
    cfg.optimizer.population_size = 4
    cfg.optimizer.hybrid_cma_epochs = 3
    cfg.optimizer.hybrid_grad_steps = 2
    cfg.optimizer.bayesian_n_initial = 2
    cfg.reward.full_eval_interval = 100  # disable periodic validation
    return cfg


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_basic_result(result: RunResult, cfg) -> None:
    """Check invariants that hold for every successful optimization run.

    Parameters
    ----------
    result : RunResult
        Output of run_single().
    cfg : RunConfig
        Configuration used for the run.
    """
    n = cfg.optimizer.n_epochs

    # History lengths match epoch count
    assert len(result.reward_history) == n, (
        f"Expected {n} reward entries, got {len(result.reward_history)}"
    )
    assert len(result.reward_std_history) == n
    assert len(result.param_history) == n
    assert len(result.drift_offset_history) == n

    # All rewards are finite (no NaN or Inf)
    rewards = np.array(result.reward_history)
    assert np.all(np.isfinite(rewards)), (
        f"Non-finite rewards found: {rewards[~np.isfinite(rewards)]}"
    )

    # Timing and epoch count
    assert result.wall_time > 0, "wall_time must be positive"
    assert result.n_epochs == n


def _assert_drift_active(result: RunResult) -> None:
    """Check that at least some drift offsets are non-zero.

    For drift scenarios that produce control offsets (amplitude, step, etc.),
    at least one epoch should have a non-zero g2_re offset.

    Parameters
    ----------
    result : RunResult
        Output of run_single() with an active drift.
    """
    offsets = result.drift_offset_history
    assert len(offsets) > 0, "drift_offset_history should not be empty"

    # Collect all g2_re offsets; at least one should be non-zero
    g2_re_values = [float(d["g2_re"]) for d in offsets]
    assert any(abs(v) > 1e-12 for v in g2_re_values), (
        f"Expected non-zero g2_re drift offsets, got all near zero: {g2_re_values}"
    )


# ---------------------------------------------------------------------------
# No-drift tests — one per optimizer
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNoDrift:
    """Each optimizer runs to completion without drift perturbations."""

    def test_cmaes_no_drift(self):
        cfg = _tiny_config()
        result = run_single("proxy", "cmaes", "none", cfg, verbose=False)
        _assert_basic_result(result, cfg)
        assert result.optimizer_type == "cmaes"
        assert result.drift_type == "none"

    def test_hybrid_no_drift(self):
        cfg = _tiny_config()
        result = run_single("proxy", "hybrid", "none", cfg, verbose=False)
        _assert_basic_result(result, cfg)
        assert result.optimizer_type == "hybrid"
        assert result.drift_type == "none"

    def test_reinforce_no_drift(self):
        cfg = _tiny_config()
        result = run_single("proxy", "reinforce", "none", cfg, verbose=False)
        _assert_basic_result(result, cfg)
        assert result.optimizer_type == "reinforce"
        assert result.drift_type == "none"

    def test_ppo_no_drift(self):
        cfg = _tiny_config()
        result = run_single("proxy", "ppo", "none", cfg, verbose=False)
        _assert_basic_result(result, cfg)
        assert result.optimizer_type == "ppo"
        assert result.drift_type == "none"

    def test_bayesian_no_drift(self):
        cfg = _tiny_config()
        result = run_single("proxy", "bayesian", "none", cfg, verbose=False)
        _assert_basic_result(result, cfg)
        assert result.optimizer_type == "bayesian"
        assert result.drift_type == "none"


# ---------------------------------------------------------------------------
# With-drift tests — one per optimizer
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestWithDrift:
    """Each optimizer runs to completion under drift perturbations.

    Uses "amplitude_slow" drift for most tests because it produces non-zero
    control offsets starting from epoch 1 (sinusoidal with freq=0.005).
    The "step" drift has step_epoch=100 by default, which is past our
    10-epoch window, so it would not trigger.
    """

    def test_cmaes_amplitude_drift(self):
        cfg = _tiny_config()
        result = run_single("proxy", "cmaes", "amplitude_slow", cfg, verbose=False)
        _assert_basic_result(result, cfg)
        _assert_drift_active(result)
        assert result.optimizer_type == "cmaes"
        assert result.drift_type == "amplitude_slow"

    def test_hybrid_amplitude_drift(self):
        """Hybrid optimizer under amplitude drift.

        Note: originally planned as step drift, but StepDrift defaults to
        step_epoch=100 which is past the 10-epoch test window. Using
        amplitude_slow instead to ensure drift offsets are non-zero.
        """
        cfg = _tiny_config()
        result = run_single("proxy", "hybrid", "amplitude_slow", cfg, verbose=False)
        _assert_basic_result(result, cfg)
        _assert_drift_active(result)
        assert result.optimizer_type == "hybrid"
        assert result.drift_type == "amplitude_slow"

    def test_reinforce_amplitude_drift(self):
        cfg = _tiny_config()
        result = run_single("proxy", "reinforce", "amplitude_slow", cfg, verbose=False)
        _assert_basic_result(result, cfg)
        _assert_drift_active(result)
        assert result.optimizer_type == "reinforce"
        assert result.drift_type == "amplitude_slow"

    def test_ppo_amplitude_drift(self):
        cfg = _tiny_config()
        result = run_single("proxy", "ppo", "amplitude_slow", cfg, verbose=False)
        _assert_basic_result(result, cfg)
        _assert_drift_active(result)
        assert result.optimizer_type == "ppo"
        assert result.drift_type == "amplitude_slow"

    def test_bayesian_amplitude_drift(self):
        cfg = _tiny_config()
        result = run_single("proxy", "bayesian", "amplitude_slow", cfg, verbose=False)
        _assert_basic_result(result, cfg)
        _assert_drift_active(result)
        assert result.optimizer_type == "bayesian"
        assert result.drift_type == "amplitude_slow"


# ---------------------------------------------------------------------------
# Benchmark structure test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBenchmarkStructure:
    """Validates that RunResult fields are populated correctly."""

    def test_benchmark_single_combo(self):
        """run_single returns a well-formed RunResult with all metadata."""
        cfg = _tiny_config()
        result = run_single("proxy", "cmaes", "none", cfg, verbose=False)

        # Type check
        assert isinstance(result, RunResult)

        # Metadata labels
        assert result.reward_type == "proxy"
        assert result.optimizer_type == "cmaes"
        assert result.drift_type == "none"
        assert result.config_name == cfg.name

        # Label property
        assert result.label == "cmaes/proxy/none"

        # History types
        assert isinstance(result.reward_history, list)
        assert isinstance(result.reward_std_history, list)
        assert isinstance(result.param_history, list)
        assert isinstance(result.drift_offset_history, list)
        assert isinstance(result.validation_history, list)

        # Param history entries are arrays with correct shape (4 control params)
        for params in result.param_history:
            arr = np.asarray(params)
            assert arr.shape == (4,), f"Expected param shape (4,), got {arr.shape}"
            assert np.all(np.isfinite(arr)), "Param values must be finite"

        # Drift offset history entries are dicts with expected keys
        expected_drift_keys = {
            "g2_re",
            "g2_im",
            "eps_d_re",
            "eps_d_im",
            "detuning",
            "kerr",
            "snr_noise",
        }
        for offset_dict in result.drift_offset_history:
            assert set(offset_dict.keys()) == expected_drift_keys, (
                f"Unexpected drift keys: {set(offset_dict.keys())}"
            )
            for key, val in offset_dict.items():
                assert np.isfinite(val), f"Non-finite drift offset: {key}={val}"

        # Validation history should be empty (full_eval_interval=100 > 10 epochs)
        assert result.validation_history == []

        # Reward std should be non-negative
        for std_val in result.reward_std_history:
            assert std_val >= 0, f"Reward std must be non-negative, got {std_val}"
