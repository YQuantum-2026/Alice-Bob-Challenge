"""Tests for src/benchmark.py — benchmark runner integration tests.

These are integration tests that run small optimization loops to verify
the full pipeline works end-to-end.
"""

import pytest

from src.cat_qubit import CatQubitParams
from src.config import (
    BenchmarkConfig,
    OptimizerConfig,
    RewardConfig,
    RunConfig,
    get_config,
)

# Tiny config for fast tests
TINY_PARAMS = CatQubitParams(na=6, nb=2, kappa_b=10.0, kappa_a=1.0)
TINY_OPT = OptimizerConfig(
    population_size=4, n_epochs=3, sigma0=0.5, learning_rate=0.01, seed=42
)
TINY_REWARD = RewardConfig(
    t_probe_z=10.0, t_probe_x=0.1, full_eval_interval=100, n_target=2.0, t_steady=2.0
)
TINY_CFG = RunConfig(
    name="tiny",
    cat_params=TINY_PARAMS,
    optimizer=TINY_OPT,
    reward=TINY_REWARD,
    benchmark=BenchmarkConfig(rewards=["proxy"], optimizers=["cmaes"], drifts=["none"]),
)


class TestBuildOptimizer:
    def test_cmaes(self):
        from src.benchmark import build_optimizer
        from src.reward import build_reward

        fn, _ = build_reward("proxy", TINY_PARAMS)
        opt = build_optimizer("cmaes", fn, TINY_CFG)
        assert opt.name  # has a name

    def test_unknown_raises(self):
        from src.benchmark import build_optimizer

        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer("nonexistent", lambda x: 0, TINY_CFG)


class TestBuildReward:
    @pytest.mark.parametrize("reward_type", ["proxy", "photon", "fidelity", "parity"])
    def test_build_reward_factory(self, reward_type):
        from src.reward import build_reward

        fn, batched = build_reward(reward_type, TINY_PARAMS)
        assert callable(fn)
        assert callable(batched)


class TestRunSingle:
    def test_cmaes_proxy_no_drift(self):
        """Smallest possible end-to-end test."""
        from src.benchmark import run_single

        result = run_single("proxy", "cmaes", "none", TINY_CFG, verbose=False)
        assert result.n_epochs == 3
        assert len(result.reward_history) == 3
        assert len(result.param_history) == 3
        assert result.wall_time > 0

    def test_result_has_correct_labels(self):
        from src.benchmark import run_single

        result = run_single("proxy", "cmaes", "none", TINY_CFG, verbose=False)
        assert result.reward_type == "proxy"
        assert result.optimizer_type == "cmaes"
        assert result.drift_type == "none"


class TestGetConfig:
    def test_local_profile(self):
        cfg = get_config("local")
        assert cfg.name == "local"
        assert cfg.cat_params.na == 10

    def test_deep_copy(self):
        """get_config should return a deep copy — mutations don't affect preset."""
        cfg1 = get_config("local")
        cfg1.optimizer.n_epochs = 9999
        cfg2 = get_config("local")
        assert cfg2.optimizer.n_epochs != 9999

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_config("nonexistent")
