"""Tests for src/gates.py — single-qubit gate extension.

Validates Zeno gate Hamiltonian, gate rewards, and fidelity.

Reference:
  iQuHack-2025 Task 1.3 — Zeno gate on dissipative cat qubit.
"""

import jax.numpy as jnp

from src.cat_qubit import build_operators
from src.config import GateConfig, RewardConfig
from src.gates import (
    GATE_BOUNDS,
    GATE_MEAN,
    build_gate_hamiltonian,
    build_gate_reward,
)
from tests.conftest import FAST_PARAMS

X_GATE = jnp.array([1.0, 0.0, 4.0, 0.0, 0.2])


class TestGateHamiltonian:
    def test_hermiticity(self):
        """Gate Hamiltonian must be Hermitian."""
        a, b = build_operators(FAST_PARAMS)
        H = build_gate_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0, 0.2)
        H_dense = H.to_jax()
        diff = float(jnp.max(jnp.abs(H_dense - jnp.conj(H_dense.T))))
        assert diff < 1e-10, f"H not Hermitian: max diff = {diff}"

    def test_reduces_to_standard_at_eps_z_zero(self):
        """With epsilon_z=0, gate H = standard H."""
        from src.cat_qubit import build_hamiltonian

        a, b = build_operators(FAST_PARAMS)
        H_std = build_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0)
        H_gate = build_gate_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0, 0.0)
        diff = float(jnp.max(jnp.abs(H_std.to_jax() - H_gate.to_jax())))
        assert diff < 1e-10, f"eps_z=0 should give standard H, diff={diff}"

    def test_shape(self):
        a, b = build_operators(FAST_PARAMS)
        H = build_gate_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0, 0.2)
        dim = FAST_PARAMS.na * FAST_PARAMS.nb
        assert H.shape == (dim, dim)


class TestGateBounds:
    def test_shape(self):
        assert GATE_BOUNDS.shape == (5, 2)

    def test_mean_within_bounds(self):
        for i in range(5):
            assert GATE_BOUNDS[i, 0] <= GATE_MEAN[i] <= GATE_BOUNDS[i, 1]


class TestGateReward:
    def test_finite_output(self):
        gate_cfg = GateConfig()
        reward_cfg = RewardConfig(t_probe_z=15.0, t_probe_x=0.2)
        fn, _ = build_gate_reward(
            "proxy", FAST_PARAMS, reward_cfg=reward_cfg, gate_cfg=gate_cfg
        )
        r = fn(X_GATE)
        assert jnp.isfinite(r), f"Gate reward not finite: {r}"

    def test_vmap(self):
        gate_cfg = GateConfig()
        reward_cfg = RewardConfig(t_probe_z=15.0, t_probe_x=0.2)
        _, batched = build_gate_reward(
            "proxy", FAST_PARAMS, reward_cfg=reward_cfg, gate_cfg=gate_cfg
        )
        xs = jnp.stack([X_GATE, X_GATE * 1.1])
        rs = batched(xs)
        assert rs.shape == (2,)
        assert jnp.all(jnp.isfinite(rs))
