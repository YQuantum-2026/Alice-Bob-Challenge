"""Tests for src/moon_cat.py — moon cat extension.

Validates moon cat Hamiltonian, 5D rewards, and comparison runner.

Reference:
  Rousseau et al. (2025), arXiv:2502.07892 — squeezed cat qubits.
"""

import dynamiqs as dq
import jax.numpy as jnp

from src.cat_qubit import build_jump_ops, build_operators
from src.moon_cat import (
    MOON_CAT_BOUNDS,
    MOON_CAT_MEAN,
    build_moon_hamiltonian,
    build_moon_reward,
    measure_moon_lifetimes,
)
from tests.conftest import FAST_PARAMS

X_MOON = jnp.array([1.0, 0.0, 4.0, 0.0, 0.3])


class TestMoonHamiltonian:
    def test_hermiticity(self):
        """Moon cat H must be Hermitian for any lambda."""
        a, b = build_operators(FAST_PARAMS)
        H = build_moon_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0, 0.3)
        H_dense = H.to_jax()
        diff = float(jnp.max(jnp.abs(H_dense - jnp.conj(H_dense.T))))
        assert diff < 1e-10, f"H not Hermitian: max diff = {diff}"

    def test_hermiticity_complex_g2(self):
        """Hermiticity with complex g2 (float32 precision: ~1e-7)."""
        a, b = build_operators(FAST_PARAMS)
        H = build_moon_hamiltonian(a, b, 1.5, 0.3, 3.0, -0.5, 0.5)
        H_dense = H.to_jax()
        diff = float(jnp.max(jnp.abs(H_dense - jnp.conj(H_dense.T))))
        assert diff < 1e-5, f"H not Hermitian: max diff = {diff}"

    def test_reduces_to_standard_at_lambda_zero(self):
        """With lambda=0, moon cat = standard cat."""
        from src.cat_qubit import build_hamiltonian

        a, b = build_operators(FAST_PARAMS)
        H_std = build_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0)
        H_moon = build_moon_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0, 0.0)
        diff = float(jnp.max(jnp.abs(H_std.to_jax() - H_moon.to_jax())))
        assert diff < 1e-10, f"lambda=0 should give standard H, diff={diff}"

    def test_shape(self):
        a, b = build_operators(FAST_PARAMS)
        H = build_moon_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0, 0.3)
        dim = FAST_PARAMS.na * FAST_PARAMS.nb
        assert H.shape == (dim, dim)

    def test_lambda_changes_steady_state(self):
        """Nonzero lambda must produce different dynamics than lambda=0."""
        a, b = build_operators(FAST_PARAMS)
        H_std = build_moon_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0, 0.0)
        H_moon = build_moon_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0, 0.5)

        jump_ops = build_jump_ops(a, b, FAST_PARAMS)
        psi0 = dq.tensor(dq.fock(FAST_PARAMS.na, 0), dq.fock(FAST_PARAMS.nb, 0))
        tsave = jnp.array([0.0, 10.0])

        n_op = a.dag() @ a
        res_std = dq.mesolve(
            H_std,
            jump_ops,
            psi0,
            tsave,
            exp_ops=[n_op],
            options=dq.Options(progress_meter=False),
        )
        res_moon = dq.mesolve(
            H_moon,
            jump_ops,
            psi0,
            tsave,
            exp_ops=[n_op],
            options=dq.Options(progress_meter=False),
        )

        n_std = float(res_std.expects[0, -1].real)
        n_moon = float(res_moon.expects[0, -1].real)
        assert abs(n_std - n_moon) > 1e-3, (
            f"lam=0.5 should change <n> vs lam=0: {n_std:.4f} vs {n_moon:.4f}"
        )


class TestMoonBounds:
    def test_shape(self):
        assert MOON_CAT_BOUNDS.shape == (5, 2)

    def test_mean_within_bounds(self):
        for i in range(5):
            assert MOON_CAT_BOUNDS[i, 0] <= MOON_CAT_MEAN[i] <= MOON_CAT_BOUNDS[i, 1]


class TestMoonReward:
    def test_finite_output(self):
        fn, _ = build_moon_reward("proxy", FAST_PARAMS)
        r = fn(X_MOON)
        assert jnp.isfinite(r), f"Moon reward not finite: {r}"

    def test_vmap(self):
        _, batched = build_moon_reward("proxy", FAST_PARAMS)
        xs = jnp.stack([X_MOON, X_MOON * 0.5])
        rs = batched(xs)
        assert rs.shape == (2,)
        assert jnp.all(jnp.isfinite(rs))


class TestMeasureMoonLifetimes:
    """Smoke test for measure_moon_lifetimes (vacuum-based)."""

    def test_returns_finite_dict(self):
        """measure_moon_lifetimes should return finite Tz, Tx, bias for good params."""
        result = measure_moon_lifetimes(
            1.0,
            0.0,
            4.0,
            0.0,
            0.3,
            tfinal_z=50.0,
            tfinal_x=0.5,
            npoints=20,
            params=FAST_PARAMS,
        )
        assert "Tz" in result
        assert "Tx" in result
        assert "bias" in result
        import numpy as np

        assert np.isfinite(result["Tz"]), f"T_Z not finite: {result['Tz']}"
        assert np.isfinite(result["Tx"]), f"T_X not finite: {result['Tx']}"
        assert np.isfinite(result["bias"]), f"bias not finite: {result['bias']}"

    def test_returns_alpha_vacuum(self):
        """measure_moon_lifetimes should return vacuum-based alpha estimate."""
        import numpy as np

        result = measure_moon_lifetimes(
            1.0,
            0.0,
            4.0,
            0.0,
            0.1,
            tfinal_z=5.0,
            tfinal_x=0.3,
            npoints=10,
            params=FAST_PARAMS,
        )
        assert "alpha_vacuum" in result, "Missing alpha_vacuum key"
        assert result["alpha_vacuum"] > 0, "alpha_vacuum should be positive"
        assert np.isfinite(result["alpha_vacuum"]), (
            f"alpha_vacuum not finite: {result['alpha_vacuum']}"
        )
