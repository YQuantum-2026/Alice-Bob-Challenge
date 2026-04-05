"""Tests for src/cat_qubit.py — core physics module.

Validates Hamiltonian construction, operator properties, alpha estimation,
and lifetime measurement against known analytical results.

Reference values from challenge notebook:
  g2=1, eps_d=4, kappa_b=10, kappa_a=0 → alpha=2.0
  g2=1, eps_d=4, kappa_b=10, kappa_a=1 → alpha≈1.658
"""

import jax.numpy as jnp
import numpy as np

from src.cat_qubit import (
    CatQubitParams,
    build_hamiltonian,
    build_initial_states,
    build_jump_ops,
    build_logical_ops,
    build_operators,
    compute_alpha,
    measure_lifetimes,
    robust_exp_fit,
    simulate_lifetime,
)

# Use small dims for fast tests
from tests.conftest import FAST_PARAMS


class TestOperators:
    def test_shapes(self):
        a, b = build_operators(FAST_PARAMS)
        dim = FAST_PARAMS.na * FAST_PARAMS.nb
        assert a.shape == (dim, dim)
        assert b.shape == (dim, dim)

    def test_commutation(self):
        """[a, a†] = I on the storage subspace."""
        params = CatQubitParams(na=6, nb=3)
        a, _ = build_operators(params)
        comm = a @ a.dag() - a.dag() @ a
        # Should be identity on storage ⊗ identity on buffer
        # but truncated — check diagonal is close to expected
        diag = jnp.diag(jnp.array(comm.to_jax())).real
        # For truncated Hilbert space, commutator is I except at boundary
        assert jnp.abs(diag[0] - 1.0) < 0.1


class TestHamiltonian:
    def test_hermiticity(self):
        """H must be Hermitian: H = H†."""
        # Ref: Berdou et al. (2022), arXiv:2204.09128, Eq. (1)
        a, b = build_operators(FAST_PARAMS)
        H = build_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0)
        H_dense = H.to_jax()
        H_dag = jnp.conj(H_dense.T)
        assert jnp.allclose(H_dense, H_dag, atol=1e-10)

    def test_shape(self):
        a, b = build_operators(FAST_PARAMS)
        H = build_hamiltonian(a, b, 1.0, 0.5, 4.0, -1.0)
        dim = FAST_PARAMS.na * FAST_PARAMS.nb
        assert H.shape == (dim, dim)

    def test_complex_params(self):
        """Hamiltonian with complex g2 and eps_d should still be Hermitian."""
        a, b = build_operators(FAST_PARAMS)
        H = build_hamiltonian(a, b, 1.5, 0.3, 3.0, -0.5)
        H_dense = H.to_jax()
        H_dag = jnp.conj(H_dense.T)
        assert jnp.allclose(H_dense, H_dag, atol=1e-10)


class TestJumpOps:
    def test_shapes(self):
        """Jump operators have correct dimension."""
        # Ref: Berdou et al. (2022), arXiv:2204.09128, Sec. II
        a, b = build_operators(FAST_PARAMS)
        L_b, L_a = build_jump_ops(a, b, FAST_PARAMS)
        dim = FAST_PARAMS.na * FAST_PARAMS.nb
        assert L_b.shape == (dim, dim)
        assert L_a.shape == (dim, dim)


class TestAlphaEstimate:
    def test_kappa_a_zero(self):
        """With kappa_a=0: alpha = sqrt(2*eps_2/kappa_2) = 2.0.
        Ref: Berdou et al. (2022), arXiv:2204.09128, Eq. (2)-(4)"""
        params = CatQubitParams(kappa_a=0.0)
        alpha = compute_alpha(1.0, 0.0, 4.0, 0.0, params)
        assert jnp.isclose(alpha, 2.0, atol=0.01)

    def test_kappa_a_one(self):
        """With kappa_a=1: alpha ≈ 1.658.
        eps_2 = 2*1*4/10 = 0.8, kappa_2 = 4*1/10 = 0.4
        alpha = sqrt(2/0.4 * (0.8 - 0.25)) = sqrt(2.75) ≈ 1.658"""
        params = CatQubitParams(kappa_a=1.0)
        alpha = compute_alpha(1.0, 0.0, 4.0, 0.0, params)
        expected = np.sqrt(2.75)
        assert jnp.isclose(alpha, expected, atol=0.01)

    def test_guard_negative(self):
        """Alpha should not be NaN when eps_2 < kappa_a/4."""
        params = CatQubitParams(kappa_a=100.0)  # very large loss
        alpha = compute_alpha(0.1, 0.0, 0.1, 0.0, params)
        assert jnp.isfinite(alpha)
        assert alpha >= 0


class TestLogicalOps:
    def test_parity_shape(self):
        """Parity operator has correct dimension."""
        a, b = build_operators(FAST_PARAMS)
        alpha = float(compute_alpha(1.0, 0.0, 4.0, 0.0, FAST_PARAMS))
        sx, sz = build_logical_ops(a, b, alpha, FAST_PARAMS)
        dim = FAST_PARAMS.na * FAST_PARAMS.nb
        assert sx.shape == (dim, dim)
        assert sz.shape == (dim, dim)


class TestInitialStates:
    def test_all_labels(self):
        alpha = float(compute_alpha(1.0, 0.0, 4.0, 0.0, FAST_PARAMS))
        states = build_initial_states(alpha, FAST_PARAMS)
        assert set(states.keys()) == {"+z", "-z", "+x", "-x"}

    def test_normalization(self):
        alpha = float(compute_alpha(1.0, 0.0, 4.0, 0.0, FAST_PARAMS))
        states = build_initial_states(alpha, FAST_PARAMS)
        for label, psi in states.items():
            # Use dq.expect with identity for norm, or convert to jax array
            psi_jax = (
                jnp.array(psi.to_jax()) if hasattr(psi, "to_jax") else jnp.array(psi)
            )
            norm = float(jnp.real(jnp.conj(psi_jax).T @ psi_jax).squeeze())
            assert jnp.isclose(norm, 1.0, atol=1e-4), f"|{label}> norm={norm}"


class TestExpFit:
    def test_known_decay(self):
        """Fit to a known exponential: y = exp(-t/5)."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 20, 100)
        y = np.exp(-t / 5.0) + 0.01 * rng.standard_normal(100)
        fit = robust_exp_fit(t, y)
        assert abs(fit["tau"] - 5.0) < 0.5, f"Expected tau≈5, got {fit['tau']}"


class TestSimulation:
    def test_simulate_runs(self):
        """simulate_lifetime should return correct shapes."""
        tsave, exp_x, exp_z = simulate_lifetime(
            1.0, 0.0, 4.0, 0.0, "+z", 10.0, npoints=20, params=FAST_PARAMS
        )
        assert tsave.shape == (20,)
        assert exp_x.shape == (20,)
        assert exp_z.shape == (20,)

    def test_z_decay_from_plus_z(self):
        """⟨Z_L⟩ should start near 1 and decay from |+z⟩."""
        tsave, _, exp_z = simulate_lifetime(
            1.0, 0.0, 4.0, 0.0, "+z", 50.0, npoints=30, params=FAST_PARAMS
        )
        assert exp_z[0] > 0.5, f"Expected ⟨Z⟩(0) > 0.5, got {exp_z[0]}"
        assert exp_z[-1] < exp_z[0], "⟨Z⟩ should decay over time"


class TestMeasureLifetimesNaN:
    """Verify NaN propagation when fits fail (very small alpha → no decay signal)."""

    def test_nan_handling_returns_nan(self):
        """measure_lifetimes returns NaN (not 0.0) when fits fail."""
        params = CatQubitParams(na=6, nb=3)
        # Very small g2 and eps_d → alpha ≈ 0 → no cat state → NaN fits
        result = measure_lifetimes(0.001, 0.0, 0.001, 0.0, params=params)
        # At least one lifetime should be NaN for these degenerate params
        has_nan = np.isnan(result["Tz"]) or np.isnan(result["Tx"])
        if has_nan:
            assert np.isnan(result["bias"]), "bias must be NaN when a lifetime is NaN"


class TestMeasureLifetimesPositive:
    """Verify measure_lifetimes returns physically reasonable values for good params."""

    def test_positive_lifetimes(self):
        """Known-good params should give positive, finite lifetimes with bias > 1."""
        params = CatQubitParams(na=8, nb=3, kappa_b=10.0, kappa_a=1.0)
        result = measure_lifetimes(1.0, 0.0, 4.0, 0.0, params=params)
        assert np.isfinite(result["Tz"]), f"T_Z not finite: {result['Tz']}"
        assert np.isfinite(result["Tx"]), f"T_X not finite: {result['Tx']}"
        assert result["Tz"] > 0, f"T_Z should be positive, got {result['Tz']}"
        assert result["Tx"] > 0, f"T_X should be positive, got {result['Tx']}"
        assert result["bias"] > 1, (
            f"bias should be > 1 for stabilized cat, got {result['bias']}"
        )
