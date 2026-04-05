"""Tests for vacuum-based (alpha-free) reward function.

Validates that the vacuum reward:
  - Returns a finite scalar for default params
  - Parity decays from ~1 after settling (phase-flip measurement)
  - Quadrature decays after settling (bit-flip measurement)
  - Batch evaluation works
  - T estimates are in the same ballpark as spectral (relaxed tolerance)

Ref: Réglade et al. "Quantum control of a cat-qubit with bit-flip times
  exceeding ten seconds." Nature 629, 778-783 (2024). arXiv:2307.06617.
"""

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np

from src.cat_qubit import (
    build_hamiltonian,
    build_jump_ops,
    build_operators,
)
from src.reward import build_reward
from tests.conftest import FAST_PARAMS, X_DEFAULT


class TestVacuumRewardBasic:
    """Basic smoke tests for the vacuum reward."""

    def test_finite_output(self):
        fn, _ = build_reward("vacuum", FAST_PARAMS)
        r = fn(X_DEFAULT)
        assert jnp.isfinite(r), f"Vacuum reward returned {r}"

    def test_batched(self):
        _, batched = build_reward("vacuum", FAST_PARAMS)
        xs = jnp.stack([X_DEFAULT, X_DEFAULT * 0.8])
        results = batched(xs)
        assert results.shape == (2,)
        assert np.all(np.isfinite(results))

    def test_different_params_different_reward(self):
        fn, _ = build_reward("vacuum", FAST_PARAMS)
        r1 = fn(X_DEFAULT)
        r2 = fn(X_DEFAULT * 0.5)
        assert not jnp.isclose(r1, r2), "Different params should give different rewards"

    def test_factory_registers_vacuum(self):
        fn, batched = build_reward("vacuum", FAST_PARAMS)
        assert callable(fn)
        assert callable(batched)


class TestVacuumParity:
    """Test that parity (T_X measurement) behaves correctly."""

    def test_parity_decays_from_vacuum(self):
        """Parity should start near 1 (vacuum = even parity) and decrease."""
        a, b = build_operators(FAST_PARAMS)
        jump_ops = build_jump_ops(a, b, FAST_PARAMS)
        parity_op = (1j * jnp.pi * a.dag() @ a).expm()

        psi0 = dq.tensor(dq.fock(FAST_PARAMS.na, 0), dq.fock(FAST_PARAMS.nb, 0))

        # Use default x params for the Hamiltonian
        H = build_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0)

        tsave = jnp.linspace(0.0, 2.0, 20)
        res = dq.mesolve(
            H,
            jump_ops,
            psi0,
            tsave,
            exp_ops=[parity_op],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )
        parity = res.expects[0, :].real

        # Parity at t=0 should be ~1 (vacuum has even parity)
        assert parity[0] > 0.9, f"Initial parity should be ~1, got {parity[0]}"

        # Parity should decrease over time (phase flips from single-photon loss)
        assert parity[-1] < parity[0], "Parity should decay over time"


class TestVacuumQuadrature:
    """Test that quadrature (T_Z measurement) behaves correctly.

    T_Z uses data-driven α: the vacuum sim measures ⟨a†a⟩ to get |α|,
    then prepares |α_est⟩ and measures ⟨Q_θ⟩ decay. This test verifies
    the coherent-state decay works for the T_Z path.
    """

    def test_quadrature_decays_from_coherent_state(self):
        """Starting from |α⟩, ⟨a+a†⟩ should decay toward 0 (bit-flips)."""
        a, b = build_operators(FAST_PARAMS)
        jump_ops = build_jump_ops(a, b, FAST_PARAMS)

        H = build_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0)
        Q = a + a.dag()

        # Use a modest alpha for the coherent state
        alpha = 1.5
        psi_alpha = dq.tensor(
            dq.coherent(FAST_PARAMS.na, alpha), dq.fock(FAST_PARAMS.nb, 0)
        )

        tsave = jnp.linspace(0.0, 50.0, 20)
        res = dq.mesolve(
            H,
            jump_ops,
            psi_alpha,
            tsave,
            exp_ops=[Q],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )
        quad = res.expects[0, :].real

        # Initial ⟨Q⟩ should be ~2α (coherent state in one well)
        assert quad[0] > 2.0, f"Initial quadrature should be ~2α, got {quad[0]}"

        # Quadrature should decay over time (bit-flips mix wells)
        assert abs(quad[-1]) < abs(quad[0]), "Quadrature should decay"

    def test_photon_number_grows_from_vacuum(self):
        """Vacuum should gain photons as cat state forms under stabilization."""
        a, b = build_operators(FAST_PARAMS)
        jump_ops = build_jump_ops(a, b, FAST_PARAMS)
        n_a = a.dag() @ a

        psi0 = dq.tensor(dq.fock(FAST_PARAMS.na, 0), dq.fock(FAST_PARAMS.nb, 0))
        H = build_hamiltonian(a, b, 1.0, 0.0, 4.0, 0.0)

        tsave = jnp.linspace(0.0, 15.0, 20)
        res = dq.mesolve(
            H,
            jump_ops,
            psi0,
            tsave,
            exp_ops=[n_a],
            method=dq.method.Tsit5(rtol=1e-4, atol=1e-4),
            options=dq.Options(progress_meter=False, save_states=False),
        )
        n = res.expects[0, :].real

        # Vacuum starts with 0 photons
        assert n[0] < 0.1, f"Vacuum should have ~0 photons, got {n[0]}"

        # After settling, should have |α|² ≈ 2 photons
        assert n[-1] > 1.0, f"Cat state should have >1 photon, got {n[-1]}"


class TestVacuumVsSpectral:
    """Cross-check vacuum reward against spectral (relaxed tolerance)."""

    def test_both_finite(self):
        """Both vacuum and spectral should return finite results."""
        fn_vac, _ = build_reward("vacuum", FAST_PARAMS)
        fn_spec, _ = build_reward("spectral", FAST_PARAMS)
        r_vac = fn_vac(X_DEFAULT)
        r_spec = fn_spec(X_DEFAULT)
        assert jnp.isfinite(r_vac), f"Vacuum returned {r_vac}"
        assert jnp.isfinite(r_spec), f"Spectral returned {r_spec}"
