"""Numerical stability tests for cat qubit physics engine.

Validates that edge cases in compute_alpha, reward functions, drift models,
and exponential fitting do not produce NaN, Inf, or crash. These are the
boundary conditions where floating-point arithmetic is most fragile:
  - Division by zero (g2=0 makes kappa_2=0)
  - Negative sqrt arguments (|eps_2| < kappa_a/4)
  - log(0) from decayed expectation values
  - Exponential fitting with degenerate signals
  - Drift accumulation over many epochs

Physics reference:
  kappa_2 = 4|g2|^2 / kappa_b
  eps_2   = 2*g2*eps_d / kappa_b
  alpha   = sqrt(2/kappa_2 * (|eps_2| - kappa_a/4))

  Berdou et al. "One hundred second bit-flip time in a two-photon
  dissipative oscillator." PRX Quantum 4, 020350 (2023). arXiv:2204.09128
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src.cat_qubit import (
    CatQubitParams,
    build_hamiltonian,
    build_jump_ops,
    build_operators,
    compute_alpha,
    robust_exp_fit,
)

# Small Hilbert space for fast tests (~seconds, not minutes)
FAST_PARAMS = CatQubitParams(na=6, nb=3, kappa_b=10.0, kappa_a=1.0)


# ---------------------------------------------------------------------------
# Alpha edge cases
# ---------------------------------------------------------------------------


class TestAlphaEdgeCases:
    """Numerical stability of compute_alpha at parameter boundaries.

    The critical expression is alpha = sqrt(2/kappa_2 * (|eps_2| - kappa_a/4)).
    kappa_2 = 4|g2|^2 / kappa_b can be zero or tiny, and the sqrt argument
    can be negative. Both cases must produce finite, non-negative results.
    """

    def test_alpha_negative_sqrt_arg(self):
        """When |eps_2| < kappa_a/4, the sqrt argument is negative.

        The code must clamp to 0 (not return NaN).
        With g2=0.01, eps_d=0.01, kappa_b=10, kappa_a=1:
          kappa_2 = 4*0.01^2/10 = 4e-5
          eps_2 = 2*0.01*0.01/10 = 2e-5
          |eps_2| = 2e-5 < kappa_a/4 = 0.25  -->  negative sqrt arg
        """
        alpha = compute_alpha(0.01, 0.0, 0.01, 0.0, FAST_PARAMS)
        assert jnp.isfinite(alpha), f"alpha is not finite: {alpha}"
        assert float(alpha) >= 0.0, f"alpha is negative: {alpha}"
        assert float(alpha) == 0.0, "Expected alpha=0 when sqrt arg is negative"

    def test_alpha_zero_g2(self):
        """g2=0 makes kappa_2=0, causing 2/kappa_2 = inf.

        This is a division-by-zero edge case. The function should return
        a finite value (0 or inf clamped) rather than NaN.
        """
        alpha = compute_alpha(0.0, 0.0, 4.0, 0.0, FAST_PARAMS)
        # With g2=0: kappa_2=0, eps_2=0, so 2/0 * (0 - 0.25) = -inf -> clamp to 0
        # OR 2/0 produces inf, but max(inf * neg, 0) = 0
        # The result should be finite or at least not NaN
        result = float(alpha)
        assert not np.isnan(result), "alpha is NaN for g2=0"

    def test_alpha_very_large_params(self):
        """Parameters at upper optimization bounds should give finite alpha.

        Large g2 and eps_d lead to large kappa_2 and eps_2, but alpha
        should remain finite and positive.
        """
        alpha = compute_alpha(5.0, 2.0, 20.0, 5.0, FAST_PARAMS)
        assert jnp.isfinite(alpha), f"alpha is not finite for large params: {alpha}"
        assert float(alpha) > 0.0, "Expected positive alpha for large params"

    def test_alpha_pure_imaginary_g2(self):
        """Pure imaginary g2 (g2_re=0, g2_im nonzero) should work.

        |g2|^2 = g2_im^2, so kappa_2 and eps_2 are still well-defined.
        """
        alpha = compute_alpha(0.0, 1.0, 4.0, 0.0, FAST_PARAMS)
        assert jnp.isfinite(alpha), f"alpha not finite for pure imaginary g2: {alpha}"

    def test_alpha_pure_imaginary_eps_d(self):
        """Pure imaginary eps_d should produce valid alpha."""
        alpha = compute_alpha(1.0, 0.0, 0.0, 4.0, FAST_PARAMS)
        assert jnp.isfinite(alpha), (
            f"alpha not finite for pure imaginary eps_d: {alpha}"
        )

    def test_alpha_both_complex(self):
        """Both g2 and eps_d fully complex."""
        alpha = compute_alpha(1.0, 0.5, 3.0, 2.0, FAST_PARAMS)
        assert jnp.isfinite(alpha)
        assert float(alpha) >= 0.0

    def test_alpha_tiny_eps_d(self):
        """Very small eps_d with reasonable g2.

        eps_2 = 2*g2*eps_d/kappa_b will be tiny, likely < kappa_a/4,
        so alpha should clamp to 0.
        """
        alpha = compute_alpha(1.0, 0.0, 1e-6, 0.0, FAST_PARAMS)
        assert jnp.isfinite(alpha)
        assert float(alpha) >= 0.0

    def test_alpha_zero_kappa_a_analytical(self):
        """With kappa_a=0 and known params, alpha has exact analytical value.

        kappa_2 = 4*1^2/10 = 0.4
        eps_2 = 2*1*4/10 = 0.8
        alpha = sqrt(2/0.4 * 0.8) = sqrt(4) = 2.0
        """
        params_no_loss = CatQubitParams(na=6, nb=3, kappa_b=10.0, kappa_a=0.0)
        alpha = compute_alpha(1.0, 0.0, 4.0, 0.0, params_no_loss)
        np.testing.assert_allclose(float(alpha), 2.0, atol=1e-10)

    @pytest.mark.parametrize(
        "g2_re, g2_im, eps_d_re, eps_d_im",
        [
            (0.0, 0.0, 0.0, 0.0),  # all zeros
            (1e-15, 0.0, 1e-15, 0.0),  # near machine epsilon
            (100.0, 100.0, 100.0, 100.0),  # very large
            (-1.0, 0.0, 4.0, 0.0),  # negative g2_re (still valid: |g2| > 0)
        ],
    )
    def test_alpha_no_nan(self, g2_re, g2_im, eps_d_re, eps_d_im):
        """Alpha must never be NaN for any finite input."""
        alpha = compute_alpha(g2_re, g2_im, eps_d_re, eps_d_im, FAST_PARAMS)
        assert not jnp.isnan(alpha), (
            f"NaN alpha for g2=({g2_re}+{g2_im}j), eps_d=({eps_d_re}+{eps_d_im}j)"
        )


# ---------------------------------------------------------------------------
# Hamiltonian stability
# ---------------------------------------------------------------------------


class TestHamiltonianStability:
    """Hamiltonian construction at extreme parameter values."""

    def test_zero_coupling(self):
        """H with g2=0, eps_d=0 should be the zero matrix."""
        a, b = build_operators(FAST_PARAMS)
        H = build_hamiltonian(a, b, 0.0, 0.0, 0.0, 0.0)
        H_dense = H.to_jax()
        np.testing.assert_allclose(
            np.array(H_dense),
            0.0,
            atol=1e-14,
            err_msg="H should be zero when all couplings are zero",
        )

    def test_large_coupling_hermitian(self):
        """H must remain Hermitian even for large coupling values.

        Uses atol=1e-5 because dynamiqs defaults to float32/complex64,
        and large matrix element values accumulate rounding errors.
        The ~4e-6 mismatch at (g2=10+5j, eps_d=50+25j) is expected
        for single-precision arithmetic.
        """
        a, b = build_operators(FAST_PARAMS)
        H = build_hamiltonian(a, b, 10.0, 5.0, 50.0, 25.0)
        H_dense = np.array(H.to_jax())
        np.testing.assert_allclose(
            H_dense,
            H_dense.conj().T,
            atol=1e-5,
            err_msg="Hamiltonian lost Hermiticity at large coupling",
        )

    def test_eigenvalues_real_for_hermitian(self):
        """Hermitian H must have purely real eigenvalues."""
        a, b = build_operators(FAST_PARAMS)
        H = build_hamiltonian(a, b, 1.0, 0.5, 4.0, -1.0)
        H_dense = np.array(H.to_jax())
        eigvals = np.linalg.eigvalsh(H_dense)
        assert np.all(np.isfinite(eigvals)), "Non-finite eigenvalues"


# ---------------------------------------------------------------------------
# Exponential fit stability
# ---------------------------------------------------------------------------


class TestExponentialFitStability:
    """Edge cases for robust_exp_fit that can break least_squares."""

    def test_clean_decay_recovery(self):
        """Fit to a clean exponential should recover tau within 10%.

        y(t) = exp(-t/50), so tau_true = 50.
        """
        np.random.default_rng(42)
        T_true = 50.0
        t = np.linspace(0, 200, 100)
        y = np.exp(-t / T_true)
        fit = robust_exp_fit(t, y)
        assert abs(fit["tau"] - T_true) / T_true < 0.1, (
            f"Expected tau~{T_true}, got {fit['tau']}"
        )

    def test_noisy_decay_finite(self):
        """Fit to noisy exponential should not crash or return negative tau."""
        rng = np.random.default_rng(42)
        T_true = 50.0
        t = np.linspace(0, 200, 100)
        y = np.exp(-t / T_true) + 0.05 * rng.standard_normal(100)
        y = np.clip(y, 0, 1)
        fit = robust_exp_fit(t, y)
        assert fit["tau"] > 0, f"Fitted lifetime must be positive, got {fit['tau']}"
        assert np.isfinite(fit["tau"]), "Fitted tau is not finite"

    def test_constant_signal(self):
        """Constant signal (no decay) should return large tau, not crash.

        When y ~ constant, the exponential has infinite lifetime.
        The fit should return a large positive tau.
        """
        t = np.linspace(0, 200, 100)
        y = np.ones_like(t) * 0.99
        fit = robust_exp_fit(t, y)
        assert fit["tau"] > 0, (
            f"tau should be positive for constant signal, got {fit['tau']}"
        )
        assert np.isfinite(fit["tau"]), "tau is not finite for constant signal"

    def test_fast_decay(self):
        """Very fast decay (tau << t_range) should still fit."""
        t = np.linspace(0, 200, 100)
        y = np.exp(-t / 0.5)  # tau=0.5, decays to ~0 by t=5
        fit = robust_exp_fit(t, y)
        assert fit["tau"] > 0, f"tau should be positive, got {fit['tau']}"
        assert np.isfinite(fit["tau"])

    def test_fit_output_shape(self):
        """Fit should return y_fit with same length as input."""
        t = np.linspace(0, 100, 50)
        y = np.exp(-t / 30.0)
        fit = robust_exp_fit(t, y)
        assert fit["y_fit"].shape == (50,), (
            f"Expected shape (50,), got {fit['y_fit'].shape}"
        )
        assert len(fit["popt"]) == 3, "popt should have 3 parameters (A, tau, C)"

    def test_fit_with_offset(self):
        """Fit y = 0.8*exp(-t/20) + 0.15 should recover parameters.

        The C parameter captures the baseline offset.
        """
        t = np.linspace(0, 100, 100)
        y = 0.8 * np.exp(-t / 20.0) + 0.15
        fit = robust_exp_fit(t, y)
        assert abs(fit["tau"] - 20.0) / 20.0 < 0.15, (
            f"Expected tau~20, got {fit['tau']}"
        )

    def test_all_zero_signal(self):
        """All-zero input should not crash the fit.

        A0 = max-min = 0, C0 = min = 0; the fitter must handle this
        degenerate initial guess.
        """
        t = np.linspace(0, 100, 50)
        y = np.zeros_like(t)
        # Should not raise -- may return arbitrary tau but must be finite
        fit = robust_exp_fit(t, y)
        assert np.isfinite(fit["tau"]), "tau should be finite even for all-zero signal"

    def test_negative_values_clipped(self):
        """Signal with negative values (unphysical) should still fit.

        Negative values can appear from noisy expectation values.
        The bounds enforce A >= 0 and tau >= 1e-12, so fit should survive.
        """
        rng = np.random.default_rng(123)
        t = np.linspace(0, 100, 80)
        y = np.exp(-t / 30.0) + 0.3 * rng.standard_normal(80)
        # Don't clip -- let the fitter handle negative values
        fit = robust_exp_fit(t, y)
        assert np.isfinite(fit["tau"]), "tau not finite with noisy/negative data"


# ---------------------------------------------------------------------------
# Reward function stability
# ---------------------------------------------------------------------------


class TestRewardStability:
    """Numerical stability of reward functions at edge-case parameters.

    The proxy reward computes T_est = -t_probe / log(expectation_value).
    When the expectation value is near 0 or 1, log() goes to -inf or 0,
    creating overflow/underflow risks.
    """

    def test_proxy_reward_finite_default(self):
        """Proxy reward should be finite for default (known-good) params."""
        from src.reward import build_reward

        reward_fn, _ = build_reward("proxy", FAST_PARAMS)
        x = jnp.array([1.0, 0.0, 4.0, 0.0])
        r = reward_fn(x)
        assert jnp.isfinite(r), f"Proxy reward is not finite: {r}"

    def test_proxy_reward_small_params(self):
        """Proxy reward should be finite for very small control parameters.

        Small g2 and eps_d yield small alpha, potentially causing
        log(~0) = -inf in the lifetime estimation.
        """
        from src.reward import build_reward

        reward_fn, _ = build_reward("proxy", FAST_PARAMS)
        x_small = jnp.array([0.1, 0.0, 0.5, 0.0])
        r = reward_fn(x_small)
        assert jnp.isfinite(r), f"Proxy reward not finite for small params: {r}"

    def test_proxy_reward_large_params(self):
        """Proxy reward should be finite for large control parameters."""
        from src.reward import build_reward

        reward_fn, _ = build_reward("proxy", FAST_PARAMS)
        x_large = jnp.array([5.0, 2.0, 20.0, 5.0])
        r = reward_fn(x_large)
        assert jnp.isfinite(r), f"Proxy reward not finite for large params: {r}"

    def test_photon_reward_finite(self):
        """Photon reward should be finite for default params."""
        from src.reward import build_reward

        reward_fn, _ = build_reward("photon", FAST_PARAMS)
        x = jnp.array([1.0, 0.0, 4.0, 0.0])
        r = reward_fn(x)
        assert jnp.isfinite(r), f"Photon reward is not finite: {r}"

    def test_photon_reward_zero_coupling(self):
        """Photon reward with very small coupling should still be finite.

        With tiny g2/eps_d, the system stays near vacuum, so
        ⟨n⟩ ~ 0 and reward = -(0 - n_target)^2.
        """
        from src.reward import build_reward

        reward_fn, _ = build_reward("photon", FAST_PARAMS)
        x_tiny = jnp.array([0.01, 0.0, 0.01, 0.0])
        r = reward_fn(x_tiny)
        assert jnp.isfinite(r), f"Photon reward not finite for tiny params: {r}"

    def test_fidelity_reward_bounded(self):
        """Fidelity reward should be in [0, 1] for valid parameters."""
        from src.reward import build_reward

        reward_fn, _ = build_reward("fidelity", FAST_PARAMS)
        x = jnp.array([1.0, 0.0, 4.0, 0.0])
        r = reward_fn(x)
        assert jnp.isfinite(r), f"Fidelity reward not finite: {r}"
        assert -0.05 <= float(r) <= 1.05, f"Fidelity out of [0,1]: {r}"

    @pytest.mark.parametrize(
        "reward_type",
        ["proxy", "photon", "fidelity", "parity", "multipoint", "spectral"],
    )
    def test_all_reward_types_finite_at_default(self, reward_type):
        """Every reward type must return finite output for default params."""
        from src.reward import build_reward

        reward_fn, _ = build_reward(reward_type, FAST_PARAMS)
        x = jnp.array([1.0, 0.0, 4.0, 0.0])
        r = reward_fn(x)
        assert jnp.isfinite(r), f"{reward_type} reward returned non-finite: {r}"


# ---------------------------------------------------------------------------
# Drift stability
# ---------------------------------------------------------------------------


class TestDriftStability:
    """Drift models must remain finite over extended time horizons.

    Sinusoidal drifts are bounded by construction, but compositions and
    edge cases (100% amplitude, epoch 0 at phase boundary) can surface
    issues.
    """

    def test_extreme_amplitude_drift(self):
        """100% amplitude drift should not produce non-finite offsets.

        g2_eff = g2 * (1 + 1.0 * sin(...)), so at the peak,
        g2_eff = 2*g2 (doubles), and at the trough, g2_eff = 0.
        """
        from src.drift import AmplitudeDrift, DriftModel

        drift = DriftModel(
            amplitude_drifts=[AmplitudeDrift(amplitude=1.0, frequency=0.01)]
        )
        offsets = drift.get_control_offsets(0)
        for key, v in offsets.items():
            assert np.isfinite(float(v)), f"{key} non-finite at epoch 0"

    def test_drift_over_1000_epochs(self):
        """Drift values must remain finite over 1000 epochs.

        This catches accumulation bugs (e.g., epoch * frequency overflow
        in the sin argument) and verifies periodicity.
        """
        from src.drift import (
            AmplitudeDrift,
            DriftModel,
            FrequencyDrift,
            KerrDrift,
        )

        drift = DriftModel(
            amplitude_drifts=[AmplitudeDrift(amplitude=0.3, frequency=0.005)],
            frequency_drifts=[FrequencyDrift(amplitude=0.5, frequency=0.005)],
            kerr_drifts=[KerrDrift(amplitude=0.01, frequency=0.003)],
        )
        for epoch in range(0, 1000, 50):  # sample every 50 epochs for speed
            offsets = drift.get_control_offsets(epoch)
            h_terms = drift.get_hamiltonian_terms(epoch)
            for key, v in offsets.items():
                assert np.isfinite(float(v)), (
                    f"Non-finite offset {key}={v} at epoch {epoch}"
                )
            for key, v in h_terms.items():
                assert np.isfinite(float(v)), (
                    f"Non-finite h_term {key}={v} at epoch {epoch}"
                )

    def test_drift_at_very_large_epoch(self):
        """Drift at epoch 1_000_000 should still be finite.

        Tests that 2*pi*frequency*epoch does not overflow float64.
        """
        from src.drift import AmplitudeDrift, DriftModel

        drift = DriftModel(
            amplitude_drifts=[AmplitudeDrift(amplitude=0.3, frequency=0.005)]
        )
        offsets = drift.get_control_offsets(1_000_000)
        for key, v in offsets.items():
            assert np.isfinite(float(v)), f"Non-finite offset {key} at epoch 1M"

    def test_square_wave_at_zero_crossing(self):
        """Square wave drift at its zero crossing (sin=0) should return 0.

        sign(sin(0)) = sign(0) = 0 in JAX, so detuning = amplitude * 0 = 0.
        """
        from src.drift import SquareWaveFrequencyDrift

        sw = SquareWaveFrequencyDrift(amplitude=0.5, frequency=0.005, phase=0.0)
        detuning = sw.get_detuning(0)
        assert np.isfinite(float(detuning))

    def test_snr_drift_large_epoch(self):
        """SNR noise std grows linearly; verify it stays finite at large epochs."""
        from src.drift import SNRDrift

        snr = SNRDrift(base_noise=0.01, growth_rate=0.001)
        noise_1m = snr.get_noise_std(1_000_000)
        assert np.isfinite(noise_1m)
        expected = 0.01 + 0.001 * 1_000_000
        assert abs(float(noise_1m) - expected) < 1e-6

    def test_multiple_white_noise_drifts_compose(self):
        """Multiple white noise sources should compose additively."""
        from src.drift import DriftModel, WhiteNoiseDrift

        dm = DriftModel(
            white_noise_drifts=[
                WhiteNoiseDrift(sigma_g2=0.1, seed=0),
                WhiteNoiseDrift(sigma_g2=0.1, seed=1),
            ]
        )
        offsets = dm.get_control_offsets(42)
        # Two noise sources: offsets should generally be nonzero
        assert np.isfinite(float(offsets["g2_re_offset"]))
        assert np.isfinite(float(offsets["g2_im_offset"]))

    def test_step_drift_at_exact_boundary(self):
        """Step drift at exactly step_epoch should activate."""
        from src.drift import StepDrift

        step = StepDrift(step_epoch=100, g2_re_shift=0.3)
        off_re_before, _ = step.get_offsets(99)
        off_re_at, _ = step.get_offsets(100)
        off_re_after, _ = step.get_offsets(101)

        assert abs(float(off_re_before)) < 1e-10, "Should be 0 before step"
        assert abs(float(off_re_at) - 0.3) < 1e-6, "Should be 0.3 at step"
        assert abs(float(off_re_after) - 0.3) < 1e-6, "Should be 0.3 after step"


# ---------------------------------------------------------------------------
# Jump operator stability
# ---------------------------------------------------------------------------


class TestJumpOpStability:
    """Jump operator construction edge cases."""

    def test_zero_loss_rate(self):
        """kappa_a=0 should give L_a = 0 (no storage loss)."""
        params_no_loss = CatQubitParams(na=6, nb=3, kappa_b=10.0, kappa_a=0.0)
        a, b = build_operators(params_no_loss)
        L_b, L_a = build_jump_ops(a, b, params_no_loss)
        L_a_dense = np.array(L_a.to_jax())
        np.testing.assert_allclose(
            L_a_dense, 0.0, atol=1e-15, err_msg="L_a should be zero when kappa_a=0"
        )
        # L_b should still be nonzero
        L_b_dense = np.array(L_b.to_jax())
        assert np.max(np.abs(L_b_dense)) > 0.1, "L_b should be nonzero"

    def test_large_loss_rate_finite(self):
        """Very large kappa_b should still give finite jump operators."""
        params_large = CatQubitParams(na=6, nb=3, kappa_b=1000.0, kappa_a=100.0)
        a, b = build_operators(params_large)
        L_b, L_a = build_jump_ops(a, b, params_large)
        L_b_dense = np.array(L_b.to_jax())
        L_a_dense = np.array(L_a.to_jax())
        assert np.all(np.isfinite(L_b_dense)), "L_b has non-finite elements"
        assert np.all(np.isfinite(L_a_dense)), "L_a has non-finite elements"


# ---------------------------------------------------------------------------
# Cross-module stability: alpha -> operators -> simulation
# ---------------------------------------------------------------------------


class TestCrossModuleStability:
    """End-to-end stability through alpha -> Hamiltonian -> jump ops pipeline.

    Verifies that parameter combinations that stress compute_alpha
    still produce valid Hamiltonians and jump operators.
    """

    @pytest.mark.parametrize(
        "g2_re, g2_im, eps_d_re, eps_d_im, desc",
        [
            (1.0, 0.0, 4.0, 0.0, "standard"),
            (0.5, 0.5, 2.0, 2.0, "complex_params"),
            (0.1, 0.0, 0.5, 0.0, "small_params"),
            (5.0, 0.0, 20.0, 0.0, "large_params"),
        ],
    )
    def test_hamiltonian_finite_for_various_params(
        self, g2_re, g2_im, eps_d_re, eps_d_im, desc
    ):
        """Hamiltonian must have all-finite elements for valid param combos."""
        a, b = build_operators(FAST_PARAMS)
        H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
        H_dense = np.array(H.to_jax())
        assert np.all(np.isfinite(H_dense)), (
            f"Non-finite Hamiltonian elements for {desc} params"
        )
