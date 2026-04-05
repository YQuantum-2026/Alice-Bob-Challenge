"""Tests for src/drift.py — all drift models.

Validates drift value correctness, composition, and presets.

References:
  Amplitude: Challenge notebook, "Drift and Noise Modeling"
  Frequency: Storage detuning — challenge notebook
  Kerr: Nonlinearity — challenge notebook
  TLS: Two-level system defect coupling — challenge notebook
  SNR: Measurement noise degradation — challenge notebook
"""

import jax.numpy as jnp
import pytest

from src.drift import (
    AmplitudeDrift,
    DriftModel,
    FrequencyDrift,
    KerrDrift,
    SNRDrift,
    SquareWaveFrequencyDrift,
    StepDrift,
    TLSDrift,
    WhiteNoiseDrift,
    fast_amplitude_drift,
    frequency_drift_scenario,
    frequency_step_drift,
    multi_drift_scenario,
    multi_drift_with_snr,
    slow_amplitude_drift,
    snr_degradation_drift,
    step_change_drift,
    tls_onset_drift,
    white_noise_drift,
)


class TestAmplitudeDrift:
    """Sinusoidal g2 coupling drift."""

    def test_zero_at_epoch_zero(self):
        d = AmplitudeDrift(amplitude=0.3, frequency=0.01, phase=0.0)
        offset_re, offset_im = d.get_offsets(0)
        assert abs(float(offset_re)) < 1e-10

    def test_correct_period(self):
        """Period = 1/frequency = 100 epochs."""
        d = AmplitudeDrift(amplitude=0.3, frequency=0.01, phase=0.0)
        off_0, _ = d.get_offsets(0)
        off_100, _ = d.get_offsets(100)
        assert jnp.isclose(off_0, off_100, atol=1e-6)


class TestFrequencyDrift:
    """Storage resonator detuning."""

    def test_returns_float(self):
        d = FrequencyDrift(amplitude=0.5, frequency=0.005)
        delta = d.get_detuning(50)
        assert jnp.isfinite(delta)

    def test_amplitude_bounded(self):
        d = FrequencyDrift(amplitude=0.5, frequency=0.005)
        for ep in range(200):
            delta = d.get_detuning(ep)
            assert abs(float(delta)) <= 0.5 + 1e-6


class TestSquareWaveFrequencyDrift:
    """Step-like (square wave) frequency drift."""

    def test_returns_finite(self):
        d = SquareWaveFrequencyDrift(amplitude=0.5, frequency=0.005)
        delta = d.get_detuning(50)
        assert jnp.isfinite(delta)

    def test_amplitude_bounded(self):
        d = SquareWaveFrequencyDrift(amplitude=0.5, frequency=0.005)
        for ep in range(200):
            delta = d.get_detuning(ep)
            assert abs(float(delta)) <= 0.5 + 1e-6

    def test_is_square_wave(self):
        """Output should be exactly +amplitude or -amplitude (or 0 at crossings)."""
        d = SquareWaveFrequencyDrift(amplitude=0.5, frequency=0.005)
        for ep in [10, 50, 150]:
            delta = float(d.get_detuning(ep))
            assert (
                delta == pytest.approx(0.5, abs=1e-6)
                or delta == pytest.approx(-0.5, abs=1e-6)
                or delta == pytest.approx(0.0, abs=1e-6)
            )


class TestKerrDrift:
    """Kerr nonlinearity."""

    def test_returns_float(self):
        d = KerrDrift(amplitude=0.01, frequency=0.003)
        k = d.get_kerr(50)
        assert jnp.isfinite(k)


class TestStepDrift:
    """Step change in g2 parameters."""

    def test_zero_before_step(self):
        d = StepDrift(step_epoch=100, g2_re_shift=0.3)
        off_re, off_im = d.get_offsets(99)
        assert abs(float(off_re)) < 1e-6

    def test_nonzero_after_step(self):
        d = StepDrift(step_epoch=100, g2_re_shift=0.3)
        off_re, off_im = d.get_offsets(100)
        assert abs(float(off_re) - 0.3) < 1e-6


class TestSNRDrift:
    """Measurement noise degradation."""

    def test_noise_increases(self):
        d = SNRDrift(base_noise=0.01, growth_rate=0.001)
        n0 = d.get_noise_std(0)
        n100 = d.get_noise_std(100)
        assert n100 > n0

    def test_base_value(self):
        d = SNRDrift(base_noise=0.01, growth_rate=0.001)
        assert abs(d.get_noise_std(0) - 0.01) < 1e-10


class TestTLSDrift:
    """TLS defect coupling."""

    def test_inactive_before_onset(self):
        d = TLSDrift(onset_epoch=50)
        assert not d.is_active(49)

    def test_active_after_onset(self):
        d = TLSDrift(onset_epoch=50)
        assert d.is_active(50)
        assert d.is_active(100)

    def test_tls_operators_dimensions(self):
        """TLS-extended operators have correct dimensions (na*nb*2)."""
        from src.cat_qubit import CatQubitParams, build_tls_operators

        params = CatQubitParams(na=5, nb=3, kappa_b=10.0, kappa_a=1.0)
        a_ext, b_ext, sigma_z, sigma_m = build_tls_operators(params)
        n_ext = params.na * params.nb * 2
        assert a_ext.shape == (n_ext, n_ext)
        assert b_ext.shape == (n_ext, n_ext)
        assert sigma_z.shape == (n_ext, n_ext)
        assert sigma_m.shape == (n_ext, n_ext)

    def test_tls_hamiltonian_hermitian(self):
        """TLS Hamiltonian term must be Hermitian."""
        import numpy as np

        from src.cat_qubit import (
            CatQubitParams,
            build_tls_hamiltonian_term,
            build_tls_operators,
        )

        params = CatQubitParams(na=5, nb=3, kappa_b=10.0, kappa_a=1.0)
        a_ext, _, sigma_z, sigma_m = build_tls_operators(params)
        H_tls = build_tls_hamiltonian_term(
            a_ext, sigma_z, sigma_m, g_tls=0.1, omega_tls=0.5
        )
        H_dense = H_tls.to_jax()
        np.testing.assert_allclose(
            H_dense, H_dense.conj().T, atol=1e-12, err_msg="H_TLS not Hermitian"
        )

    def test_tls_zero_coupling_recovers_standard(self):
        """With g_tls=0, TLS Hamiltonian term is just omega_tls*sigma_z/2."""
        import jax.numpy as jnp
        import numpy as np

        from src.cat_qubit import (
            CatQubitParams,
            build_tls_hamiltonian_term,
            build_tls_operators,
        )

        params = CatQubitParams(na=5, nb=3, kappa_b=10.0, kappa_a=1.0)
        a_ext, _, sigma_z, sigma_m = build_tls_operators(params)
        H_tls = build_tls_hamiltonian_term(
            a_ext, sigma_z, sigma_m, g_tls=0.0, omega_tls=1.0
        )
        H_dense = H_tls.to_jax()
        # Should just be omega_tls/2 * sigma_z (no coupling)
        expected = 0.5 * sigma_z.to_jax()
        np.testing.assert_allclose(jnp.abs(H_dense - expected), 0, atol=1e-12)

    def test_tls_jump_ops_count(self):
        """TLS jump ops should include buffer, storage, and TLS decay."""
        from src.cat_qubit import (
            CatQubitParams,
            build_tls_jump_ops,
            build_tls_operators,
        )

        params = CatQubitParams(na=5, nb=3, kappa_b=10.0, kappa_a=1.0)
        a_ext, b_ext, _, sigma_m = build_tls_operators(params)
        jump_ops = build_tls_jump_ops(a_ext, b_ext, sigma_m, params, gamma_tls=0.5)
        assert len(jump_ops) == 3  # L_b, L_a, L_tls

    def test_drift_model_get_tls_coupling(self):
        """DriftModel.get_tls_coupling returns correct values."""
        dm = DriftModel(
            tls_drifts=[
                TLSDrift(g_tls=0.2, omega_tls=0.5, gamma_tls=0.3, onset_epoch=10)
            ]
        )
        # Before onset
        g, omega, gamma = dm.get_tls_coupling(5)
        assert g == 0.0
        # After onset
        g, omega, gamma = dm.get_tls_coupling(10)
        assert abs(g - 0.2) < 1e-10
        assert abs(omega - 0.5) < 1e-10
        assert abs(gamma - 0.3) < 1e-10


class TestWhiteNoiseDrift:
    """Gaussian white noise on control parameters."""

    def test_returns_four_values(self):
        d = WhiteNoiseDrift(sigma_g2=0.1, sigma_epsd=0.05, seed=0)
        g2r, g2i, er, ei = d.get_offsets(0)
        assert all(isinstance(v, float) for v in [g2r, g2i, er, ei])

    def test_different_epochs_different_noise(self):
        d = WhiteNoiseDrift(sigma_g2=0.1, seed=42)
        off0 = d.get_offsets(0)
        off1 = d.get_offsets(1)
        assert off0[0] != off1[0], "Different epochs should give different noise"

    def test_same_epoch_reproducible(self):
        d = WhiteNoiseDrift(sigma_g2=0.1, seed=42)
        off_a = d.get_offsets(10)
        off_b = d.get_offsets(10)
        assert off_a == off_b, "Same seed+epoch should be reproducible"

    def test_zero_sigma_gives_zero(self):
        d = WhiteNoiseDrift(sigma_g2=0.0, sigma_epsd=0.0, seed=0)
        g2r, g2i, er, ei = d.get_offsets(50)
        assert g2r == 0.0 and g2i == 0.0 and er == 0.0 and ei == 0.0

    def test_composable_in_drift_model(self):
        dm = DriftModel(white_noise_drifts=[WhiteNoiseDrift(sigma_g2=0.1, seed=0)])
        offsets = dm.get_control_offsets(5)
        assert offsets["g2_re_offset"] != 0.0


class TestDriftModel:
    """Composable drift model aggregation."""

    def test_empty_model(self):
        dm = DriftModel()
        offsets = dm.get_control_offsets(50)
        assert offsets["g2_re_offset"] == 0.0

    def test_snr_noise(self):
        dm = DriftModel(snr_drifts=[SNRDrift(base_noise=0.01, growth_rate=0.001)])
        assert dm.get_snr_noise(0) == pytest.approx(0.01)
        assert dm.get_snr_noise(100) == pytest.approx(0.11)

    def test_has_tls(self):
        dm = DriftModel(tls_drifts=[TLSDrift(onset_epoch=50)])
        assert not dm.has_tls(49)
        assert dm.has_tls(50)

    def test_describe(self):
        dm = multi_drift_with_snr()
        desc = dm.describe()
        assert "AmplitudeDrift" in desc
        assert "SNRDrift" in desc

    def test_composition(self):
        """Multiple drift types compose correctly."""
        dm = DriftModel(
            amplitude_drifts=[AmplitudeDrift(amplitude=0.3, frequency=0.01)],
            snr_drifts=[SNRDrift(base_noise=0.02)],
        )
        offsets = dm.get_control_offsets(25)  # quarter period
        assert abs(float(offsets["g2_re_offset"])) > 0.01
        assert dm.get_snr_noise(0) == pytest.approx(0.02)


class TestPresets:
    """Preset drift scenarios return valid DriftModels."""

    def test_slow_amplitude(self):
        dm = slow_amplitude_drift()
        assert len(dm.amplitude_drifts) == 1

    def test_fast_amplitude(self):
        dm = fast_amplitude_drift()
        assert dm.amplitude_drifts[0].frequency == 0.02

    def test_step_change(self):
        dm = step_change_drift(100)
        assert len(dm.step_drifts) == 1

    def test_frequency(self):
        dm = frequency_drift_scenario()
        assert len(dm.frequency_drifts) == 1

    def test_multi(self):
        dm = multi_drift_scenario()
        assert len(dm.amplitude_drifts) > 0
        assert len(dm.frequency_drifts) > 0

    def test_snr(self):
        dm = snr_degradation_drift()
        assert len(dm.snr_drifts) == 1

    def test_tls(self):
        dm = tls_onset_drift(50)
        assert len(dm.tls_drifts) == 1

    def test_multi_with_snr(self):
        dm = multi_drift_with_snr()
        assert len(dm.snr_drifts) > 0

    def test_white_noise(self):
        dm = white_noise_drift(sigma_g2=0.1, sigma_epsd=0.05)
        assert len(dm.white_noise_drifts) == 1
        assert dm.white_noise_drifts[0].sigma_g2 == 0.1

    def test_frequency_step(self):
        dm = frequency_step_drift()
        assert len(dm.square_wave_drifts) == 1


class TestAmplitudeDriftPhysics:
    """Verify amplitude drift scales with actual g2 values."""

    def test_imaginary_drift_nonzero_when_g2_im_nonzero(self):
        """With nonzero g2_im_base, imaginary drift must be nonzero at non-zero epoch."""
        d = AmplitudeDrift(amplitude=0.3, frequency=0.01, phase=0.0)
        # epoch 25 is quarter-period -> sin(pi/2) = 1
        _, offset_im = d.get_offsets(25, g2_re_base=1.0, g2_im_base=0.5)
        assert abs(float(offset_im)) > 0.1, "Imaginary drift should be nonzero"

    def test_imaginary_drift_zero_with_default_base(self):
        """Default g2_im_base=0 produces zero imaginary drift (legacy behaviour)."""
        d = AmplitudeDrift(amplitude=0.3, frequency=0.01, phase=0.0)
        _, offset_im = d.get_offsets(25)
        assert float(offset_im) == 0.0

    def test_drift_scales_with_g2_magnitude(self):
        """Drift offset should scale proportionally to g2 base value."""
        d = AmplitudeDrift(amplitude=0.3, frequency=0.01, phase=0.0)
        off_small, _ = d.get_offsets(25, g2_re_base=1.0, g2_im_base=0.0)
        off_large, _ = d.get_offsets(25, g2_re_base=2.0, g2_im_base=0.0)
        assert jnp.isclose(off_large, 2.0 * off_small, atol=1e-10)

    def test_drift_model_forwards_current_params(self):
        """DriftModel.get_control_offsets passes current_params to AmplitudeDrift."""
        dm = DriftModel(
            amplitude_drifts=[AmplitudeDrift(amplitude=0.3, frequency=0.01)]
        )
        # With current_params having nonzero g2_im
        offsets = dm.get_control_offsets(
            25, current_params={"g2_re": 1.0, "g2_im": 0.5}
        )
        assert abs(float(offsets["g2_im_offset"])) > 0.1

        # Without current_params -> legacy (g2_im_base=0)
        offsets_legacy = dm.get_control_offsets(25)
        assert float(offsets_legacy["g2_im_offset"]) == 0.0


# ---------------------------------------------------------------------------
# build_drift_model factory
# ---------------------------------------------------------------------------


class TestBuildDriftModel:
    """Validate the build_drift_model factory for all named drift scenarios."""

    @pytest.mark.parametrize(
        "drift_name",
        [
            "none",
            "amplitude_slow",
            "amplitude_fast",
            "frequency",
            "kerr",
            "frequency_step",
            "step",
            "snr",
            "multi",
            "white_noise",
            "tls",
        ],
    )
    def test_valid_drift_names_return_drift_model(self, drift_name):
        from src.config import DriftConfig, build_drift_model
        from src.drift import DriftModel

        model = build_drift_model(drift_name, DriftConfig())
        assert isinstance(model, DriftModel)

    def test_unknown_drift_raises(self):
        from src.config import DriftConfig, build_drift_model

        with pytest.raises(ValueError, match="Unknown drift"):
            build_drift_model("nonexistent_drift", DriftConfig())
