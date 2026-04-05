"""Drift models for non-stationary cat qubit optimization.

Provides:
  - Amplitude drift: multiplicative drift on g2 coupling
  - Frequency drift: detuning term added to Hamiltonian (Delta * a†a)
  - Kerr drift: nonlinearity term (K * (a†a)^2)
  - Step drift: sudden parameter shifts
  - SNR drift: measurement noise degradation over time
  - TLS drift: two-level system defect coupling to storage mode
  - Composable DriftModel that combines multiple mechanisms

Follows the pattern from the challenge notebook (cells 21-23):
drift parameters are appended to the loss function input vector,
and the optimizer only controls the first N_KNOBS entries.

Reference:
  Sivak et al. "Reinforcement Learning Control of Quantum Error Correction."
  arXiv:2511.08493 (2025). — Drift characterization methodology.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Drift mechanism configurations
# ---------------------------------------------------------------------------


@dataclass
class AmplitudeDrift:
    """Sinusoidal drift on g2 coupling strength.

    g2_eff = g2 * (1 + amplitude * sin(2*pi*frequency*epoch + phase))

    Returns additive offsets to [g2_re, g2_im] at each epoch.
    """

    amplitude: float = 0.3  # fractional amplitude of drift
    frequency: float = 0.01  # cycles per epoch
    phase: float = 0.0  # initial phase [radians]

    def get_offsets(self, epoch: int, g2_re_base: float = 1.0, g2_im_base: float = 0.0):
        """Return (g2_re_offset, g2_im_offset) at given epoch."""
        factor = self.amplitude * jnp.sin(
            2 * jnp.pi * self.frequency * epoch + self.phase
        )
        return g2_re_base * factor, g2_im_base * factor


@dataclass
class FrequencyDrift:
    """Sinusoidal detuning of storage resonator frequency.

    Adds Delta(epoch) * a†a to the Hamiltonian.
    The optimizer can compensate with a tunable detuning knob.
    """

    amplitude: float = 0.5  # max detuning [MHz]
    frequency: float = 0.005  # cycles per epoch
    phase: float = 0.0

    def get_detuning(self, epoch: int) -> float:
        """Return detuning Delta at given epoch [MHz]."""
        return self.amplitude * jnp.sin(
            2 * jnp.pi * self.frequency * epoch + self.phase
        )


@dataclass
class SquareWaveFrequencyDrift:
    """Step-like (square wave) frequency drift on storage resonator.

    Alternates between +amplitude and -amplitude detuning.
    Tests optimizer recovery from discontinuous parameter jumps.

    delta_a(epoch) = A * sign(sin(2*pi*f*epoch + phase))
    """

    amplitude: float = 0.5  # max detuning [MHz]
    frequency: float = 0.005  # cycles per epoch
    phase: float = 0.0

    def get_detuning(self, epoch: int) -> float:
        """Return step-like detuning at given epoch [MHz]."""
        return self.amplitude * jnp.sign(
            jnp.sin(2 * jnp.pi * self.frequency * epoch + self.phase)
        )


@dataclass
class KerrDrift:
    """Time-varying Kerr nonlinearity in storage mode.

    Adds K(epoch) * (a†a)^2 to the Hamiltonian.
    """

    amplitude: float = 0.01  # max Kerr coefficient [MHz]
    frequency: float = 0.003  # cycles per epoch
    phase: float = 0.0
    baseline: float = 0.0  # constant Kerr offset

    def get_kerr(self, epoch: int) -> float:
        """Return Kerr coefficient K at given epoch [MHz]."""
        return self.baseline + self.amplitude * jnp.sin(
            2 * jnp.pi * self.frequency * epoch + self.phase
        )


@dataclass
class StepDrift:
    """Sudden step change in g2 amplitude at a given epoch.

    Useful for testing optimizer recovery time.
    """

    step_epoch: int = 100  # epoch at which step occurs
    g2_re_shift: float = 0.3  # additive shift to g2_re
    g2_im_shift: float = 0.0

    def get_offsets(self, epoch: int):
        """Return (g2_re_offset, g2_im_offset) — nonzero after step_epoch."""
        active = jnp.where(epoch >= self.step_epoch, 1.0, 0.0)
        return self.g2_re_shift * active, self.g2_im_shift * active


@dataclass
class WhiteNoiseDrift:
    """Gaussian white noise on control parameters.

    Adds i.i.d. Gaussian noise to g2 and/or eps_d offsets at each epoch,
    simulating random shot-to-shot fluctuations in hardware parameters.

    noise_g2(epoch)    ~ N(0, sigma_g2)
    noise_epsd(epoch)  ~ N(0, sigma_epsd)
    """

    sigma_g2: float = 0.05  # noise std on g2 real/imag [MHz]
    sigma_epsd: float = 0.0  # noise std on eps_d real/imag [MHz]
    seed: int = 0  # RNG seed for reproducibility

    def get_offsets(self, epoch: int):
        """Return (g2_re_noise, g2_im_noise, epsd_re_noise, epsd_im_noise)."""
        import numpy as np

        # Per-call allocation is intentional: deterministic epoch-keyed noise
        # without mutable RNG state, so get_offsets(epoch) is idempotent.
        rng = np.random.default_rng([self.seed, epoch])
        g2_re = rng.normal(0, self.sigma_g2) if self.sigma_g2 > 0 else 0.0
        g2_im = rng.normal(0, self.sigma_g2) if self.sigma_g2 > 0 else 0.0
        epsd_re = rng.normal(0, self.sigma_epsd) if self.sigma_epsd > 0 else 0.0
        epsd_im = rng.normal(0, self.sigma_epsd) if self.sigma_epsd > 0 else 0.0
        return g2_re, g2_im, epsd_re, epsd_im


@dataclass
class SNRDrift:
    """Measurement noise degradation. Applied AFTER reward evaluation.

    noise_std(epoch) = base_noise + growth_rate * epoch

    This is NOT applied inside the simulation — it's applied to the
    reward signal to simulate degrading measurement quality.

    Ref: Challenge notebook, "Drift and Noise Modeling" — "Degradation in measurement SNR"
    """

    base_noise: float = 0.01
    growth_rate: float = 0.001

    def get_noise_std(self, epoch: int) -> float:
        """Return noise standard deviation at given epoch."""
        return self.base_noise + self.growth_rate * epoch


@dataclass
class TLSDrift:
    """Two-level system coupling to storage mode.

    Adds H_TLS = omega_tls * sigma_z/2 + g_tls * (a† sigma_- + a sigma_+)
    and TLS decay jump operator sqrt(gamma_tls) * sigma_-.

    This EXTENDS the Hilbert space by x2 (tensor with qubit).
    Only active after onset_epoch.

    Ref: Challenge notebook, "Drift and Noise Modeling" — "Coupling to a TLS resonant with the storage mode"
    """

    g_tls: float = 0.1
    omega_tls: float = 0.0  # 0 = resonant with storage
    gamma_tls: float = 0.5
    onset_epoch: int = 50

    def is_active(self, epoch: int) -> bool:
        """Return True if TLS defect is active at given epoch."""
        return epoch >= self.onset_epoch


# ---------------------------------------------------------------------------
# Composable drift model
# ---------------------------------------------------------------------------


@dataclass
class DriftModel:
    """Composable drift model combining multiple mechanisms.

    Aggregates parameter offsets and Hamiltonian modifications from all
    active drift mechanisms at each epoch.
    """

    amplitude_drifts: list[AmplitudeDrift] = field(default_factory=list)
    frequency_drifts: list[FrequencyDrift] = field(default_factory=list)
    square_wave_drifts: list[SquareWaveFrequencyDrift] = field(default_factory=list)
    kerr_drifts: list[KerrDrift] = field(default_factory=list)
    step_drifts: list[StepDrift] = field(default_factory=list)
    white_noise_drifts: list[WhiteNoiseDrift] = field(default_factory=list)
    snr_drifts: list[SNRDrift] = field(default_factory=list)
    tls_drifts: list[TLSDrift] = field(default_factory=list)

    def get_control_offsets(
        self, epoch: int, current_params: dict[str, float] | None = None
    ):
        """Get total additive offsets to control parameters at this epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        current_params : dict or None
            Current control parameter values with keys ``"g2_re"`` and
            ``"g2_im"``.  When provided, amplitude drifts scale
            proportionally to the actual coupling strength.  When ``None``,
            defaults ``g2_re_base=1.0, g2_im_base=0.0`` are used (legacy
            behaviour).

        Returns
        -------
        dict with keys:
            "g2_re_offset", "g2_im_offset": offsets to g2 real/imag
            "eps_d_re_offset", "eps_d_im_offset": offsets to eps_d real/imag
        """
        g2_re_off = 0.0
        g2_im_off = 0.0

        for drift in self.amplitude_drifts:
            if current_params is not None:
                dre, dim = drift.get_offsets(
                    epoch,
                    g2_re_base=current_params["g2_re"],
                    g2_im_base=current_params["g2_im"],
                )
            else:
                dre, dim = drift.get_offsets(epoch)
            g2_re_off = g2_re_off + dre
            g2_im_off = g2_im_off + dim

        for drift in self.step_drifts:
            dre, dim = drift.get_offsets(epoch)
            g2_re_off = g2_re_off + dre
            g2_im_off = g2_im_off + dim

        eps_d_re_off = 0.0
        eps_d_im_off = 0.0

        for drift in self.white_noise_drifts:
            g2_re_n, g2_im_n, epsd_re_n, epsd_im_n = drift.get_offsets(epoch)
            g2_re_off = g2_re_off + g2_re_n
            g2_im_off = g2_im_off + g2_im_n
            eps_d_re_off = eps_d_re_off + epsd_re_n
            eps_d_im_off = eps_d_im_off + epsd_im_n

        return {
            "g2_re_offset": g2_re_off,
            "g2_im_offset": g2_im_off,
            "eps_d_re_offset": eps_d_re_off,
            "eps_d_im_offset": eps_d_im_off,
        }

    def get_hamiltonian_terms(self, epoch: int):
        """Get additional Hamiltonian terms at this epoch.

        Returns
        -------
        dict with keys:
            "detuning": coefficient for a†a term [MHz]
            "kerr": coefficient for (a†a)^2 term [MHz]
        """
        detuning = 0.0
        for drift in self.frequency_drifts:
            detuning = detuning + drift.get_detuning(epoch)
        for drift in self.square_wave_drifts:
            detuning = detuning + drift.get_detuning(epoch)

        kerr = 0.0
        for drift in self.kerr_drifts:
            kerr = kerr + drift.get_kerr(epoch)

        return {"detuning": detuning, "kerr": kerr}

    def get_snr_noise(self, epoch: int) -> float:
        """Get total SNR noise standard deviation at given epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
        float
            Sum of noise_std from all SNR drifts.
        """
        total = 0.0
        for drift in self.snr_drifts:
            total = total + drift.get_noise_std(epoch)
        return total

    def has_tls(self, epoch: int) -> bool:
        """Return True if any TLS drift is active at the given epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
        bool
            True if at least one TLS defect is active.
        """
        return any(drift.is_active(epoch) for drift in self.tls_drifts)

    def get_tls_coupling(self, epoch: int) -> tuple[float, float, float]:
        """Get effective TLS coupling parameters at given epoch.

        Sums over all active TLS drifts (typically just one).

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
        g_tls : float
            Total TLS-storage coupling strength [MHz]. 0 if no TLS active.
        omega_tls : float
            TLS detuning [MHz]. Uses the last active TLS drift's value.
        gamma_tls : float
            TLS decay rate [MHz]. Uses the last active TLS drift's value.
        """
        g_tls = 0.0
        omega_tls = 0.0
        gamma_tls = 0.0
        for drift in self.tls_drifts:
            if drift.is_active(epoch):
                g_tls += drift.g_tls
                omega_tls = drift.omega_tls
                gamma_tls = drift.gamma_tls
        return g_tls, omega_tls, gamma_tls

    def describe(self) -> str:
        """Human-readable description of active drift mechanisms."""
        parts = []
        for d in self.amplitude_drifts:
            parts.append(f"AmplitudeDrift(A={d.amplitude}, f={d.frequency})")
        for d in self.frequency_drifts:
            parts.append(f"FrequencyDrift(A={d.amplitude}, f={d.frequency})")
        for d in self.square_wave_drifts:
            parts.append(f"SquareWaveDrift(A={d.amplitude}, f={d.frequency})")
        for d in self.kerr_drifts:
            parts.append(f"KerrDrift(A={d.amplitude}, f={d.frequency})")
        for d in self.step_drifts:
            parts.append(f"StepDrift(epoch={d.step_epoch}, dg2_re={d.g2_re_shift})")
        for d in self.white_noise_drifts:
            parts.append(f"WhiteNoise(σ_g2={d.sigma_g2}, σ_εd={d.sigma_epsd})")
        for d in self.snr_drifts:
            parts.append(f"SNRDrift(base={d.base_noise}, rate={d.growth_rate})")
        for d in self.tls_drifts:
            parts.append(f"TLSDrift(g={d.g_tls}, onset={d.onset_epoch})")
        return " + ".join(parts) if parts else "No drift"


# ---------------------------------------------------------------------------
# Preset drift scenarios for benchmarking
# ---------------------------------------------------------------------------


def slow_amplitude_drift() -> DriftModel:
    """Slow sinusoidal g2 amplitude drift. Period = 200 epochs."""
    return DriftModel(amplitude_drifts=[AmplitudeDrift(amplitude=0.3, frequency=0.005)])


def fast_amplitude_drift() -> DriftModel:
    """Fast sinusoidal g2 amplitude drift. Period = 50 epochs."""
    return DriftModel(amplitude_drifts=[AmplitudeDrift(amplitude=0.3, frequency=0.02)])


def step_change_drift(step_epoch: int = 100) -> DriftModel:
    """Sudden step change in g2 at a given epoch."""
    return DriftModel(step_drifts=[StepDrift(step_epoch=step_epoch, g2_re_shift=0.3)])


def frequency_drift_scenario() -> DriftModel:
    """Slow storage frequency drift."""
    return DriftModel(frequency_drifts=[FrequencyDrift(amplitude=0.5, frequency=0.005)])


def multi_drift_scenario() -> DriftModel:
    """Combined amplitude + frequency + Kerr drift."""
    return DriftModel(
        amplitude_drifts=[AmplitudeDrift(amplitude=0.2, frequency=0.005)],
        frequency_drifts=[FrequencyDrift(amplitude=0.3, frequency=0.003)],
        kerr_drifts=[KerrDrift(amplitude=0.005, frequency=0.002)],
    )


def snr_degradation_drift() -> DriftModel:
    """SNR degradation: noise grows linearly with epoch."""
    return DriftModel(snr_drifts=[SNRDrift(base_noise=0.01, growth_rate=0.001)])


def tls_onset_drift(onset: int = 50) -> DriftModel:
    """TLS defect appears at given epoch."""
    return DriftModel(tls_drifts=[TLSDrift(onset_epoch=onset)])


def multi_drift_with_snr() -> DriftModel:
    """Combined amplitude + frequency + Kerr + SNR drift."""
    return DriftModel(
        amplitude_drifts=[AmplitudeDrift(amplitude=0.2, frequency=0.005)],
        frequency_drifts=[FrequencyDrift(amplitude=0.3, frequency=0.003)],
        kerr_drifts=[KerrDrift(amplitude=0.005, frequency=0.002)],
        snr_drifts=[SNRDrift(base_noise=0.005, growth_rate=0.0005)],
    )


def frequency_step_drift() -> DriftModel:
    """Step-like (square wave) frequency drift.

    Tests optimizer recovery from discontinuous parameter jumps.
    """
    return DriftModel(
        square_wave_drifts=[SquareWaveFrequencyDrift(amplitude=0.5, frequency=0.005)]
    )


def white_noise_drift(
    sigma_g2: float = 0.05, sigma_epsd: float = 0.0, seed: int = 0
) -> DriftModel:
    """Gaussian white noise on control parameters.

    Simulates random shot-to-shot hardware fluctuations.
    """
    return DriftModel(
        white_noise_drifts=[
            WhiteNoiseDrift(sigma_g2=sigma_g2, sigma_epsd=sigma_epsd, seed=seed)
        ]
    )
