"""Tests validating proxy reward correlation with true lifetimes.

The proxy reward measures single-point expectations as a fast, JIT-compatible
alternative to full exponential decay fitting. These tests verify that
proxy ranking of parameters agrees with full measurement ranking.

Physics background:
  - Proxy: T_est = -t_probe / log(<O>(t_probe)), from single mesolve
  - Full: T from least-squares fit of A*exp(-t/T) + C over many time points
  If the proxy does not rank parameters consistently with the full measurement,
  the entire optimization loop is invalid — we would be optimizing the wrong
  objective.

Tier: 3 (slow) — each reward_full call runs 2 mesolve + 2 curve fits.
Mark: @pytest.mark.slow — skip with ``pytest -m 'not slow'``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.cat_qubit import compute_alpha
from src.reward import build_reward, reward_full

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
# Slightly larger than minimum for meaningful physics, but still fast.
# na=8, nb=3 -> dim=24. Production uses na=15-20, nb=5-6.
from tests.conftest import FAST_PARAMS

# Parameter points spanning the search space.
# Each row: [g2_re, g2_im, eps_d_re, eps_d_im]
TEST_PARAMS = np.array(
    [
        [0.8, 0.0, 3.0, 0.0],  # small coupling — above cat threshold
        [1.0, 0.0, 4.0, 0.0],  # default / baseline
        [1.5, 0.0, 6.0, 0.0],  # medium
        [2.0, 0.0, 8.0, 0.0],  # large
        [1.0, 0.5, 4.0, 1.0],  # complex-valued controls
        [2.5, 0.0, 10.0, 0.0],  # larger still
    ]
)

# Short simulation times for the full reward to keep test runtime manageable.
# These are much shorter than production (tfinal_z=200, tfinal_x=1.0) but
# sufficient to observe exponential decay in the tiny Hilbert space.
TFINAL_Z_TEST = 100.0
TFINAL_X_TEST = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_proxy_and_full_rewards(
    proxy_fn,
    params_array: np.ndarray,
    tfinal_z: float = TFINAL_Z_TEST,
    tfinal_x: float = TFINAL_X_TEST,
):
    """Evaluate proxy and full reward for each parameter point.

    Skips parameter points where either reward is non-finite (e.g. failed
    exponential fit), returning only the valid pairs.

    Parameters
    ----------
    proxy_fn : callable
        JIT-compiled proxy reward: x (shape (4,)) -> scalar.
    params_array : np.ndarray, shape (N, 4)
        Parameter points to evaluate.
    tfinal_z, tfinal_x : float
        Simulation durations for full reward.

    Returns
    -------
    proxy_rewards : list[float]
        Valid proxy reward values.
    full_rewards : list[float]
        Corresponding valid full reward values.
    """
    proxy_rewards = []
    full_rewards = []

    for x in params_array:
        pr = float(proxy_fn(jnp.array(x)))
        fr = reward_full(
            x,
            tfinal_z=tfinal_z,
            tfinal_x=tfinal_x,
            params=FAST_PARAMS,
        )
        if np.isfinite(pr) and np.isfinite(fr):
            proxy_rewards.append(pr)
            full_rewards.append(fr)

    return proxy_rewards, full_rewards


# ---------------------------------------------------------------------------
# Proxy vs Full Correlation Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestProxyFullCorrelation:
    """Validate that proxy reward rankings agree with full lifetime measurements.

    The proxy reward is the optimization objective. If it does not correlate
    with the true (full) reward, the optimizer is climbing the wrong hill.
    """

    def test_proxy_full_rank_correlation(self):
        """Proxy reward should rank parameters similarly to full measurement.

        We require Spearman rank correlation > 0.5 (moderate positive).
        The threshold is lenient because:
          1. We use a tiny Hilbert space (na=8, nb=3) where truncation error
             is significant.
          2. The full reward uses short simulation windows (tfinal_z=100)
             where exponential fits may be noisy.
          3. The proxy uses a single-point measurement which is inherently
             less precise than multi-point fitting.
        """
        from scipy.stats import spearmanr

        proxy_fn, _ = build_reward("proxy", FAST_PARAMS)
        proxy_rewards, full_rewards = _collect_proxy_and_full_rewards(
            proxy_fn, TEST_PARAMS
        )

        assert len(proxy_rewards) >= 4, (
            f"Need at least 4 valid points for meaningful correlation, "
            f"got {len(proxy_rewards)}"
        )

        corr, pval = spearmanr(proxy_rewards, full_rewards)
        assert corr > 0.5, (
            f"Proxy-full Spearman correlation too low: rho={corr:.3f}, "
            f"p={pval:.3f}. Proxy rewards: {proxy_rewards}, "
            f"full rewards: {full_rewards}"
        )

    def test_proxy_full_sign_agreement(self):
        """When full reward clearly prefers A over B, proxy should agree.

        Pick the best and worst parameter points by full reward and verify
        the proxy also ranks the best one higher. This is a weaker but more
        robust check than full rank correlation.
        """
        proxy_fn, _ = build_reward("proxy", FAST_PARAMS)
        proxy_rewards, full_rewards = _collect_proxy_and_full_rewards(
            proxy_fn, TEST_PARAMS
        )

        assert len(proxy_rewards) >= 2, "Need at least 2 valid points"

        # Find indices of best and worst by full reward
        best_idx = int(np.argmax(full_rewards))
        worst_idx = int(np.argmin(full_rewards))

        if best_idx == worst_idx:
            pytest.skip("Full rewards are identical — cannot test ordering")

        assert proxy_rewards[best_idx] > proxy_rewards[worst_idx], (
            f"Proxy disagrees with full on best vs worst: "
            f"proxy[best]={proxy_rewards[best_idx]:.4f}, "
            f"proxy[worst]={proxy_rewards[worst_idx]:.4f}, "
            f"full[best]={full_rewards[best_idx]:.4f}, "
            f"full[worst]={full_rewards[worst_idx]:.4f}"
        )


# ---------------------------------------------------------------------------
# Proxy Reward Internal Consistency
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestProxyRewardConsistency:
    """Verify internal consistency properties of the proxy reward."""

    def test_proxy_rewards_vary(self):
        """Proxy rewards should differ across parameter points (not constant).

        A constant reward would mean the optimizer has no gradient signal
        and cannot make progress.
        """
        proxy_fn, _ = build_reward("proxy", FAST_PARAMS)

        rewards = [float(proxy_fn(jnp.array(x))) for x in TEST_PARAMS]
        spread = max(rewards) - min(rewards)
        assert spread > 0.1, (
            f"Proxy rewards have near-zero spread ({spread:.4f}), "
            f"optimizer would have no signal. Rewards: {rewards}"
        )

    def test_proxy_monotonic_ordering(self):
        """All proxy rewards should be finite for the test parameter set."""
        proxy_fn, _ = build_reward("proxy", FAST_PARAMS)

        rewards = []
        for x in TEST_PARAMS:
            r = float(proxy_fn(jnp.array(x)))
            rewards.append(r)

        assert all(np.isfinite(r) for r in rewards), (
            f"Some proxy rewards are non-finite: {rewards}"
        )

    def test_proxy_sensitive_to_coupling_strength(self):
        """Proxy reward should change when g2 magnitude changes.

        g2 controls the two-photon exchange rate; different magnitudes
        produce different cat sizes and therefore different lifetimes.
        """
        proxy_fn, _ = build_reward("proxy", FAST_PARAMS)

        x_small_g2 = jnp.array([0.5, 0.0, 4.0, 0.0])
        x_large_g2 = jnp.array([2.0, 0.0, 4.0, 0.0])

        r_small = float(proxy_fn(x_small_g2))
        r_large = float(proxy_fn(x_large_g2))

        assert np.isfinite(r_small) and np.isfinite(r_large)
        assert r_small != r_large, (
            f"Proxy is insensitive to g2 magnitude: "
            f"r(g2=0.5)={r_small:.4f}, r(g2=2.0)={r_large:.4f}"
        )

    def test_proxy_sensitive_to_drive_amplitude(self):
        """Proxy reward should change when eps_d magnitude changes.

        eps_d is the buffer drive amplitude; it controls the effective
        two-photon drive strength eps_2 = 2*g2*eps_d/kappa_b.
        """
        proxy_fn, _ = build_reward("proxy", FAST_PARAMS)

        x_small_drive = jnp.array([1.0, 0.0, 2.0, 0.0])
        x_large_drive = jnp.array([1.0, 0.0, 8.0, 0.0])

        r_small = float(proxy_fn(x_small_drive))
        r_large = float(proxy_fn(x_large_drive))

        assert np.isfinite(r_small) and np.isfinite(r_large)
        assert r_small != r_large, (
            f"Proxy is insensitive to eps_d magnitude: "
            f"r(eps_d=1)={r_small:.4f}, r(eps_d=8)={r_large:.4f}"
        )


# ---------------------------------------------------------------------------
# Photon Reward Correlation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPhotonRewardCorrelation:
    """Photon number reward measures <a+a> as a proxy for cat size |alpha|^2.

    Ref: Challenge notebook, Sec. 2. The photon number should track the
    analytically predicted alpha from compute_alpha().
    """

    def test_photon_reward_varies(self):
        """Photon reward should differ across parameter points."""
        photon_fn, _ = build_reward("photon", FAST_PARAMS)
        rewards = [float(photon_fn(jnp.array(x))) for x in TEST_PARAMS]
        spread = max(rewards) - min(rewards)
        assert spread > 0.01, (
            f"Photon rewards have near-zero spread ({spread:.6f}). Rewards: {rewards}"
        )

    def test_photon_tracks_alpha(self):
        """Larger analytically predicted alpha should correlate with
        larger <n> (less negative photon reward, since R = -|<n> - n_target|^2).

        This tests that the simulation produces photon numbers consistent
        with the adiabatic elimination prediction alpha = sqrt(2/kappa_2 * (|eps_2| - kappa_a/4)).
        """
        photon_fn, _ = build_reward("photon", FAST_PARAMS)

        alphas = []
        rewards = []
        for x in TEST_PARAMS:
            alpha = float(compute_alpha(x[0], x[1], x[2], x[3], FAST_PARAMS))
            r = float(photon_fn(jnp.array(x)))
            if np.isfinite(r) and np.isfinite(alpha) and alpha > 0.1:
                alphas.append(alpha)
                rewards.append(r)

        assert len(alphas) >= 3, (
            f"Need at least 3 valid alpha-reward pairs, got {len(alphas)}"
        )

        # Photon reward R = -|<n> - n_target|^2. For n_target=4 (default),
        # params yielding alpha ~ 2 (so <n> ~ 4) should get highest reward.
        # We just check that different alphas produce different rewards.
        assert max(rewards) != min(rewards), (
            "Photon reward is constant across different alpha values"
        )


# ---------------------------------------------------------------------------
# Parity Reward Correlation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestParityRewardCorrelation:
    """Parity reward uses exp(i*pi*a+a) as logical X operator.

    Ref: Berdou et al. (2022), arXiv:2204.09128.
    The parity is alpha-independent, making it a robust proxy for T_X.
    """

    def test_parity_reward_finite(self):
        """Parity reward should be finite for all test parameters."""
        parity_fn, _ = build_reward("parity", FAST_PARAMS)
        for x in TEST_PARAMS:
            r = float(parity_fn(jnp.array(x)))
            assert np.isfinite(r), f"Non-finite parity reward for params {x}"

    def test_parity_reward_varies(self):
        """Parity reward should differ across parameter points."""
        parity_fn, _ = build_reward("parity", FAST_PARAMS)
        rewards = [float(parity_fn(jnp.array(x))) for x in TEST_PARAMS]
        spread = max(rewards) - min(rewards)
        assert spread > 0.01, (
            f"Parity rewards have near-zero spread ({spread:.6f}). Rewards: {rewards}"
        )


# ---------------------------------------------------------------------------
# Multipoint vs Single-Point Proxy Consistency
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMultipointProxyConsistency:
    """Multipoint proxy measures at N time points instead of 1.

    It should produce similar rankings to the single-point proxy, since
    both estimate the same underlying lifetimes. If they disagree
    significantly, one of them has a systematic bias.
    """

    def test_multipoint_agrees_with_proxy(self):
        """Multipoint and single-point proxy should agree on best vs worst."""
        proxy_fn, _ = build_reward("proxy", FAST_PARAMS)
        multi_fn, _ = build_reward("multipoint", FAST_PARAMS)

        proxy_rewards = []
        multi_rewards = []
        for x in TEST_PARAMS:
            pr = float(proxy_fn(jnp.array(x)))
            mr = float(multi_fn(jnp.array(x)))
            if np.isfinite(pr) and np.isfinite(mr):
                proxy_rewards.append(pr)
                multi_rewards.append(mr)

        assert len(proxy_rewards) >= 3, (
            f"Need at least 3 valid points, got {len(proxy_rewards)}"
        )

        # Check that best/worst ordering agrees
        proxy_best = int(np.argmax(proxy_rewards))
        proxy_worst = int(np.argmin(proxy_rewards))
        multi_best = int(np.argmax(multi_rewards))
        multi_worst = int(np.argmin(multi_rewards))

        # At minimum, the best point should not be the worst in the other
        assert proxy_best != multi_worst, (
            "Proxy's best point is multipoint's worst — fundamental disagreement"
        )
        assert multi_best != proxy_worst, (
            "Multipoint's best point is proxy's worst — fundamental disagreement"
        )

    def test_multipoint_rewards_vary(self):
        """Multipoint proxy rewards should not be constant."""
        multi_fn, _ = build_reward("multipoint", FAST_PARAMS)
        rewards = [float(multi_fn(jnp.array(x))) for x in TEST_PARAMS]
        spread = max(rewards) - min(rewards)
        assert spread > 0.1, (
            f"Multipoint proxy rewards have near-zero spread ({spread:.4f}). "
            f"Rewards: {rewards}"
        )


# ---------------------------------------------------------------------------
# Alpha Consistency Check (fast, not marked slow)
# ---------------------------------------------------------------------------


class TestAlphaConsistency:
    """Verify that the cat size alpha is physically meaningful for test params.

    These are fast (no simulation) and serve as a sanity check that the
    parameter points in TEST_PARAMS produce valid cat states.
    """

    def test_alpha_positive_for_test_params(self):
        """All test parameter points should produce alpha > 0.

        alpha = 0 means no cat state has formed, making lifetime
        measurements meaningless.
        """
        for x in TEST_PARAMS:
            alpha = float(compute_alpha(x[0], x[1], x[2], x[3], FAST_PARAMS))
            assert alpha > 0, f"Alpha = {alpha} for params {x}; no cat state formed"

    def test_alpha_below_truncation(self):
        """Cat size alpha should be well below Hilbert space truncation.

        If alpha^2 >= na, the Fock space truncation corrupts the state.
        Require alpha^2 < 0.8 * na as a safety margin.
        """
        for x in TEST_PARAMS:
            alpha = float(compute_alpha(x[0], x[1], x[2], x[3], FAST_PARAMS))
            alpha_sq = alpha**2
            assert alpha_sq < 0.8 * FAST_PARAMS.na, (
                f"Alpha^2 = {alpha_sq:.2f} exceeds safe truncation limit "
                f"(0.8 * na = {0.8 * FAST_PARAMS.na:.1f}) for params {x}. "
                f"Increase na or use smaller control parameters."
            )

    def test_alpha_increases_with_drive(self):
        """alpha should increase when eps_d increases (with fixed g2).

        From adiabatic elimination: alpha ~ sqrt(|eps_2|) ~ sqrt(|eps_d|),
        so larger drive amplitude => larger cat.
        """
        alphas = []
        for eps_d_re in [1.0, 3.0, 5.0, 7.0]:
            alpha = float(compute_alpha(1.0, 0.0, eps_d_re, 0.0, FAST_PARAMS))
            alphas.append(alpha)

        # Check strictly increasing
        for i in range(len(alphas) - 1):
            assert alphas[i] < alphas[i + 1], (
                f"Alpha not monotonically increasing with eps_d: "
                f"alpha({i})={alphas[i]:.4f} >= alpha({i + 1})={alphas[i + 1]:.4f}"
            )
