"""Reward functions for cat qubit online optimization (package).

Re-exports all public and semi-public names for backward compatibility.
Use ``from src.reward import build_reward`` etc.

Provides:
  - Full measurement reward (T_X/T_Z via exponential fits — slow, for validation)
  - Proxy reward (single-point expectations — fast, JIT-compatible, differentiable)
  - Batched versions for CMA-ES population evaluation
"""

# Shared helpers (semi-public: some used by moon_cat.py and tests)
from src.reward._enhanced import _build_enhanced_proxy_fn as _build_enhanced_proxy_fn

# Factory dispatchers (primary public API)
from src.reward._factory import (
    _NON_JIT_REWARDS as _NON_JIT_REWARDS,
)
from src.reward._factory import (
    build_drift_aware_reward as build_drift_aware_reward,
)
from src.reward._factory import (
    build_reward as build_reward,
)
from src.reward._helpers import (
    _compute_lifetime_score as _compute_lifetime_score,
)
from src.reward._helpers import (
    _estimate_T_from_log_derivative as _estimate_T_from_log_derivative,
)
from src.reward._helpers import (
    _estimate_T_from_trace as _estimate_T_from_trace,
)
from src.reward._helpers import (
    _estimate_T_single_point as _estimate_T_single_point,
)
from src.reward._helpers import (
    _make_time_grid as _make_time_grid,
)
from src.reward._helpers import (
    reward_full as reward_full,
)
from src.reward._parity import _build_parity_reward_fn as _build_parity_reward_fn

# Proxy reward convenience wrappers
# Individual builders (internal, but re-exported for direct access)
from src.reward._proxy import (
    _build_multipoint_proxy_fn as _build_multipoint_proxy_fn,
)
from src.reward._proxy import (
    _build_proxy_loss_fn as _build_proxy_loss_fn,
)
from src.reward._proxy import (
    build_cmaes_loss as build_cmaes_loss,
)
from src.reward._proxy import (
    build_proxy_reward as build_proxy_reward,
)
from src.reward._simple import (
    _build_fidelity_reward_fn as _build_fidelity_reward_fn,
)
from src.reward._simple import (
    _build_photon_reward_fn as _build_photon_reward_fn,
)
from src.reward._spectral import (
    _build_spectral_reward_fn as _build_spectral_reward_fn,
)
from src.reward._vacuum import _build_vacuum_reward_fn as _build_vacuum_reward_fn
