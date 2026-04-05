"""Shared test fixtures and constants for the cat qubit test suite."""

import jax.numpy as jnp
import pytest

from src.cat_qubit import CatQubitParams

# ---------------------------------------------------------------------------
# Shared constants (importable by test files)
# ---------------------------------------------------------------------------

# Standard fast test parameters (na=8 gives enough Hilbert space for
# meaningful physics while keeping tests under ~1s each).
FAST_PARAMS = CatQubitParams(na=8, nb=3, kappa_b=10.0, kappa_a=1.0)

# Default control vector: [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]
X_DEFAULT = jnp.array([1.0, 0.0, 4.0, 0.0])

# Small control vector for "bad parameter" testing
X_BAD = jnp.array([0.2, 0.0, 1.0, 0.0])

# Tiny params for convergence / benchmark tests (smaller Hilbert space)
TINY_PARAMS = CatQubitParams(na=6, nb=3, kappa_b=10.0, kappa_a=1.0)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_params():
    """Standard fast test CatQubitParams (na=8, nb=3)."""
    return FAST_PARAMS


@pytest.fixture
def x_default():
    """Default control vector [1.0, 0.0, 4.0, 0.0]."""
    return X_DEFAULT


@pytest.fixture
def x_bad():
    """Bad (small) control vector [0.2, 0.0, 1.0, 0.0]."""
    return X_BAD


@pytest.fixture
def tiny_params():
    """Tiny CatQubitParams for convergence tests (na=6, nb=3)."""
    return TINY_PARAMS
