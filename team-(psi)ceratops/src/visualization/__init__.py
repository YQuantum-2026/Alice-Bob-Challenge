"""Wigner function animations and phase-space visualization (package).

Re-exports all public names for backward compatibility.
Use ``from src.visualization import animate_wigner`` etc.
"""

from src.visualization._challenge import (
    generate_challenge_figures as generate_challenge_figures,
)
from src.visualization._moon_wigner import (
    plot_moon_cat_wigner as plot_moon_cat_wigner,
)
from src.visualization._pedagogical import (
    animate_cat_decay as animate_cat_decay,
)
from src.visualization._pedagogical import (
    animate_driven_two_photon as animate_driven_two_photon,
)
from src.visualization._pedagogical import (
    animate_full_cat_qubit as animate_full_cat_qubit,
)
from src.visualization._pedagogical import (
    animate_simple_two_photon as animate_simple_two_photon,
)
from src.visualization._static import (
    plot_cat_states as plot_cat_states,
)
from src.visualization._static import (
    plot_lifetime_decay as plot_lifetime_decay,
)
from src.visualization._static import (
    plot_lifetimes_pair as plot_lifetimes_pair,
)
from src.visualization._static import (
    plot_wigner_snapshots as plot_wigner_snapshots,
)
from src.visualization._vacuum import (
    animate_vacuum_wigner as animate_vacuum_wigner,
)
from src.visualization._vacuum import (
    plot_vacuum_diagnostics as plot_vacuum_diagnostics,
)
from src.visualization._wigner import (
    animate_alpha_sweep as animate_alpha_sweep,
)
from src.visualization._wigner import (
    animate_before_after as animate_before_after,
)
from src.visualization._wigner import (
    animate_wigner as animate_wigner,
)
from src.visualization._wigner import (
    generate_all_animations as generate_all_animations,
)
