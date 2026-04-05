from __future__ import annotations

from src.visualization._pedagogical import (
    animate_cat_decay,
    animate_driven_two_photon,
    animate_full_cat_qubit,
    animate_simple_two_photon,
)
from src.visualization._static import (
    plot_cat_states,
    plot_lifetimes_pair,
    plot_wigner_snapshots,
)

# ---------------------------------------------------------------------------
# Generate all challenge-replication figures
# ---------------------------------------------------------------------------


def generate_challenge_figures(
    g2_re: float = 1.0,
    g2_im: float = 0.0,
    eps_d_re: float = 4.0,
    eps_d_im: float = 0.0,
) -> list[str]:
    """Generate all challenge-replication visualizations.

    Produces:
    - 4 Wigner animations (simple, driven, full, decay)
    - 1 lifetime decay pair plot (T_Z + T_X)
    - 1 Wigner snapshot grid
    - 1 even/odd cat state plot

    Parameters
    ----------
    g2_re, g2_im, eps_d_re, eps_d_im : float
        Default control parameters.

    Returns
    -------
    list[str]
        Paths to all generated files.
    """
    paths = []

    print("[challenge] Generating challenge-replication figures...")

    # Static plots
    paths.append(plot_cat_states())
    paths.append(plot_wigner_snapshots(g2_re, g2_im, eps_d_re, eps_d_im))
    paths.append(plot_lifetimes_pair(g2_re, g2_im, eps_d_re, eps_d_im))

    # Animations
    paths.append(animate_simple_two_photon())
    paths.append(animate_driven_two_photon())
    paths.append(
        animate_full_cat_qubit(
            g2_re,
            g2_im,
            eps_d_re,
            eps_d_im,
            kappa_a=0.0,
        )
    )
    paths.append(
        animate_cat_decay(
            g2_re,
            g2_im,
            eps_d_re,
            eps_d_im,
            kappa_a=1.0,
        )
    )

    print(f"[challenge] Done. Generated {len(paths)} files.")
    return paths
