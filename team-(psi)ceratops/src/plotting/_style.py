from __future__ import annotations

import matplotlib.pyplot as plt


def set_plot_style():
    """Set matplotlib rcParams for consistent, publication-quality figures."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

OPTIMIZER_COLORS = {
    "cmaes": "#1f77b4",
    "hybrid": "#2ca02c",
    "reinforce": "#ff7f0e",
    "ppo": "#d62728",
    "bayesian": "#9467bd",
}

REWARD_COLORS = {
    "proxy": "#1f77b4",
    "photon": "#ff7f0e",
    "fidelity": "#2ca02c",
    "parity": "#d62728",
    "multipoint": "#8c564b",  # brown
    "spectral": "#e377c2",  # pink
    "enhanced_proxy": "#17becf",  # teal
    "vacuum": "#7f7f7f",  # gray — primary alpha-free reward
}

DRIFT_STYLES = {
    "none": "-",
    "amplitude_slow": "--",
    "amplitude_fast": "-.",
    "frequency": ":",
    "kerr": "--",
    "snr": "-.",
    "multi": "-",
    "step": "--",
    "white_noise": (0, (3, 1, 1, 1)),  # dash-dot-dot
    "frequency_step": "-.",
    "tls": ":",
}
