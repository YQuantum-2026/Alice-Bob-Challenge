"""Publication-quality comparison plots (package).

Re-exports all public names for backward compatibility.
Use ``from src.plotting import plot_reward_convergence`` etc.
"""

from src.plotting._analysis import (
    plot_alpha_evolution as plot_alpha_evolution,
)
from src.plotting._analysis import (
    plot_convergence_speed as plot_convergence_speed,
)
from src.plotting._analysis import (
    plot_efficiency_scatter as plot_efficiency_scatter,
)
from src.plotting._analysis import (
    plot_pareto_frontier as plot_pareto_frontier,
)
from src.plotting._convergence import (
    plot_drift_tracking_matrix as plot_drift_tracking_matrix,
)
from src.plotting._convergence import (
    plot_lifetime_comparison as plot_lifetime_comparison,
)
from src.plotting._convergence import (
    plot_parameter_tracking as plot_parameter_tracking,
)
from src.plotting._convergence import (
    plot_reward_convergence as plot_reward_convergence,
)
from src.plotting._convergence import (
    plot_reward_type_comparison as plot_reward_type_comparison,
)
from src.plotting._convergence import (
    plot_summary_heatmap as plot_summary_heatmap,
)
from src.plotting._detail import (
    plot_drift_detail as plot_drift_detail,
)
from src.plotting._detail import (
    plot_logical_decay as plot_logical_decay,
)
from src.plotting._detail import (
    plot_run_detail as plot_run_detail,
)
from src.plotting._diagnostics import (
    plot_enhanced_vs_proxy as plot_enhanced_vs_proxy,
)
from src.plotting._diagnostics import (
    plot_reward_correlation_matrix as plot_reward_correlation_matrix,
)
from src.plotting._diagnostics import (
    plot_weight_sweep_heatmap as plot_weight_sweep_heatmap,
)
from src.plotting._moon import (
    plot_moon_cat_convergence as plot_moon_cat_convergence,
)
from src.plotting._moon import (
    plot_moon_cat_lifetimes as plot_moon_cat_lifetimes,
)
from src.plotting._style import (
    DRIFT_STYLES as DRIFT_STYLES,
)
from src.plotting._style import (
    OPTIMIZER_COLORS as OPTIMIZER_COLORS,
)
from src.plotting._style import (
    REWARD_COLORS as REWARD_COLORS,
)
from src.plotting._style import (
    set_plot_style as set_plot_style,
)
from src.plotting._dashboard import (
    plot_drift_robustness as plot_drift_robustness,
)
from src.plotting._dashboard import (
    plot_drift_scenario as plot_drift_scenario,
)
from src.plotting._dashboard import (
    plot_lifetime_scatter as plot_lifetime_scatter,
)
from src.plotting._dashboard import (
    plot_optimizer_comparison_panel as plot_optimizer_comparison_panel,
)
from src.plotting._dashboard import (
    plot_optimizer_convergence as plot_optimizer_convergence,
)
from src.plotting._dashboard import (
    plot_optimizer_detail_page as plot_optimizer_detail_page,
)
from src.plotting._dashboard import (
    plot_reward_effectiveness as plot_reward_effectiveness,
)
from src.plotting._dashboard import (
    plot_summary_dashboard as plot_summary_dashboard,
)
