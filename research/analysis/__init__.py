"""
Analysis Tools for Contemplative MARL
=====================================
Statistical testing and result analysis.
"""

from .significance import (
    compare_methods,
    run_significance_tests,
    compute_effect_size,
    generate_results_table,
)

from .visualization import (
    plot_learning_curves,
    plot_comparison_bars,
    plot_ablation_heatmap,
)

__all__ = [
    "compare_methods",
    "run_significance_tests",
    "compute_effect_size",
    "generate_results_table",
    "plot_learning_curves",
    "plot_comparison_bars",
    "plot_ablation_heatmap",
]
