"""
Analysis Tools for Contemplative MARL
=====================================
Comprehensive statistical testing, calibration, communication analysis,
adversarial robustness, sample efficiency, human evaluation, and
transfer learning evaluation.
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

from .statistical_tests import (
    StatisticalAnalyzer,
    PairwiseComparison,
    FactorialAnalysis,
)

from .calibration import CalibrationAnalyzer
from .communication_analysis import CommunicationAnalyzer
from .adversarial_robustness import AdversarialRobustnessEvaluator
from .sample_efficiency import SampleEfficiencyAnalyzer
from .human_evaluation import HumanEvaluationProtocol
from .transfer_learning import TransferLearningEvaluator
from .diffusion_analysis import DiffusionPaperAnalyzer, DiffusionFieldMetrics

__all__ = [
    # Original
    "compare_methods",
    "run_significance_tests",
    "compute_effect_size",
    "generate_results_table",
    "plot_learning_curves",
    "plot_comparison_bars",
    "plot_ablation_heatmap",
    # New statistical analysis
    "StatisticalAnalyzer",
    "PairwiseComparison",
    "FactorialAnalysis",
    # New analysis modules
    "CalibrationAnalyzer",
    "CommunicationAnalyzer",
    "AdversarialRobustnessEvaluator",
    "SampleEfficiencyAnalyzer",
    "HumanEvaluationProtocol",
    "TransferLearningEvaluator",
    # Diffusion paper analysis
    "DiffusionPaperAnalyzer",
    "DiffusionFieldMetrics",
]
