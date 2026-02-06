"""
Statistical Significance Testing
================================
Tools for comparing experimental results with proper statistical tests.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two methods."""
    method_a: str
    method_b: str
    metric: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    difference: float
    p_value: float
    significant: bool
    effect_size: float
    effect_size_interpretation: str


def compute_effect_size(mean_a: float, mean_b: float, std_a: float, std_b: float, n_a: int, n_b: int) -> Tuple[float, str]:
    """
    Compute Cohen's d effect size.

    Returns:
        effect_size: Cohen's d value
        interpretation: 'negligible', 'small', 'medium', or 'large'
    """
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))

    if pooled_std < 1e-8:
        return 0.0, "negligible"

    d = abs(mean_a - mean_b) / pooled_std

    if d < 0.2:
        interpretation = "negligible"
    elif d < 0.5:
        interpretation = "small"
    elif d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return d, interpretation


def compare_methods(
    results_a: List[float],
    results_b: List[float],
    method_a_name: str = "Method A",
    method_b_name: str = "Method B",
    metric_name: str = "reward",
    alpha: float = 0.05,
) -> ComparisonResult:
    """
    Compare two methods using appropriate statistical test.

    Uses Welch's t-test (unequal variance) by default.
    Falls back to Mann-Whitney U if normality assumption is violated.

    Args:
        results_a: Results from method A (one value per seed)
        results_b: Results from method B (one value per seed)
        method_a_name: Name of method A
        method_b_name: Name of method B
        metric_name: Name of the metric being compared
        alpha: Significance level

    Returns:
        ComparisonResult with statistical analysis
    """
    try:
        from scipy.stats import ttest_ind, shapiro, mannwhitneyu
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False

    a = np.array(results_a)
    b = np.array(results_b)

    mean_a, mean_b = np.mean(a), np.mean(b)
    std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)

    if SCIPY_AVAILABLE:
        # Check normality (if enough samples)
        use_parametric = True
        if len(a) >= 8 and len(b) >= 8:
            _, p_norm_a = shapiro(a)
            _, p_norm_b = shapiro(b)
            if p_norm_a < 0.05 or p_norm_b < 0.05:
                use_parametric = False

        if use_parametric:
            # Welch's t-test (unequal variance)
            _, p_value = ttest_ind(a, b, equal_var=False)
        else:
            # Mann-Whitney U test
            _, p_value = mannwhitneyu(a, b, alternative='two-sided')
    else:
        # Simple t-test approximation without scipy
        n_a, n_b = len(a), len(b)
        se = np.sqrt(std_a**2 / n_a + std_b**2 / n_b)
        t_stat = (mean_a - mean_b) / (se + 1e-8)
        # Approximate p-value (very rough)
        df = n_a + n_b - 2
        p_value = 2 * (1 - min(0.99, abs(t_stat) / 3))  # Rough approximation

    effect_size, effect_interp = compute_effect_size(
        mean_a, mean_b, std_a, std_b, len(a), len(b)
    )

    return ComparisonResult(
        method_a=method_a_name,
        method_b=method_b_name,
        metric=metric_name,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        difference=mean_a - mean_b,
        p_value=p_value,
        significant=p_value < alpha,
        effect_size=effect_size,
        effect_size_interpretation=effect_interp,
    )


def run_significance_tests(
    results: Dict[str, Dict[str, List[float]]],
    baseline_name: str = "baseline",
    metrics: List[str] = None,
    alpha: float = 0.05,
) -> List[ComparisonResult]:
    """
    Run significance tests comparing all methods to a baseline.

    Args:
        results: Dict of method_name -> {metric_name -> [values per seed]}
        baseline_name: Name of baseline method to compare against
        metrics: List of metrics to compare (default: all)
        alpha: Significance level

    Returns:
        List of ComparisonResult objects
    """
    if baseline_name not in results:
        raise ValueError(f"Baseline '{baseline_name}' not found in results")

    baseline = results[baseline_name]
    metrics = metrics or list(baseline.keys())

    comparisons = []

    for method_name, method_results in results.items():
        if method_name == baseline_name:
            continue

        for metric in metrics:
            if metric not in method_results or metric not in baseline:
                continue

            comparison = compare_methods(
                results_a=method_results[metric],
                results_b=baseline[metric],
                method_a_name=method_name,
                method_b_name=baseline_name,
                metric_name=metric,
                alpha=alpha,
            )
            comparisons.append(comparison)

    return comparisons


def generate_results_table(
    comparisons: List[ComparisonResult],
    output_format: str = "markdown",
) -> str:
    """
    Generate formatted results table.

    Args:
        comparisons: List of comparison results
        output_format: 'markdown' or 'latex'

    Returns:
        Formatted table string
    """
    if output_format == "markdown":
        lines = [
            "| Method | Metric | Mean +/- Std | vs Baseline | p-value | Significant | Effect Size |",
            "|--------|--------|--------------|-------------|---------|-------------|-------------|",
        ]

        for c in comparisons:
            sig_mark = "*" if c.significant else ""
            lines.append(
                f"| {c.method_a} | {c.metric} | "
                f"{c.mean_a:.3f} +/- {c.std_a:.3f} | "
                f"{c.difference:+.3f} | "
                f"{c.p_value:.4f}{sig_mark} | "
                f"{'Yes' if c.significant else 'No'} | "
                f"{c.effect_size:.2f} ({c.effect_size_interpretation}) |"
            )

        return "\n".join(lines)

    elif output_format == "latex":
        lines = [
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Method & Metric & Mean $\pm$ Std & $\Delta$ & p-value & Effect \\",
            r"\midrule",
        ]

        for c in comparisons:
            sig_mark = r"$^*$" if c.significant else ""
            lines.append(
                f"{c.method_a} & {c.metric} & "
                f"${c.mean_a:.3f} \\pm {c.std_a:.3f}$ & "
                f"${c.difference:+.3f}$ & "
                f"{c.p_value:.4f}{sig_mark} & "
                f"{c.effect_size:.2f} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
        ])

        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown format: {output_format}")


def load_experiment_results(experiment_dir: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Load results from experiment directory.

    Args:
        experiment_dir: Path to experiment output directory

    Returns:
        Dict of method_name -> {metric_name -> [values per seed]}
    """
    results = {}
    exp_path = Path(experiment_dir)

    for method_dir in exp_path.iterdir():
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name
        method_results = {
            'reward': [],
            'social_welfare': [],
            'gini': [],
            'cooperation': [],
        }

        for seed_dir in method_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            results_file = seed_dir / 'results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)

                method_results['reward'].append(data.get('final_reward', 0))
                method_results['social_welfare'].append(data.get('mean_social_welfare', 0))
                method_results['gini'].append(data.get('mean_gini', 0))
                method_results['cooperation'].append(data.get('mean_cooperation', 0))

        if method_results['reward']:
            results[method_name] = method_results

    return results
