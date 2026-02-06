"""
Statistical Significance Testing
================================
Tools for comparing experimental results with proper statistical tests.

Features:
- Welch's t-test and Mann-Whitney U tests
- Cohen's d effect size
- Bootstrap confidence intervals (BCa method)
- Multiple comparison correction (Bonferroni, Holm)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""
    mean: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int


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
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    method: str = "percentile",
    random_state: Optional[int] = None,
) -> BootstrapCI:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Input data array
        statistic: Statistic to compute ("mean", "median", "std")
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        method: CI method ("percentile" or "bca")
        random_state: Random seed for reproducibility

    Returns:
        BootstrapCI with mean and confidence bounds
    """
    if random_state is not None:
        np.random.seed(random_state)

    data = np.asarray(data)
    n = len(data)

    # Compute statistic function
    if statistic == "mean":
        stat_func = np.mean
    elif statistic == "median":
        stat_func = np.median
    elif statistic == "std":
        stat_func = np.std
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Original statistic
    original_stat = stat_func(data)

    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample = data[np.random.randint(0, n, size=n)]
        bootstrap_stats[i] = stat_func(resample)

    alpha = 1 - ci_level

    if method == "percentile":
        # Simple percentile method
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    elif method == "bca":
        # Bias-corrected and accelerated (BCa) method
        # Compute bias correction
        z0 = _norm_ppf(np.mean(bootstrap_stats < original_stat))

        # Compute acceleration (jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jackknife_sample = np.delete(data, i)
            jackknife_stats[i] = stat_func(jackknife_sample)

        jackknife_mean = np.mean(jackknife_stats)
        num = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denom = 6 * np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5
        a = num / (denom + 1e-10)

        # Compute adjusted percentiles
        z_alpha_lower = _norm_ppf(alpha / 2)
        z_alpha_upper = _norm_ppf(1 - alpha / 2)

        p_lower = _norm_cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        p_upper = _norm_cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

        ci_lower = np.percentile(bootstrap_stats, 100 * p_lower)
        ci_upper = np.percentile(bootstrap_stats, 100 * p_upper)

    else:
        raise ValueError(f"Unknown method: {method}")

    return BootstrapCI(
        mean=original_stat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )


def _norm_ppf(p: float) -> float:
    """Approximate normal percent point function (inverse CDF)."""
    # Approximation using Abramowitz and Stegun formula 26.2.23
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf

    if p < 0.5:
        sign = -1
        p = 1 - p
    else:
        sign = 1
        p = p

    t = np.sqrt(-2 * np.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    return sign * (t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3))


def _norm_cdf(x: float) -> float:
    """Approximate normal CDF."""
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


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
    use_bootstrap_ci: bool = True,
    n_bootstrap: int = 10000,
) -> ComparisonResult:
    """
    Compare two methods using appropriate statistical test.

    Uses Welch's t-test (unequal variance) by default.
    Falls back to Mann-Whitney U if normality assumption is violated.
    Optionally computes bootstrap CI for the difference.

    Args:
        results_a: Results from method A (one value per seed)
        results_b: Results from method B (one value per seed)
        method_a_name: Name of method A
        method_b_name: Name of method B
        metric_name: Name of the metric being compared
        alpha: Significance level
        use_bootstrap_ci: Whether to compute bootstrap confidence intervals
        n_bootstrap: Number of bootstrap samples

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

    # Bootstrap confidence interval for the difference
    ci_lower, ci_upper = None, None
    if use_bootstrap_ci and len(a) >= 3 and len(b) >= 3:
        diff_ci = bootstrap_difference_ci(a, b, n_bootstrap=n_bootstrap, ci_level=1 - alpha)
        ci_lower = diff_ci.ci_lower
        ci_upper = diff_ci.ci_upper

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
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def bootstrap_difference_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
) -> BootstrapCI:
    """
    Compute bootstrap CI for the difference of means.

    Args:
        a: First sample
        b: Second sample
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level

    Returns:
        BootstrapCI for the difference (a - b)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n_a, n_b = len(a), len(b)

    original_diff = np.mean(a) - np.mean(b)

    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample_a = a[np.random.randint(0, n_a, size=n_a)]
        resample_b = b[np.random.randint(0, n_b, size=n_b)]
        bootstrap_diffs[i] = np.mean(resample_a) - np.mean(resample_b)

    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return BootstrapCI(
        mean=original_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
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
