"""
Comprehensive Statistical Analysis Suite
=========================================
Publication-grade statistical testing for Contemplative MARL experiments.

Includes:
- Welch's t-test and Mann-Whitney U (pairwise)
- Holm-Bonferroni multiple comparison correction
- Full 2^3 factorial ANOVA with interaction effects
- Cohen's d and Hedge's g effect sizes
- BCa bootstrap confidence intervals
- Bayesian estimation (BEST alternative to t-test)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from itertools import combinations
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PairwiseComparison:
    """Result of a pairwise statistical comparison."""
    method_a: str
    method_b: str
    metric: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    se_a: float
    se_b: float
    n_a: int
    n_b: int
    difference: float
    ci_lower: float
    ci_upper: float
    p_value_raw: float
    p_value_corrected: Optional[float]
    significant_raw: bool
    significant_corrected: bool
    test_used: str
    effect_size_cohens_d: float
    effect_size_hedges_g: float
    effect_interpretation: str


@dataclass
class FactorialResult:
    """Result of 2^k factorial ANOVA analysis."""
    main_effects: Dict[str, Dict[str, float]]
    interaction_effects: Dict[str, Dict[str, float]]
    full_model_r_squared: float
    residual_variance: float
    condition_means: Dict[str, float]
    condition_stds: Dict[str, float]


@dataclass
class BayesianEstimate:
    """Bayesian estimation result (BEST)."""
    posterior_mean_diff: float
    hdi_lower: float
    hdi_upper: float
    prob_superior: float
    effect_size_posterior_mean: float
    rope_lower: float
    rope_upper: float
    prob_in_rope: float


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for multi-condition experiments.

    Performs pairwise comparisons, multiple correction, factorial analysis,
    and Bayesian estimation on experimental results.
    """

    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap

    def run_full_analysis(
        self,
        results: Dict[str, Dict[str, List[float]]],
        baseline_name: str = "no_modules",
        metrics: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run complete statistical analysis pipeline.

        Args:
            results: {condition_name: {metric_name: [values_per_seed]}}
            baseline_name: Name of the baseline condition
            metrics: Metrics to analyze (default: all available)

        Returns:
            Dict with pairwise comparisons, factorial analysis, and summary
        """
        if metrics is None:
            all_metrics = set()
            for cond_results in results.values():
                all_metrics.update(cond_results.keys())
            metrics = sorted(all_metrics)

        # 1. Pairwise comparisons against baseline
        pairwise = self._run_pairwise_comparisons(results, baseline_name, metrics)

        # 2. All-pairs comparisons with Holm-Bonferroni
        all_pairs = self._run_all_pairs_comparisons(results, metrics)

        # 3. Factorial analysis (if factorial design detected)
        factorial = self._run_factorial_analysis(results, metrics)

        # 4. Summary statistics with bootstrap CIs
        summary = self._compute_summary_statistics(results, metrics)

        return {
            "pairwise_vs_baseline": pairwise,
            "all_pairs_corrected": all_pairs,
            "factorial_analysis": factorial,
            "summary_statistics": summary,
            "analysis_config": {
                "alpha": self.alpha,
                "n_bootstrap": self.n_bootstrap,
                "baseline": baseline_name,
                "metrics": metrics,
                "n_conditions": len(results),
            },
        }

    def _run_pairwise_comparisons(
        self,
        results: Dict[str, Dict[str, List[float]]],
        baseline_name: str,
        metrics: List[str],
    ) -> List[PairwiseComparison]:
        """Run pairwise comparisons against baseline with Holm-Bonferroni correction."""
        if baseline_name not in results:
            logger.warning(f"Baseline '{baseline_name}' not in results, skipping pairwise")
            return []

        comparisons = []
        baseline_data = results[baseline_name]

        for condition_name, cond_data in results.items():
            if condition_name == baseline_name:
                continue
            for metric in metrics:
                if metric not in cond_data or metric not in baseline_data:
                    continue
                comp = self._compare_two_samples(
                    cond_data[metric], baseline_data[metric],
                    condition_name, baseline_name, metric
                )
                comparisons.append(comp)

        # Apply Holm-Bonferroni correction
        self._apply_holm_bonferroni(comparisons)

        return comparisons

    def _run_all_pairs_comparisons(
        self,
        results: Dict[str, Dict[str, List[float]]],
        metrics: List[str],
    ) -> List[PairwiseComparison]:
        """Run all-pairs comparisons with Holm-Bonferroni correction."""
        comparisons = []
        condition_names = list(results.keys())

        for (name_a, name_b) in combinations(condition_names, 2):
            for metric in metrics:
                data_a = results[name_a].get(metric, [])
                data_b = results[name_b].get(metric, [])
                if not data_a or not data_b:
                    continue
                comp = self._compare_two_samples(
                    data_a, data_b, name_a, name_b, metric
                )
                comparisons.append(comp)

        self._apply_holm_bonferroni(comparisons)
        return comparisons

    def _compare_two_samples(
        self,
        data_a: List[float],
        data_b: List[float],
        name_a: str,
        name_b: str,
        metric: str,
    ) -> PairwiseComparison:
        """Compare two samples using appropriate statistical test."""
        a = np.array(data_a, dtype=np.float64)
        b = np.array(data_b, dtype=np.float64)

        mean_a, mean_b = np.mean(a), np.mean(b)
        std_a = np.std(a, ddof=1) if len(a) > 1 else 0.0
        std_b = np.std(b, ddof=1) if len(b) > 1 else 0.0
        se_a = std_a / np.sqrt(len(a))
        se_b = std_b / np.sqrt(len(b))

        # Choose test based on normality
        test_used, p_value = self._select_and_run_test(a, b)

        # Effect sizes
        cohens_d = self._cohens_d(mean_a, mean_b, std_a, std_b, len(a), len(b))
        hedges_g = self._hedges_g(cohens_d, len(a), len(b))
        interpretation = self._interpret_effect_size(abs(hedges_g))

        # Bootstrap CI for difference
        ci_lower, ci_upper = self._bootstrap_difference_ci(a, b)

        return PairwiseComparison(
            method_a=name_a,
            method_b=name_b,
            metric=metric,
            mean_a=float(mean_a),
            mean_b=float(mean_b),
            std_a=float(std_a),
            std_b=float(std_b),
            se_a=float(se_a),
            se_b=float(se_b),
            n_a=len(a),
            n_b=len(b),
            difference=float(mean_a - mean_b),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value_raw=float(p_value),
            p_value_corrected=None,
            significant_raw=p_value < self.alpha,
            significant_corrected=False,
            test_used=test_used,
            effect_size_cohens_d=float(cohens_d),
            effect_size_hedges_g=float(hedges_g),
            effect_interpretation=interpretation,
        )

    def _select_and_run_test(
        self, a: np.ndarray, b: np.ndarray
    ) -> Tuple[str, float]:
        """Select and run appropriate statistical test."""
        try:
            from scipy.stats import ttest_ind, mannwhitneyu, shapiro
            use_parametric = True
            if len(a) >= 8 and len(b) >= 8:
                _, p_a = shapiro(a)
                _, p_b = shapiro(b)
                if p_a < 0.05 or p_b < 0.05:
                    use_parametric = False

            if use_parametric:
                _, p = ttest_ind(a, b, equal_var=False)
                return "welch_t_test", float(p)
            else:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
                return "mann_whitney_u", float(p)
        except ImportError:
            # Fallback: simple Welch's t-test
            p = self._welch_t_test_fallback(a, b)
            return "welch_t_test_approx", float(p)

    def _welch_t_test_fallback(self, a: np.ndarray, b: np.ndarray) -> float:
        """Welch's t-test without scipy."""
        n_a, n_b = len(a), len(b)
        var_a = np.var(a, ddof=1) if n_a > 1 else 1e-8
        var_b = np.var(b, ddof=1) if n_b > 1 else 1e-8
        se = np.sqrt(var_a / n_a + var_b / n_b)
        if se < 1e-12:
            return 1.0
        t_stat = (np.mean(a) - np.mean(b)) / se
        # Welch-Satterthwaite degrees of freedom
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = num / (denom + 1e-12)
        # Approximate p-value using normal for large df
        p = 2 * (1 - self._norm_cdf(abs(t_stat)))
        return max(p, 1e-300)

    def _apply_holm_bonferroni(self, comparisons: List[PairwiseComparison]):
        """Apply Holm-Bonferroni step-down correction for multiple comparisons."""
        if not comparisons:
            return

        m = len(comparisons)
        # Sort by raw p-value
        sorted_indices = sorted(range(m), key=lambda i: comparisons[i].p_value_raw)

        for rank, idx in enumerate(sorted_indices):
            adjusted_alpha = self.alpha / (m - rank)
            corrected_p = min(
                comparisons[idx].p_value_raw * (m - rank), 1.0
            )
            comparisons[idx].p_value_corrected = corrected_p
            comparisons[idx].significant_corrected = corrected_p < self.alpha

    def _run_factorial_analysis(
        self,
        results: Dict[str, Dict[str, List[float]]],
        metrics: List[str],
    ) -> Optional[Dict[str, FactorialResult]]:
        """
        Run 2^3 factorial analysis for ethics x diffusion x mindfulness design.

        Detects if results contain a factorial design and computes
        main effects, interaction effects, and R-squared.
        """
        # Map conditions to factor levels
        factor_map = self._detect_factorial_design(results)
        if factor_map is None:
            return None

        factorial_results = {}
        for metric in metrics:
            result = self._compute_factorial_effects(results, factor_map, metric)
            if result is not None:
                factorial_results[metric] = result

        return factorial_results if factorial_results else None

    def _detect_factorial_design(
        self, results: Dict[str, Dict[str, List[float]]]
    ) -> Optional[Dict[str, Tuple[bool, bool, bool]]]:
        """
        Detect if conditions form a 2^3 factorial design.

        Maps condition names to (ethics_on, diffusion_on, mindfulness_on) tuples.
        """
        factor_map = {}
        condition_names = list(results.keys())

        known_patterns = {
            # Full 8-condition factorial
            "full": (True, True, True),
            "contemplative_full": (True, True, True),
            "ethics_only": (True, False, False),
            "diffusion_only": (False, True, False),
            "mindfulness_only": (False, False, True),
            "ethics_diffusion": (True, True, False),
            "ethics_mindfulness": (True, False, True),
            "diffusion_mindfulness": (False, True, True),
            "no_modules": (False, False, False),
            "baseline": (False, False, False),
            # 5-condition ablation (subset of factorial)
            "no_ethics": (False, True, True),
            "no_diffusion": (True, False, True),
            "no_mindfulness": (True, True, False),
            "ablation_no_ethics": (False, True, True),
            "ablation_no_diffusion": (True, False, True),
            "ablation_no_mindfulness": (True, True, False),
        }

        for name in condition_names:
            if name in known_patterns:
                factor_map[name] = known_patterns[name]

        # Need at least 4 conditions for meaningful factorial analysis
        if len(factor_map) < 4:
            return None

        return factor_map

    def _compute_factorial_effects(
        self,
        results: Dict[str, Dict[str, List[float]]],
        factor_map: Dict[str, Tuple[bool, bool, bool]],
        metric: str,
    ) -> Optional[FactorialResult]:
        """Compute main and interaction effects for a metric."""
        # Gather all data with factor labels
        all_data = []
        for cond_name, (e, d, m) in factor_map.items():
            if cond_name not in results or metric not in results[cond_name]:
                continue
            for val in results[cond_name][metric]:
                all_data.append({
                    "value": val,
                    "ethics": 1.0 if e else -1.0,
                    "diffusion": 1.0 if d else -1.0,
                    "mindfulness": 1.0 if m else -1.0,
                    "condition": cond_name,
                })

        if len(all_data) < 8:
            return None

        values = np.array([d["value"] for d in all_data])
        ethics = np.array([d["ethics"] for d in all_data])
        diffusion = np.array([d["diffusion"] for d in all_data])
        mindfulness = np.array([d["mindfulness"] for d in all_data])

        grand_mean = np.mean(values)
        ss_total = np.sum((values - grand_mean) ** 2)

        # Main effects (using contrast coding)
        main_effects = {}
        for name, factor in [("ethics", ethics), ("diffusion", diffusion),
                             ("mindfulness", mindfulness)]:
            high = values[factor > 0]
            low = values[factor < 0]
            if len(high) == 0 or len(low) == 0:
                continue
            effect = np.mean(high) - np.mean(low)
            ss_effect = len(values) * (effect / 2) ** 2
            # F-test approximation
            ms_resid = max(ss_total / (len(values) - 1), 1e-12)
            f_stat = (ss_effect / 1) / ms_resid
            p_value = self._f_to_p_approx(f_stat, 1, len(values) - 2)
            main_effects[name] = {
                "effect_size": float(effect),
                "ss": float(ss_effect),
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
                "mean_high": float(np.mean(high)),
                "mean_low": float(np.mean(low)),
            }

        # Two-way interaction effects
        interaction_effects = {}
        factor_pairs = [
            ("ethics_x_diffusion", ethics, diffusion),
            ("ethics_x_mindfulness", ethics, mindfulness),
            ("diffusion_x_mindfulness", diffusion, mindfulness),
        ]
        for name, f1, f2 in factor_pairs:
            interaction = f1 * f2
            high = values[interaction > 0]
            low = values[interaction < 0]
            if len(high) == 0 or len(low) == 0:
                continue
            effect = np.mean(high) - np.mean(low)
            ss_effect = len(values) * (effect / 2) ** 2
            ms_resid = max(ss_total / (len(values) - 1), 1e-12)
            f_stat = (ss_effect / 1) / ms_resid
            p_value = self._f_to_p_approx(f_stat, 1, len(values) - 2)
            interaction_effects[name] = {
                "effect_size": float(effect),
                "ss": float(ss_effect),
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
            }

        # Three-way interaction
        interaction_3way = ethics * diffusion * mindfulness
        high = values[interaction_3way > 0]
        low = values[interaction_3way < 0]
        if len(high) > 0 and len(low) > 0:
            effect = np.mean(high) - np.mean(low)
            ss_effect = len(values) * (effect / 2) ** 2
            ms_resid = max(ss_total / (len(values) - 1), 1e-12)
            f_stat = (ss_effect / 1) / ms_resid
            p_value = self._f_to_p_approx(f_stat, 1, len(values) - 2)
            interaction_effects["ethics_x_diffusion_x_mindfulness"] = {
                "effect_size": float(effect),
                "ss": float(ss_effect),
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
            }

        # R-squared from regression
        X = np.column_stack([ethics, diffusion, mindfulness,
                             ethics * diffusion, ethics * mindfulness,
                             diffusion * mindfulness,
                             ethics * diffusion * mindfulness])
        X = np.column_stack([np.ones(len(values)), X])
        try:
            beta = np.linalg.lstsq(X, values, rcond=None)[0]
            predicted = X @ beta
            ss_reg = np.sum((predicted - grand_mean) ** 2)
            r_squared = ss_reg / (ss_total + 1e-12)
        except np.linalg.LinAlgError:
            r_squared = 0.0

        # Condition means
        cond_means = {}
        cond_stds = {}
        for cond_name in factor_map:
            if cond_name in results and metric in results[cond_name]:
                vals = results[cond_name][metric]
                cond_means[cond_name] = float(np.mean(vals))
                cond_stds[cond_name] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        return FactorialResult(
            main_effects=main_effects,
            interaction_effects=interaction_effects,
            full_model_r_squared=float(r_squared),
            residual_variance=float(ss_total / max(len(values) - 8, 1)),
            condition_means=cond_means,
            condition_stds=cond_stds,
        )

    def _compute_summary_statistics(
        self,
        results: Dict[str, Dict[str, List[float]]],
        metrics: List[str],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute summary statistics with bootstrap CIs for all conditions."""
        summary = {}
        for cond_name, cond_data in results.items():
            summary[cond_name] = {}
            for metric in metrics:
                if metric not in cond_data or not cond_data[metric]:
                    continue
                vals = np.array(cond_data[metric])
                ci_lo, ci_hi = self._bootstrap_ci(vals)
                summary[cond_name][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                    "median": float(np.median(vals)),
                    "ci_lower_95": float(ci_lo),
                    "ci_upper_95": float(ci_hi),
                    "n": len(vals),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                }
        return summary

    def bayesian_estimation(
        self,
        data_a: List[float],
        data_b: List[float],
        rope_width: float = 0.1,
        n_samples: int = 20000,
    ) -> BayesianEstimate:
        """
        Bayesian Estimation Supersedes the T-test (BEST).

        Uses bootstrap-based approximation of posterior.

        Args:
            data_a: Samples from condition A
            data_b: Samples from condition B
            rope_width: Region of Practical Equivalence half-width
            n_samples: Number of posterior samples
        """
        a = np.array(data_a)
        b = np.array(data_b)

        # Bootstrap posterior for difference of means
        diffs = np.zeros(n_samples)
        for i in range(n_samples):
            boot_a = a[np.random.randint(0, len(a), size=len(a))]
            boot_b = b[np.random.randint(0, len(b), size=len(b))]
            diffs[i] = np.mean(boot_a) - np.mean(boot_b)

        # HDI (Highest Density Interval)
        hdi_lo, hdi_hi = self._compute_hdi(diffs, credible_mass=0.95)

        # Probability that A > B
        prob_superior = float(np.mean(diffs > 0))

        # Effect size posterior
        pooled_stds = np.zeros(n_samples)
        for i in range(n_samples):
            boot_a = a[np.random.randint(0, len(a), size=len(a))]
            boot_b = b[np.random.randint(0, len(b), size=len(b))]
            pooled_std = np.sqrt(
                ((len(a) - 1) * np.var(boot_a, ddof=1) +
                 (len(b) - 1) * np.var(boot_b, ddof=1)) /
                (len(a) + len(b) - 2)
            )
            pooled_stds[i] = pooled_std

        effect_sizes = diffs / (pooled_stds + 1e-12)
        effect_mean = float(np.mean(effect_sizes))

        # ROPE analysis
        prob_in_rope = float(np.mean(
            (diffs >= -rope_width) & (diffs <= rope_width)
        ))

        return BayesianEstimate(
            posterior_mean_diff=float(np.mean(diffs)),
            hdi_lower=float(hdi_lo),
            hdi_upper=float(hdi_hi),
            prob_superior=prob_superior,
            effect_size_posterior_mean=effect_mean,
            rope_lower=-rope_width,
            rope_upper=rope_width,
            prob_in_rope=prob_in_rope,
        )

    def generate_latex_table(
        self, comparisons: List[PairwiseComparison]
    ) -> str:
        """Generate publication-ready LaTeX table from comparisons."""
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Statistical comparison of experimental conditions. "
            r"Effect sizes reported as Hedge's $g$. "
            r"$p$-values corrected with Holm-Bonferroni. "
            r"$^*p<.05$, $^{**}p<.01$, $^{***}p<.001$.}",
            r"\label{tab:statistical_comparison}",
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            r"Condition & Metric & Mean $\pm$ SE & $\Delta$ & 95\% CI & $p_{\mathrm{adj}}$ & $g$ \\",
            r"\midrule",
        ]

        for c in comparisons:
            sig = ""
            p = c.p_value_corrected if c.p_value_corrected is not None else c.p_value_raw
            if p < 0.001:
                sig = r"$^{***}$"
            elif p < 0.01:
                sig = r"$^{**}$"
            elif p < 0.05:
                sig = r"$^{*}$"

            lines.append(
                f"  {c.method_a} & {c.metric} & "
                f"${c.mean_a:.3f} \\pm {c.se_a:.3f}$ & "
                f"${c.difference:+.3f}$ & "
                f"$[{c.ci_lower:.3f}, {c.ci_upper:.3f}]$ & "
                f"${p:.4f}${sig} & "
                f"${c.effect_size_hedges_g:.2f}$ \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def generate_factorial_latex_table(
        self, factorial_results: Dict[str, FactorialResult]
    ) -> str:
        """Generate LaTeX ANOVA table for factorial analysis."""
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{2$^3$ factorial ANOVA: Main effects and interactions of "
            r"ethics (E), diffusion (D), and mindfulness (M) modules. "
            r"$^*p<.05$, $^{**}p<.01$, $^{***}p<.001$.}",
            r"\label{tab:factorial_anova}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Metric & Source & Effect Size & $F$ & $p$ & Sig. \\",
            r"\midrule",
        ]

        for metric, result in factorial_results.items():
            lines.append(f"  \\multicolumn{{6}}{{l}}{{\\textbf{{{metric}}}}} \\\\")

            for source_name, effect in result.main_effects.items():
                sig = self._significance_stars(effect["p_value"])
                lines.append(
                    f"  & {source_name.title()} & "
                    f"${effect['effect_size']:+.3f}$ & "
                    f"${effect['f_statistic']:.2f}$ & "
                    f"${effect['p_value']:.4f}$ & {sig} \\\\"
                )

            for source_name, effect in result.interaction_effects.items():
                readable = source_name.replace("_x_", r" $\times$ ").replace(
                    "ethics", "E").replace("diffusion", "D").replace("mindfulness", "M")
                sig = self._significance_stars(effect["p_value"])
                lines.append(
                    f"  & {readable} & "
                    f"${effect['effect_size']:+.3f}$ & "
                    f"${effect['f_statistic']:.2f}$ & "
                    f"${effect['p_value']:.4f}$ & {sig} \\\\"
                )

            lines.append(
                f"  & $R^2$ & \\multicolumn{{4}}{{c}}{{{result.full_model_r_squared:.3f}}} \\\\"
            )
            lines.append(r"  \midrule")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    # --- Helper methods ---

    @staticmethod
    def _cohens_d(
        mean_a: float, mean_b: float,
        std_a: float, std_b: float,
        n_a: int, n_b: int
    ) -> float:
        pooled = np.sqrt(
            ((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) / max(n_a + n_b - 2, 1)
        )
        return (mean_a - mean_b) / (pooled + 1e-12)

    @staticmethod
    def _hedges_g(cohens_d: float, n_a: int, n_b: int) -> float:
        """Hedge's g: bias-corrected effect size for small samples."""
        correction = 1 - 3 / (4 * (n_a + n_b) - 9)
        return cohens_d * correction

    @staticmethod
    def _interpret_effect_size(d: float) -> str:
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def _bootstrap_ci(
        self, data: np.ndarray, ci_level: float = 0.95
    ) -> Tuple[float, float]:
        alpha = 1 - ci_level
        n = len(data)
        boot_means = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            boot_means[i] = np.mean(data[np.random.randint(0, n, size=n)])
        return (
            np.percentile(boot_means, 100 * alpha / 2),
            np.percentile(boot_means, 100 * (1 - alpha / 2)),
        )

    def _bootstrap_difference_ci(
        self, a: np.ndarray, b: np.ndarray, ci_level: float = 0.95
    ) -> Tuple[float, float]:
        alpha = 1 - ci_level
        diffs = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            boot_a = a[np.random.randint(0, len(a), size=len(a))]
            boot_b = b[np.random.randint(0, len(b), size=len(b))]
            diffs[i] = np.mean(boot_a) - np.mean(boot_b)
        return (
            np.percentile(diffs, 100 * alpha / 2),
            np.percentile(diffs, 100 * (1 - alpha / 2)),
        )

    @staticmethod
    def _compute_hdi(samples: np.ndarray, credible_mass: float = 0.95) -> Tuple[float, float]:
        """Compute Highest Density Interval from posterior samples."""
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        ci_size = int(np.ceil(credible_mass * n))
        n_cis = n - ci_size

        ci_widths = sorted_samples[ci_size:] - sorted_samples[:n_cis]
        best_idx = np.argmin(ci_widths)
        return float(sorted_samples[best_idx]), float(sorted_samples[best_idx + ci_size])

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    @staticmethod
    def _f_to_p_approx(f_stat: float, df1: int, df2: int) -> float:
        """Approximate F-distribution p-value."""
        try:
            from scipy.stats import f as f_dist
            return float(1 - f_dist.cdf(f_stat, df1, df2))
        except ImportError:
            # Rough approximation: F -> chi-squared -> normal
            x = (f_stat ** (1 / 3) * (1 - 2 / (9 * df2)) -
                 (1 - 2 / (9 * df1))) / np.sqrt(2 / (9 * df1) + 2 / (9 * df2) * f_stat ** (2 / 3))
            return max(1e-12, float(1 - StatisticalAnalyzer._norm_cdf(x)))

    @staticmethod
    def _significance_stars(p: float) -> str:
        if p < 0.001:
            return r"$^{***}$"
        elif p < 0.01:
            return r"$^{**}$"
        elif p < 0.05:
            return r"$^{*}$"
        return ""


class FactorialAnalysis:
    """Convenience wrapper for factorial analysis."""

    @staticmethod
    def analyze(
        results: Dict[str, Dict[str, List[float]]],
        metrics: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> Dict:
        analyzer = StatisticalAnalyzer(alpha=alpha)
        return analyzer.run_full_analysis(results, metrics=metrics)
