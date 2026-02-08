"""
Sample Efficiency Analysis
===========================
Analyzes learning speed and data efficiency across conditions.

Includes:
- Learning curve extraction and smoothing
- Area under the learning curve (AUC)
- Time to threshold metrics
- Asymptotic performance estimation
- Statistical comparison of learning speed
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class LearningCurveStats:
    """Statistics from a learning curve."""
    method_name: str
    timesteps: np.ndarray
    mean_rewards: np.ndarray
    std_rewards: np.ndarray
    se_rewards: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    auc: float
    time_to_threshold: Optional[float]
    asymptotic_performance: float
    learning_rate_initial: float
    n_seeds: int


class SampleEfficiencyAnalyzer:
    """
    Analyzes sample efficiency across experimental conditions.

    Computes learning curves, area under the curve, time to threshold,
    and asymptotic performance for each condition.
    """

    def __init__(self, smooth_window: int = 50):
        self.smooth_window = smooth_window

    def extract_learning_curves(
        self,
        experiment_dir: str,
        conditions: Optional[List[str]] = None,
    ) -> Dict[str, LearningCurveStats]:
        """
        Extract and analyze learning curves from experiment directory.

        Args:
            experiment_dir: Path to experiment output
            conditions: Optional list of condition names to analyze

        Returns:
            Dict of condition_name -> LearningCurveStats
        """
        exp_path = Path(experiment_dir)
        results = {}

        for cond_dir in exp_path.iterdir():
            if not cond_dir.is_dir():
                continue
            if conditions and cond_dir.name not in conditions:
                continue

            seed_curves = []
            for seed_dir in cond_dir.iterdir():
                if not seed_dir.is_dir():
                    continue
                results_file = seed_dir / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        data = json.load(f)
                    if "training_rewards" in data:
                        seed_curves.append(data["training_rewards"])

            if seed_curves:
                stats = self._compute_curve_statistics(
                    cond_dir.name, seed_curves
                )
                results[cond_dir.name] = stats

        return results

    def analyze_from_data(
        self,
        curves: Dict[str, List[List[float]]],
    ) -> Dict[str, LearningCurveStats]:
        """
        Analyze learning curves from raw data.

        Args:
            curves: {condition_name: [[rewards_seed_0], [rewards_seed_1], ...]}
        """
        results = {}
        for name, seed_curves in curves.items():
            if seed_curves:
                results[name] = self._compute_curve_statistics(name, seed_curves)
        return results

    def _compute_curve_statistics(
        self,
        name: str,
        seed_curves: List[List[float]],
    ) -> LearningCurveStats:
        """Compute statistics from multiple seed curves."""
        # Align curves to same length
        min_len = min(len(c) for c in seed_curves)
        aligned = np.array([c[:min_len] for c in seed_curves])

        # Smooth each curve
        smoothed = np.array([
            self._smooth(curve) for curve in aligned
        ])

        # Truncate to smoothed length
        min_smoothed_len = min(len(s) for s in smoothed)
        smoothed = np.array([s[:min_smoothed_len] for s in smoothed])

        timesteps = np.arange(min_smoothed_len)
        mean_r = np.mean(smoothed, axis=0)
        std_r = np.std(smoothed, axis=0)
        n = len(seed_curves)
        se_r = std_r / np.sqrt(n)

        # 95% CI
        ci_lower = mean_r - 1.96 * se_r
        ci_upper = mean_r + 1.96 * se_r

        # Area under the learning curve (normalized by length)
        auc = float(np.trapz(mean_r, timesteps) / max(len(timesteps), 1))

        # Time to threshold (80% of final performance)
        final_perf = np.mean(mean_r[-max(1, len(mean_r) // 10):])
        threshold = 0.8 * final_perf
        time_to_thresh = None
        for t, r in enumerate(mean_r):
            if r >= threshold:
                time_to_thresh = float(t)
                break

        # Asymptotic performance (mean of last 10%)
        asymptotic = float(final_perf)

        # Initial learning rate (slope of first 10%)
        early_len = max(10, len(mean_r) // 10)
        if early_len > 1:
            early_slope = float(
                np.polyfit(timesteps[:early_len], mean_r[:early_len], 1)[0]
            )
        else:
            early_slope = 0.0

        return LearningCurveStats(
            method_name=name,
            timesteps=timesteps,
            mean_rewards=mean_r,
            std_rewards=std_r,
            se_rewards=se_r,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            auc=auc,
            time_to_threshold=time_to_thresh,
            asymptotic_performance=asymptotic,
            learning_rate_initial=early_slope,
            n_seeds=n,
        )

    def _smooth(self, data: List[float]) -> np.ndarray:
        """Apply exponential moving average smoothing."""
        arr = np.array(data)
        if len(arr) < self.smooth_window:
            return arr
        kernel = np.ones(self.smooth_window) / self.smooth_window
        return np.convolve(arr, kernel, mode="valid")

    def compare_efficiency(
        self,
        curves: Dict[str, LearningCurveStats],
    ) -> Dict:
        """
        Compare sample efficiency across conditions.

        Returns a structured comparison of AUC, time-to-threshold,
        and asymptotic performance.
        """
        comparison = {}
        for name, stats in curves.items():
            comparison[name] = {
                "auc_normalized": stats.auc,
                "time_to_80pct": stats.time_to_threshold,
                "asymptotic_reward": stats.asymptotic_performance,
                "initial_learning_speed": stats.learning_rate_initial,
                "n_seeds": stats.n_seeds,
            }

        # Rank by AUC
        sorted_by_auc = sorted(
            comparison.items(), key=lambda x: x[1]["auc_normalized"], reverse=True
        )
        for rank, (name, _) in enumerate(sorted_by_auc):
            comparison[name]["auc_rank"] = rank + 1

        return comparison

    def plot_learning_curves(
        self,
        curves: Dict[str, LearningCurveStats],
        title: str = "Learning Curves with 95% CI",
        save_path: Optional[str] = None,
    ):
        """Plot learning curves with confidence intervals."""
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))

        # Learning curves with CI
        ax = axes[0]
        for (name, stats), color in zip(curves.items(), colors):
            ax.plot(stats.timesteps, stats.mean_rewards,
                    label=f"{name} (AUC={stats.auc:.1f})", color=color, linewidth=2)
            ax.fill_between(
                stats.timesteps, stats.ci_lower, stats.ci_upper,
                color=color, alpha=0.15,
            )
        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Mean Reward", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        # Sample efficiency bar chart
        ax = axes[1]
        names = list(curves.keys())
        aucs = [curves[n].auc for n in names]
        bars = ax.barh(names, aucs, color=colors[:len(names)], alpha=0.7, edgecolor="black")
        ax.set_xlabel("Area Under Learning Curve", fontsize=12)
        ax.set_title("Sample Efficiency Comparison", fontsize=13)
        ax.grid(True, axis="x", alpha=0.3)

        # Add time-to-threshold annotations
        for i, name in enumerate(names):
            ttt = curves[name].time_to_threshold
            if ttt is not None:
                ax.annotate(
                    f"T80={ttt:.0f}",
                    xy=(aucs[i], i),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=8,
                    va="center",
                )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
