#!/usr/bin/env python3
"""
Visualization - Plot Generation
================================
Generate comparison plots for experiments:
- Casualties over time (resilience)
- Suffering comparison
- Inequality trends (society)
- Trust evolution (society)
- Compute overhead analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


class ExperimentVisualizer:
    """
    Generate plots comparing experimental results
    """

    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_resilience_comparison(
        self,
        reactive_results: Dict[str, Any],
        contemplative_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        Generate comparison plots for resilience experiments

        Shows:
        - Casualties over time
        - Suffering levels
        - Rescue counts
        - Compute time
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Resilience Comparison: Reactive vs Contemplative Agents', fontsize=16, fontweight='bold')

        # Extract data
        reactive_runs = reactive_results.get('all_runs', [])
        contemplative_runs = contemplative_results.get('all_runs', [])

        # Plot 1: Casualties over time
        ax = axes[0, 0]
        self._plot_metric_over_time(
            ax, reactive_runs, 'casualties', 'Reactive', 'red'
        )
        self._plot_metric_over_time(
            ax, contemplative_runs, 'casualties', 'Contemplative', 'blue'
        )
        ax.set_xlabel('Step')
        ax.set_ylabel('Casualties')
        ax.set_title('Casualties Over Time')
        ax.legend()

        # Plot 2: Average Suffering
        ax = axes[0, 1]
        self._plot_metric_over_time(
            ax, reactive_runs, 'avg_suffering', 'Reactive', 'red'
        )
        self._plot_metric_over_time(
            ax, contemplative_runs, 'avg_suffering', 'Contemplative', 'blue'
        )
        ax.set_xlabel('Step')
        ax.set_ylabel('Avg Suffering')
        ax.set_title('Average Suffering Over Time')
        ax.legend()

        # Plot 3: Total Rescues
        ax = axes[1, 0]
        reactive_rescues = [run['final_metrics'].get('total_rescues', 0) for run in reactive_runs]
        contemplative_rescues = [run['final_metrics'].get('total_rescues', 0) for run in contemplative_runs]

        ax.bar(['Reactive', 'Contemplative'],
               [np.mean(reactive_rescues), np.mean(contemplative_rescues)],
               yerr=[np.std(reactive_rescues), np.std(contemplative_rescues)],
               color=['red', 'blue'],
               alpha=0.7)
        ax.set_ylabel('Total Rescues')
        ax.set_title('Total Rescues Performed')

        # Plot 4: Compute Time Comparison
        ax = axes[1, 1]
        reactive_compute = reactive_results.get('compute_metrics', {})
        contemplative_compute = contemplative_results.get('compute_metrics', {})

        compute_times = [
            reactive_compute.get('avg_time_per_operation_ms', 0),
            contemplative_compute.get('avg_time_per_operation_ms', 0)
        ]

        ax.bar(['Reactive', 'Contemplative'], compute_times, color=['red', 'blue'], alpha=0.7)
        ax.set_ylabel('Avg Compute Time (ms)')
        ax.set_title('Computational Cost per Decision')

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / "resilience_comparison.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved resilience comparison plot to {save_path}")
        plt.close()

    def plot_society_comparison(
        self,
        baseline_results: Dict[str, Any],
        myco_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        Generate comparison plots for society experiments

        Shows:
        - Inequality trends
        - Trust evolution
        - Suffering levels
        - Crime rates
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Society Comparison: Baseline vs Myco Policy', fontsize=16, fontweight='bold')

        # Extract data
        baseline_runs = baseline_results.get('all_runs', [])
        myco_runs = myco_results.get('all_runs', [])

        # Plot 1: Inequality over time
        ax = axes[0, 0]
        self._plot_metric_over_time(
            ax, baseline_runs, 'inequality', 'Baseline', 'orange'
        )
        self._plot_metric_over_time(
            ax, myco_runs, 'inequality', 'Myco', 'green'
        )
        ax.set_xlabel('Step')
        ax.set_ylabel('Inequality (Gini)')
        ax.set_title('Inequality Over Time')
        ax.legend()

        # Plot 2: Trust over time
        ax = axes[0, 1]
        self._plot_metric_over_time(
            ax, baseline_runs, 'avg_trust', 'Baseline', 'orange'
        )
        self._plot_metric_over_time(
            ax, myco_runs, 'avg_trust', 'Myco', 'green'
        )
        ax.set_xlabel('Step')
        ax.set_ylabel('Avg Trust')
        ax.set_title('Average Trust Over Time')
        ax.legend()

        # Plot 3: Suffering over time
        ax = axes[1, 0]
        self._plot_metric_over_time(
            ax, baseline_runs, 'avg_suffering', 'Baseline', 'orange'
        )
        self._plot_metric_over_time(
            ax, myco_runs, 'avg_suffering', 'Myco', 'green'
        )
        ax.set_xlabel('Step')
        ax.set_ylabel('Avg Suffering')
        ax.set_title('Average Suffering Over Time')
        ax.legend()

        # Plot 4: Crime comparison
        ax = axes[1, 1]
        baseline_crimes = [run['final_metrics'].get('total_crimes', 0) for run in baseline_runs]
        myco_crimes = [run['final_metrics'].get('total_crimes', 0) for run in myco_runs]

        ax.bar(['Baseline', 'Myco'],
               [np.mean(baseline_crimes), np.mean(myco_crimes)],
               yerr=[np.std(baseline_crimes), np.std(myco_crimes)],
               color=['orange', 'green'],
               alpha=0.7)
        ax.set_ylabel('Total Crimes')
        ax.set_title('Total Crimes')

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / "society_comparison.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved society comparison plot to {save_path}")
        plt.close()

    def _plot_metric_over_time(
        self,
        ax,
        runs: List[Dict[str, Any]],
        metric_name: str,
        label: str,
        color: str
    ):
        """Plot a metric over time with mean and std band"""
        if not runs:
            return

        # Extract time series
        all_series = []
        for run in runs:
            series = [m.get(metric_name, 0) for m in run['step_metrics']]
            all_series.append(series)

        # Pad to same length
        max_len = max(len(s) for s in all_series)
        padded = [s + [s[-1]] * (max_len - len(s)) for s in all_series]

        # Compute mean and std
        mean_series = np.mean(padded, axis=0)
        std_series = np.std(padded, axis=0)

        steps = np.arange(len(mean_series))

        # Plot
        ax.plot(steps, mean_series, label=label, color=color, linewidth=2)
        ax.fill_between(steps,
                        mean_series - std_series,
                        mean_series + std_series,
                        alpha=0.2, color=color)

    def plot_compute_overhead(
        self,
        reactive_results: Dict[str, Any],
        contemplative_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Generate detailed compute overhead analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Computational Overhead Analysis', fontsize=14, fontweight='bold')

        # Extract compute metrics
        reactive_compute = reactive_results.get('compute_metrics', {})
        contemplative_compute = contemplative_results.get('compute_metrics', {})

        # Plot 1: Time comparison
        ax = axes[0]
        reactive_time = reactive_compute.get('avg_time_per_operation_ms', 0)
        contemplative_time = contemplative_compute.get('avg_time_per_operation_ms', 0)

        bars = ax.bar(['Reactive', 'Contemplative'],
                     [reactive_time, contemplative_time],
                     color=['red', 'blue'],
                     alpha=0.7)

        ax.set_ylabel('Time per Decision (ms)')
        ax.set_title('Average Decision Time')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}ms',
                   ha='center', va='bottom')

        # Plot 2: Overhead percentage
        ax = axes[1]
        if reactive_time > 0:
            overhead_pct = ((contemplative_time - reactive_time) / reactive_time) * 100
        else:
            overhead_pct = 0

        ax.bar(['Overhead'], [overhead_pct], color='purple', alpha=0.7)
        ax.set_ylabel('Overhead (%)')
        ax.set_title('Contemplative Overhead vs Reactive')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / "compute_overhead.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved compute overhead plot to {save_path}")
        plt.close()

    def plot_ethical_metrics(
        self,
        contemplative_results: Dict[str, Any],
        myco_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Plot ethical metrics for contemplative agents"""
        # This would extract ethical scores from logs
        # For now, create placeholder
        logger.info("Ethical metrics plot would be generated here")


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(results_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    print("Visualization Test:")
    print("  Creating sample visualizer...")

    visualizer = ExperimentVisualizer(output_dir="test_plots")

    # Create dummy data for testing
    dummy_reactive = {
        'all_runs': [
            {
                'step_metrics': [
                    {'casualties': i, 'avg_suffering': 0.5 + i*0.01, 'total_rescues': i//2}
                    for i in range(50)
                ],
                'final_metrics': {'total_rescues': 25}
            }
        ],
        'compute_metrics': {'avg_time_per_operation_ms': 2.5}
    }

    dummy_contemplative = {
        'all_runs': [
            {
                'step_metrics': [
                    {'casualties': i//2, 'avg_suffering': 0.4 + i*0.005, 'total_rescues': i}
                    for i in range(50)
                ],
                'final_metrics': {'total_rescues': 50}
            }
        ],
        'compute_metrics': {'avg_time_per_operation_ms': 8.5}
    }

    visualizer.plot_resilience_comparison(dummy_reactive, dummy_contemplative)
    visualizer.plot_compute_overhead(dummy_reactive, dummy_contemplative)

    print("  ✓ Sample plots generated in test_plots/")
    print("\n✓ Visualization module initialized successfully")
