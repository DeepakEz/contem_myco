"""
Visualization Tools
===================
Plotting functions for experiment results.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

# Try importing matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def plot_learning_curves(
    results: Dict[str, List[float]],
    title: str = "Learning Curves",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    save_path: Optional[str] = None,
    window: int = 100,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot learning curves for multiple methods.

    Args:
        results: Dict of method_name -> [rewards per episode]
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (optional)
        window: Smoothing window size
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not available")
        return

    plt.figure(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (method_name, rewards), color in zip(results.items(), colors):
        rewards = np.array(rewards)

        # Compute smoothed curve
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            x = np.arange(window - 1, len(rewards))
        else:
            smoothed = rewards
            x = np.arange(len(rewards))

        plt.plot(x, smoothed, label=method_name, color=color, linewidth=2)

        # Add confidence band (std over rolling window)
        if len(rewards) >= window:
            stds = []
            for i in range(len(rewards) - window + 1):
                stds.append(np.std(rewards[i:i+window]))
            stds = np.array(stds)
            plt.fill_between(x, smoothed - stds, smoothed + stds, color=color, alpha=0.2)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()


def plot_comparison_bars(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot bar chart comparing methods across metrics.

    Args:
        results: Dict of method_name -> {metric_name -> value}
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not available")
        return

    methods = list(results.keys())
    metrics = metrics or ['reward', 'social_welfare', 'gini', 'cooperation']

    # Filter to available metrics
    available_metrics = []
    for m in metrics:
        if all(m in results[method] for method in methods):
            available_metrics.append(m)

    if not available_metrics:
        print("No common metrics found")
        return

    n_methods = len(methods)
    n_metrics = len(available_metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    for ax, metric in zip(axes, available_metrics):
        values = [results[m].get(metric, 0) for m in methods]
        stds = [results[m].get(f'{metric}_std', 0) for m in methods]

        bars = ax.bar(range(n_methods), values, color=colors, yerr=stds, capsize=5)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()


def plot_ablation_heatmap(
    results: Dict[str, Dict[str, float]],
    modules: List[str] = None,
    metrics: List[str] = None,
    title: str = "Ablation Study",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot heatmap for ablation study results.

    Args:
        results: Dict of config_name -> {metric_name -> value}
        modules: List of module names (for x-axis)
        metrics: List of metrics (for y-axis)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not available")
        return

    modules = modules or ['ethics', 'diffusion', 'mindfulness']
    metrics = metrics or ['reward', 'social_welfare', 'gini', 'cooperation']

    # Build ablation matrix
    configs = ['full'] + [f'no_{m}' for m in modules] + ['baseline']

    # Check which configs exist
    available_configs = [c for c in configs if c in results]

    if len(available_configs) < 2:
        print("Not enough ablation configs found")
        return

    # Build data matrix
    data = []
    for metric in metrics:
        row = []
        for config in available_configs:
            if config in results and metric in results[config]:
                row.append(results[config][metric])
            else:
                row.append(np.nan)
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize per row for better visualization
    data_norm = np.zeros_like(data)
    for i in range(data.shape[0]):
        row = data[i]
        if not np.all(np.isnan(row)):
            valid = row[~np.isnan(row)]
            if len(valid) > 0:
                data_norm[i] = (row - np.nanmin(row)) / (np.nanmax(row) - np.nanmin(row) + 1e-8)

    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto')

    # Labels
    ax.set_xticks(range(len(available_configs)))
    ax.set_xticklabels(available_configs, rotation=45, ha='right')
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics])

    # Add values as text
    for i in range(len(metrics)):
        for j in range(len(available_configs)):
            if not np.isnan(data[i, j]):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                              ha='center', va='center', color='black', fontsize=9)

    plt.colorbar(im, ax=ax, label='Normalized Score')
    plt.title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()


def plot_diffusion_field(
    field: np.ndarray,
    channel_names: List[str] = None,
    title: str = "Diffusion Field State",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 3),
):
    """
    Visualize diffusion field channels.

    Args:
        field: Diffusion field array (channels, height, width)
        channel_names: Names for each channel
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not available")
        return

    n_channels = field.shape[0]
    channel_names = channel_names or [f'Channel {i}' for i in range(n_channels)]

    fig, axes = plt.subplots(1, n_channels, figsize=figsize)
    if n_channels == 1:
        axes = [axes]

    for ax, channel_data, name in zip(axes, field, channel_names):
        im = ax.imshow(channel_data, cmap='hot', interpolation='nearest')
        ax.set_title(name, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()


def generate_paper_figures(
    experiment_dir: str,
    output_dir: str,
):
    """
    Generate all figures for paper from experiment results.

    Args:
        experiment_dir: Directory with experiment results
        output_dir: Directory to save figures
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot generate figures: matplotlib not available")
        return

    from .significance import load_experiment_results

    exp_path = Path(experiment_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_experiment_results(experiment_dir)

    if not results:
        print(f"No results found in {experiment_dir}")
        return

    # Compute aggregated results
    agg_results = {}
    for method, metrics in results.items():
        agg_results[method] = {
            'reward': np.mean(metrics.get('reward', [0])),
            'reward_std': np.std(metrics.get('reward', [0])),
            'social_welfare': np.mean(metrics.get('social_welfare', [0])),
            'gini': np.mean(metrics.get('gini', [0])),
            'cooperation': np.mean(metrics.get('cooperation', [0])),
        }

    # Generate comparison bar chart
    plot_comparison_bars(
        agg_results,
        title="Method Comparison",
        save_path=str(out_path / 'comparison_bars.png'),
    )

    # Generate ablation heatmap
    plot_ablation_heatmap(
        agg_results,
        title="Ablation Study Results",
        save_path=str(out_path / 'ablation_heatmap.png'),
    )

    print(f"Generated figures in {output_dir}")
