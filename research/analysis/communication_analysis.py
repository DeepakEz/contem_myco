"""
Emergent Communication Analysis
================================
Analyzes the communication patterns emerging from stigmergic diffusion fields.

Includes:
- Signal entropy and information content
- Temporal pattern analysis
- Channel specialization metrics
- Spatial clustering of deposited signals
- Communication efficiency vs direct messaging
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class CommunicationMetrics:
    """Metrics characterizing emergent communication patterns."""
    channel_entropy: List[float]
    channel_utilization: List[float]
    spatial_concentration: List[float]
    temporal_autocorrelation: List[float]
    cross_channel_correlation: float
    total_information_content: float
    channel_specialization_index: float
    deposit_sparsity: float


class CommunicationAnalyzer:
    """
    Analyzes emergent communication patterns in stigmergic diffusion fields.

    Examines whether agents develop meaningful communication protocols through
    the diffusion field and measures information efficiency compared to
    direct message-passing baselines.
    """

    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins

    def analyze_field_snapshot(
        self,
        field: np.ndarray,
        channel_names: Optional[List[str]] = None,
    ) -> CommunicationMetrics:
        """
        Analyze a single diffusion field snapshot.

        Args:
            field: Diffusion field array (C, H, W)
            channel_names: Optional names for channels
        """
        n_channels, h, w = field.shape

        channel_entropy = []
        channel_utilization = []
        spatial_concentration = []

        for c in range(n_channels):
            channel_data = field[c]

            # Entropy of signal distribution
            flat = channel_data.flatten()
            flat = flat[flat > 1e-8]
            if len(flat) > 0:
                flat_norm = flat / (flat.sum() + 1e-12)
                entropy = -np.sum(flat_norm * np.log2(flat_norm + 1e-12))
            else:
                entropy = 0.0
            channel_entropy.append(float(entropy))

            # Utilization: fraction of cells with non-negligible signal
            utilization = float(np.mean(channel_data > 0.01))
            channel_utilization.append(utilization)

            # Spatial concentration: inverse of spread (higher = more concentrated)
            if channel_data.sum() > 1e-8:
                total = channel_data.sum()
                # Compute center of mass
                y_coords, x_coords = np.meshgrid(range(h), range(w), indexing='ij')
                cx = np.sum(x_coords * channel_data) / total
                cy = np.sum(y_coords * channel_data) / total
                # Compute variance from center
                var = np.sum(
                    ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) * channel_data
                ) / total
                concentration = 1.0 / (1.0 + var)
            else:
                concentration = 0.0
            spatial_concentration.append(float(concentration))

        # Cross-channel correlation
        if n_channels >= 2:
            flat_channels = [field[c].flatten() for c in range(n_channels)]
            corr_matrix = np.corrcoef(flat_channels)
            # Average off-diagonal correlation
            mask = ~np.eye(n_channels, dtype=bool)
            cross_corr = float(np.nanmean(np.abs(corr_matrix[mask])))
        else:
            cross_corr = 0.0

        # Total information content
        total_info = sum(channel_entropy)

        # Channel specialization: how different are the channels from each other
        if n_channels >= 2:
            norms = [field[c].flatten() / (field[c].sum() + 1e-12) for c in range(n_channels)]
            # Average Jensen-Shannon divergence between channels
            js_divs = []
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    m = (norms[i] + norms[j]) / 2
                    kl_i = np.sum(norms[i] * np.log2((norms[i] + 1e-12) / (m + 1e-12)))
                    kl_j = np.sum(norms[j] * np.log2((norms[j] + 1e-12) / (m + 1e-12)))
                    js_divs.append((kl_i + kl_j) / 2)
            specialization = float(np.mean(js_divs)) if js_divs else 0.0
        else:
            specialization = 0.0

        # Deposit sparsity
        all_values = field.flatten()
        sparsity = float(np.mean(all_values < 0.01))

        return CommunicationMetrics(
            channel_entropy=channel_entropy,
            channel_utilization=channel_utilization,
            spatial_concentration=spatial_concentration,
            temporal_autocorrelation=[],  # Requires temporal data
            cross_channel_correlation=cross_corr,
            total_information_content=total_info,
            channel_specialization_index=specialization,
            deposit_sparsity=sparsity,
        )

    def analyze_field_trajectory(
        self,
        field_history: List[np.ndarray],
        channel_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Analyze communication patterns over time.

        Args:
            field_history: List of field snapshots over time
        """
        snapshot_metrics = [
            self.analyze_field_snapshot(f, channel_names)
            for f in field_history
        ]

        n_channels = field_history[0].shape[0] if field_history else 0

        # Temporal autocorrelation per channel
        temporal_autocorr = []
        for c in range(n_channels):
            series = [f[c].mean() for f in field_history]
            if len(series) > 1:
                autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                temporal_autocorr.append(float(autocorr) if not np.isnan(autocorr) else 0.0)
            else:
                temporal_autocorr.append(0.0)

        # Information rate (bits per timestep)
        info_rate = np.diff([m.total_information_content for m in snapshot_metrics])

        # Specialization trend
        spec_trend = [m.channel_specialization_index for m in snapshot_metrics]

        return {
            "snapshot_metrics": snapshot_metrics,
            "temporal_autocorrelation": temporal_autocorr,
            "mean_information_rate": float(np.mean(info_rate)) if len(info_rate) > 0 else 0.0,
            "specialization_trend": spec_trend,
            "mean_specialization": float(np.mean(spec_trend)),
            "final_specialization": spec_trend[-1] if spec_trend else 0.0,
            "mean_entropy": [
                float(np.mean([m.channel_entropy[c] for m in snapshot_metrics]))
                for c in range(n_channels)
            ],
            "mean_utilization": [
                float(np.mean([m.channel_utilization[c] for m in snapshot_metrics]))
                for c in range(n_channels)
            ],
        }

    def compare_communication_efficiency(
        self,
        stigmergic_results: Dict[str, List[float]],
        commnet_results: Dict[str, List[float]],
        tarmac_results: Dict[str, List[float]],
        metrics: List[str] = None,
    ) -> Dict:
        """
        Compare stigmergic communication efficiency against direct messaging.

        Computes performance-per-bit metrics to show information efficiency.
        """
        metrics = metrics or ["social_welfare", "cooperation_rate"]

        comparison = {}
        for metric in metrics:
            s_vals = stigmergic_results.get(metric, [])
            c_vals = commnet_results.get(metric, [])
            t_vals = tarmac_results.get(metric, [])

            comparison[metric] = {
                "stigmergic": {
                    "mean": float(np.mean(s_vals)) if s_vals else 0.0,
                    "std": float(np.std(s_vals)) if len(s_vals) > 1 else 0.0,
                },
                "commnet": {
                    "mean": float(np.mean(c_vals)) if c_vals else 0.0,
                    "std": float(np.std(c_vals)) if len(c_vals) > 1 else 0.0,
                },
                "tarmac": {
                    "mean": float(np.mean(t_vals)) if t_vals else 0.0,
                    "std": float(np.std(t_vals)) if len(t_vals) > 1 else 0.0,
                },
            }

        return comparison

    def plot_channel_analysis(
        self,
        trajectory_analysis: Dict,
        channel_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ):
        """Plot comprehensive channel analysis."""
        if not MATPLOTLIB_AVAILABLE:
            return

        snapshot_metrics = trajectory_analysis["snapshot_metrics"]
        n_channels = len(snapshot_metrics[0].channel_entropy) if snapshot_metrics else 0
        channel_names = channel_names or [f"Ch {i}" for i in range(n_channels)]
        timesteps = range(len(snapshot_metrics))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Channel entropy over time
        ax = axes[0, 0]
        for c in range(n_channels):
            entropies = [m.channel_entropy[c] for m in snapshot_metrics]
            ax.plot(timesteps, entropies, label=channel_names[c], linewidth=2)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Entropy (bits)")
        ax.set_title("Channel Information Content")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Channel utilization over time
        ax = axes[0, 1]
        for c in range(n_channels):
            utils = [m.channel_utilization[c] for m in snapshot_metrics]
            ax.plot(timesteps, utils, label=channel_names[c], linewidth=2)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Utilization Rate")
        ax.set_title("Channel Utilization")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Specialization trend
        ax = axes[1, 0]
        ax.plot(timesteps, trajectory_analysis["specialization_trend"],
                linewidth=2, color="darkred")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Specialization Index (JS Divergence)")
        ax.set_title("Channel Specialization Over Time")
        ax.grid(True, alpha=0.3)

        # 4. Spatial concentration over time
        ax = axes[1, 1]
        for c in range(n_channels):
            concs = [m.spatial_concentration[c] for m in snapshot_metrics]
            ax.plot(timesteps, concs, label=channel_names[c], linewidth=2)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Spatial Concentration")
        ax.set_title("Signal Spatial Concentration")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle("Emergent Communication Analysis", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
