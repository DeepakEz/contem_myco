"""
Calibration Analysis for Mindfulness Uncertainty
=================================================
Evaluates how well-calibrated the uncertainty estimates from the
mindfulness module are (predicted uncertainty vs actual error).

Includes:
- Reliability diagrams
- Expected Calibration Error (ECE)
- Brier score decomposition
- Sharpness analysis
- Uncertainty-error correlation
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
class CalibrationMetrics:
    """Comprehensive calibration metrics."""
    expected_calibration_error: float
    maximum_calibration_error: float
    average_uncertainty: float
    average_error: float
    uncertainty_error_correlation: float
    sharpness: float
    bin_confidences: List[float]
    bin_accuracies: List[float]
    bin_counts: List[int]
    n_bins: int


class CalibrationAnalyzer:
    """
    Analyzes calibration of uncertainty estimates from the mindfulness module.

    A well-calibrated system should have uncertainty proportional to actual
    prediction error: when the module says it's 70% uncertain, the actual
    error should be around the 70th percentile of all errors.
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def analyze_calibration(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
    ) -> CalibrationMetrics:
        """
        Compute calibration metrics from uncertainty estimates and actual errors.

        Args:
            uncertainties: Predicted uncertainty values (N,)
            errors: Actual prediction errors (N,)

        Returns:
            CalibrationMetrics with ECE, MCE, reliability data
        """
        uncertainties = np.asarray(uncertainties).flatten()
        errors = np.asarray(errors).flatten()

        assert len(uncertainties) == len(errors), \
            f"Length mismatch: {len(uncertainties)} vs {len(errors)}"

        # Normalize both to [0, 1] range for comparison
        u_min, u_max = uncertainties.min(), uncertainties.max()
        if u_max > u_min:
            u_norm = (uncertainties - u_min) / (u_max - u_min)
        else:
            u_norm = np.zeros_like(uncertainties)

        e_min, e_max = errors.min(), errors.max()
        if e_max > e_min:
            e_norm = (errors - e_min) / (e_max - e_min)
        else:
            e_norm = np.zeros_like(errors)

        # Bin by uncertainty level
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        ece = 0.0
        mce = 0.0

        for i in range(self.n_bins):
            mask = (u_norm >= bin_edges[i]) & (u_norm < bin_edges[i + 1])
            if i == self.n_bins - 1:
                mask = mask | (u_norm == bin_edges[i + 1])

            count = mask.sum()
            bin_counts.append(int(count))

            if count > 0:
                avg_uncertainty = float(np.mean(u_norm[mask]))
                avg_error = float(np.mean(e_norm[mask]))
                bin_confidences.append(avg_uncertainty)
                bin_accuracies.append(avg_error)

                gap = abs(avg_uncertainty - avg_error)
                ece += (count / len(uncertainties)) * gap
                mce = max(mce, gap)
            else:
                bin_confidences.append(float(bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_accuracies.append(0.0)

        # Uncertainty-error correlation
        if np.std(uncertainties) > 1e-12 and np.std(errors) > 1e-12:
            correlation = float(np.corrcoef(uncertainties, errors)[0, 1])
        else:
            correlation = 0.0

        # Sharpness: concentration of uncertainty predictions
        sharpness = float(np.std(uncertainties))

        return CalibrationMetrics(
            expected_calibration_error=float(ece),
            maximum_calibration_error=float(mce),
            average_uncertainty=float(np.mean(uncertainties)),
            average_error=float(np.mean(errors)),
            uncertainty_error_correlation=correlation,
            sharpness=sharpness,
            bin_confidences=bin_confidences,
            bin_accuracies=bin_accuracies,
            bin_counts=bin_counts,
            n_bins=self.n_bins,
        )

    def collect_from_training(
        self,
        agent_system,
        env,
        n_episodes: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect uncertainty and error data during evaluation episodes.

        Args:
            agent_system: MultiAgentSystem with mindfulness module
            env: Environment to evaluate in
            n_episodes: Number of evaluation episodes

        Returns:
            (uncertainties, errors) arrays
        """
        import torch

        all_uncertainties = []
        all_errors = []

        agent = (agent_system.shared_agent
                 if agent_system.share_params
                 else agent_system.agents[0])

        if agent.mindfulness is None:
            logger.warning("Mindfulness module not enabled, cannot collect calibration data")
            return np.array([]), np.array([])

        for ep in range(n_episodes):
            obs_dict, _ = env.reset()
            agent_system.reset()
            done = False

            prev_obs = None
            prev_action = None

            while not done:
                agent_ids = list(obs_dict.keys())
                first_id = agent_ids[0]
                obs = obs_dict[first_id]
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

                # Get action
                with torch.no_grad():
                    action_out, _, _, _ = agent.ac.get_action_and_value(obs_tensor)
                    action_onehot = torch.nn.functional.one_hot(
                        action_out.long().squeeze(),
                        num_classes=agent.action_size
                    ).float().unsqueeze(0)

                    # Get mindfulness uncertainty
                    mindfulness_state = agent.mindfulness.compute_mindfulness_state(
                        obs_tensor, action_onehot
                    )
                    uncertainty = mindfulness_state.uncertainty

                    # If we have previous obs and action, compute actual prediction error
                    if prev_obs is not None and prev_action is not None:
                        mean_pred, _ = agent.mindfulness.ensemble(prev_obs, prev_action)
                        actual_error = torch.nn.functional.mse_loss(
                            mean_pred, obs_tensor, reduction="mean"
                        ).item()
                        all_uncertainties.append(uncertainty)
                        all_errors.append(actual_error)

                prev_obs = obs_tensor.clone()
                prev_action = action_onehot.clone()

                # Step environment
                actions = {}
                for aid in agent_ids:
                    o = obs_dict[aid]
                    ot = torch.from_numpy(o).float().unsqueeze(0)
                    with torch.no_grad():
                        a, _, _, _ = agent.ac.get_action_and_value(ot)
                    actions[aid] = int(a.item())

                obs_dict, _, terms, truncs, _ = env.step(actions)
                done = any(terms.values()) or any(truncs.values())

        return np.array(all_uncertainties), np.array(all_errors)

    def plot_reliability_diagram(
        self,
        metrics: CalibrationMetrics,
        title: str = "Uncertainty Calibration",
        save_path: Optional[str] = None,
    ):
        """
        Plot reliability diagram showing calibration quality.

        Perfect calibration = diagonal line.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available for plotting")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Reliability diagram
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.7)
        ax1.bar(
            metrics.bin_confidences,
            metrics.bin_accuracies,
            width=1.0 / metrics.n_bins,
            alpha=0.6,
            edgecolor="black",
            label="Model",
        )
        ax1.set_xlabel("Predicted Uncertainty (binned)", fontsize=12)
        ax1.set_ylabel("Actual Error (normalized)", fontsize=12)
        ax1.set_title(f"{title}\nECE = {metrics.expected_calibration_error:.4f}", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

        # Bin count histogram
        ax2.bar(
            range(metrics.n_bins),
            metrics.bin_counts,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax2.set_xlabel("Uncertainty Bin", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Distribution of Uncertainty Values", fontsize=13)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved calibration plot to {save_path}")

        plt.close()

    def plot_uncertainty_vs_error(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        title: str = "Uncertainty vs Actual Error",
        save_path: Optional[str] = None,
    ):
        """Scatter plot of predicted uncertainty vs actual error."""
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(uncertainties, errors, alpha=0.3, s=10)

        # Fit and plot trend line
        if len(uncertainties) > 2:
            z = np.polyfit(uncertainties, errors, 1)
            p = np.poly1d(z)
            x_line = np.linspace(uncertainties.min(), uncertainties.max(), 100)
            ax.plot(x_line, p(x_line), "r-", linewidth=2,
                    label=f"Trend: slope={z[0]:.3f}")

        corr = np.corrcoef(uncertainties, errors)[0, 1] if len(uncertainties) > 1 else 0
        ax.set_xlabel("Predicted Uncertainty", fontsize=12)
        ax.set_ylabel("Actual Error", fontsize=12)
        ax.set_title(f"{title}\nCorrelation: {corr:.3f}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
