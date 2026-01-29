"""
Module C: Mindfulness for Robust Decision-Making
=================================================
Uncertainty-aware gating mechanism for robustness under distribution shift.

Key innovation: Agents estimate their own uncertainty and switch to
conservative behavior when facing novel/surprising situations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MindfulnessState:
    """Internal state of the mindfulness module."""
    surprise_level: float
    uncertainty: float
    action_entropy: float
    is_conservative: bool
    attention_weights: Optional[np.ndarray] = None


class PredictionModel(nn.Module):
    """
    World model for estimating surprise/uncertainty.
    Predicts next observation given current state and action.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 128
    ):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(obs_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, obs_size)
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next observation."""
        x = torch.cat([obs, action], dim=-1)
        return self.predictor(x)


class EnsemblePredictor(nn.Module):
    """
    Ensemble of prediction models for uncertainty estimation.
    Disagreement between ensemble members indicates uncertainty.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        ensemble_size: int = 3,
        hidden_size: int = 128
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList([
            PredictionModel(obs_size, action_size, hidden_size)
            for _ in range(ensemble_size)
        ])

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prediction and uncertainty.

        Returns:
            mean_pred: Mean prediction across ensemble
            uncertainty: Variance across ensemble (epistemic uncertainty)
        """
        predictions = torch.stack([
            model(obs, action) for model in self.models
        ], dim=0)  # (ensemble, batch, obs_size)

        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=-1)  # (batch,)

        return mean_pred, uncertainty

    def compute_surprise(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute surprise as prediction error.
        """
        mean_pred, _ = self.forward(obs, action)
        surprise = F.mse_loss(mean_pred, next_obs, reduction='none').mean(dim=-1)
        return surprise


class GatingMechanism(nn.Module):
    """
    Learned gating between reactive and conservative policies.
    """

    def __init__(
        self,
        obs_size: int,
        hidden_size: int = 64,
        surprise_threshold: float = 0.5
    ):
        super().__init__()

        self.surprise_threshold = surprise_threshold

        # Input: obs + surprise + uncertainty
        self.gate_net = nn.Sequential(
            nn.Linear(obs_size + 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        obs: torch.Tensor,
        surprise: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gating weight.

        Returns:
            gate: (batch,) in [0, 1], where 1 = fully conservative
        """
        x = torch.cat([
            obs,
            surprise.unsqueeze(-1),
            uncertainty.unsqueeze(-1)
        ], dim=-1)

        gate = self.gate_net(x).squeeze(-1)
        return gate


class ActionSmoother:
    """
    Temporal smoothing of actions to reduce oscillation.
    """

    def __init__(self, smoothing: float = 0.3):
        self.smoothing = smoothing
        self.prev_action = None

    def smooth(self, action: np.ndarray) -> np.ndarray:
        """Apply exponential moving average to actions."""
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action

        smoothed = (1 - self.smoothing) * action + self.smoothing * self.prev_action
        self.prev_action = smoothed.copy()
        return smoothed

    def reset(self):
        """Reset smoother state."""
        self.prev_action = None


class MindfulnessModule(nn.Module):
    """
    Complete mindfulness module for robust decision-making.

    Components:
    1. Ensemble predictor for uncertainty estimation
    2. Gating mechanism for policy switching
    3. Action smoother for stability
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        config,
    ):
        super().__init__()

        self.config = config
        self.obs_size = obs_size
        self.action_size = action_size

        # Ensemble for uncertainty
        self.ensemble = EnsemblePredictor(
            obs_size=obs_size,
            action_size=action_size,
            ensemble_size=config.ensemble_size
        )

        # Gating mechanism
        self.gate = GatingMechanism(
            obs_size=obs_size,
            surprise_threshold=config.surprise_threshold
        )

        # Action smoother (one per agent, managed externally)
        self.action_smoothing = config.action_smoothing

        # Conservative policy entropy bonus
        self.conservative_entropy_bonus = config.conservative_entropy_bonus

        # Running statistics for normalization
        self.surprise_mean = 0.0
        self.surprise_std = 1.0
        self.uncertainty_mean = 0.0
        self.uncertainty_std = 1.0
        self.update_count = 0

    def compute_mindfulness_state(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: Optional[torch.Tensor] = None
    ) -> MindfulnessState:
        """
        Compute current mindfulness state.

        Args:
            obs: Current observation
            action: Action taken (or to be taken)
            next_obs: Next observation (for surprise computation)

        Returns:
            MindfulnessState with uncertainty and surprise metrics
        """
        with torch.no_grad():
            _, uncertainty = self.ensemble(obs, action)

            if next_obs is not None:
                surprise = self.ensemble.compute_surprise(obs, action, next_obs)
            else:
                surprise = torch.zeros_like(uncertainty)

            # Normalize
            surprise_norm = (surprise - self.surprise_mean) / (self.surprise_std + 1e-8)
            uncertainty_norm = (uncertainty - self.uncertainty_mean) / (self.uncertainty_std + 1e-8)

            # Compute gate
            gate = self.gate(obs, surprise_norm, uncertainty_norm)

            is_conservative = gate.mean().item() > 0.5

        return MindfulnessState(
            surprise_level=surprise.mean().item(),
            uncertainty=uncertainty.mean().item(),
            action_entropy=0.0,  # Computed externally from policy
            is_conservative=is_conservative,
        )

    def get_entropy_bonus(self, gate: torch.Tensor) -> torch.Tensor:
        """
        Get entropy bonus for conservative mode.
        Higher gate = more conservative = more entropy encouraged.
        """
        return gate * self.conservative_entropy_bonus

    def update_statistics(
        self,
        surprise: torch.Tensor,
        uncertainty: torch.Tensor
    ):
        """Update running statistics for normalization."""
        self.update_count += 1
        alpha = 0.01

        self.surprise_mean = (1 - alpha) * self.surprise_mean + alpha * surprise.mean().item()
        self.surprise_std = (1 - alpha) * self.surprise_std + alpha * surprise.std().item()
        self.uncertainty_mean = (1 - alpha) * self.uncertainty_mean + alpha * uncertainty.mean().item()
        self.uncertainty_std = (1 - alpha) * self.uncertainty_std + alpha * uncertainty.std().item()

    def compute_prediction_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prediction loss for training the ensemble.
        """
        total_loss = 0.0
        for model in self.ensemble.models:
            pred = model(obs, action)
            loss = F.mse_loss(pred, next_obs)
            total_loss += loss

        return total_loss / len(self.ensemble.models)


def create_mindfulness_module(
    config,
    obs_size: int,
    action_size: int
) -> MindfulnessModule:
    """Create mindfulness module from config."""
    return MindfulnessModule(
        obs_size=obs_size,
        action_size=action_size,
        config=config
    )
