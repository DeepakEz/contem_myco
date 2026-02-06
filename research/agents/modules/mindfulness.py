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


class MCDropoutPredictor(nn.Module):
    """
    MC Dropout predictor for uncertainty estimation.

    Gal & Ghahramani, 2016 - "Dropout as a Bayesian Approximation"

    Uses dropout at test time to sample from approximate posterior,
    providing uncertainty estimates without explicit ensembles.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 128,
        dropout_rate: float = 0.1,
        n_forward_passes: int = 10,
    ):
        super().__init__()

        self.n_forward_passes = n_forward_passes
        self.dropout_rate = dropout_rate

        self.predictor = nn.Sequential(
            nn.Linear(obs_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, obs_size)
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional MC Dropout uncertainty.

        Args:
            obs: Current observation
            action: Action taken
            return_uncertainty: Whether to compute MC Dropout uncertainty

        Returns:
            prediction: Mean prediction
            uncertainty: Optional uncertainty estimate (variance)
        """
        x = torch.cat([obs, action], dim=-1)

        if not return_uncertainty or not self.training:
            # Single forward pass
            pred = self.predictor(x)
            return pred, None

        # MC Dropout: multiple forward passes with dropout
        self.train()  # Enable dropout
        predictions = []

        for _ in range(self.n_forward_passes):
            pred = self.predictor(x)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # (n_passes, batch, obs_size)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=-1)  # (batch,)

        return mean_pred, uncertainty

    def compute_mc_uncertainty(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MC Dropout uncertainty explicitly.

        Returns:
            mean_pred: Mean prediction across MC samples
            uncertainty: Variance across MC samples (epistemic uncertainty)
        """
        x = torch.cat([obs, action], dim=-1)

        # Enable dropout for inference
        was_training = self.training
        self.train()

        predictions = []
        for _ in range(self.n_forward_passes):
            pred = self.predictor(x)
            predictions.append(pred)

        # Restore training state
        self.train(was_training)

        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=-1)

        return mean_pred, uncertainty


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
    2. MC Dropout predictor for Bayesian uncertainty
    3. Gating mechanism for policy switching
    4. Action smoother for stability

    Uncertainty estimation combines:
    - Ensemble disagreement (epistemic uncertainty from diverse models)
    - MC Dropout variance (Bayesian approximation)
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        config,
        use_mc_dropout: bool = True,
        mc_dropout_rate: float = 0.1,
        mc_n_passes: int = 10,
    ):
        super().__init__()

        self.config = config
        self.obs_size = obs_size
        self.action_size = action_size
        self.use_mc_dropout = use_mc_dropout

        # Ensemble for uncertainty
        self.ensemble = EnsemblePredictor(
            obs_size=obs_size,
            action_size=action_size,
            ensemble_size=config.ensemble_size
        )

        # MC Dropout predictor (optional, complementary uncertainty)
        self.mc_dropout = None
        if use_mc_dropout:
            self.mc_dropout = MCDropoutPredictor(
                obs_size=obs_size,
                action_size=action_size,
                dropout_rate=mc_dropout_rate,
                n_forward_passes=mc_n_passes,
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
        self.mc_uncertainty_mean = 0.0
        self.mc_uncertainty_std = 1.0
        self.update_count = 0

    def compute_mindfulness_state(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: Optional[torch.Tensor] = None
    ) -> MindfulnessState:
        """
        Compute current mindfulness state.

        Combines uncertainty from:
        - Ensemble disagreement
        - MC Dropout variance (if enabled)

        Args:
            obs: Current observation
            action: Action taken (or to be taken)
            next_obs: Next observation (for surprise computation)

        Returns:
            MindfulnessState with uncertainty and surprise metrics
        """
        with torch.no_grad():
            # Ensemble uncertainty
            _, ensemble_uncertainty = self.ensemble(obs, action)

            # MC Dropout uncertainty (if enabled)
            mc_uncertainty = torch.zeros_like(ensemble_uncertainty)
            if self.mc_dropout is not None:
                _, mc_uncertainty = self.mc_dropout.compute_mc_uncertainty(obs, action)
                # Normalize MC uncertainty
                mc_uncertainty_norm = (mc_uncertainty - self.mc_uncertainty_mean) / (self.mc_uncertainty_std + 1e-8)
            else:
                mc_uncertainty_norm = torch.zeros_like(ensemble_uncertainty)

            # Combined uncertainty (average of normalized uncertainties)
            ensemble_uncertainty_norm = (ensemble_uncertainty - self.uncertainty_mean) / (self.uncertainty_std + 1e-8)
            combined_uncertainty = (ensemble_uncertainty_norm + mc_uncertainty_norm) / 2 if self.mc_dropout else ensemble_uncertainty_norm

            if next_obs is not None:
                surprise = self.ensemble.compute_surprise(obs, action, next_obs)
            else:
                surprise = torch.zeros_like(ensemble_uncertainty)

            # Normalize surprise
            surprise_norm = (surprise - self.surprise_mean) / (self.surprise_std + 1e-8)

            # Compute gate using combined uncertainty
            gate = self.gate(obs, surprise_norm, combined_uncertainty)

            is_conservative = gate.mean().item() > 0.5

        return MindfulnessState(
            surprise_level=surprise.mean().item(),
            uncertainty=ensemble_uncertainty.mean().item() + (mc_uncertainty.mean().item() if self.mc_dropout else 0),
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
        uncertainty: torch.Tensor,
        mc_uncertainty: Optional[torch.Tensor] = None,
    ):
        """Update running statistics for normalization."""
        self.update_count += 1
        alpha = 0.01

        self.surprise_mean = (1 - alpha) * self.surprise_mean + alpha * surprise.mean().item()
        self.surprise_std = (1 - alpha) * self.surprise_std + alpha * (surprise.std().item() + 1e-8)
        self.uncertainty_mean = (1 - alpha) * self.uncertainty_mean + alpha * uncertainty.mean().item()
        self.uncertainty_std = (1 - alpha) * self.uncertainty_std + alpha * (uncertainty.std().item() + 1e-8)

        if mc_uncertainty is not None:
            self.mc_uncertainty_mean = (1 - alpha) * self.mc_uncertainty_mean + alpha * mc_uncertainty.mean().item()
            self.mc_uncertainty_std = (1 - alpha) * self.mc_uncertainty_std + alpha * (mc_uncertainty.std().item() + 1e-8)

    def compute_prediction_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prediction loss for training the ensemble and MC Dropout models.
        """
        # Ensemble loss
        total_loss = 0.0
        for model in self.ensemble.models:
            pred = model(obs, action)
            loss = F.mse_loss(pred, next_obs)
            total_loss += loss

        ensemble_loss = total_loss / len(self.ensemble.models)

        # MC Dropout loss (if enabled)
        mc_loss = torch.tensor(0.0, device=obs.device)
        if self.mc_dropout is not None:
            mc_pred, _ = self.mc_dropout(obs, action, return_uncertainty=False)
            mc_loss = F.mse_loss(mc_pred, next_obs)

        # Combined loss
        if self.mc_dropout is not None:
            return 0.7 * ensemble_loss + 0.3 * mc_loss
        else:
            return ensemble_loss

    def state_dict(self):
        """Get state dict including running statistics."""
        base_state = super().state_dict()
        base_state['_running_stats'] = {
            'surprise_mean': self.surprise_mean,
            'surprise_std': self.surprise_std,
            'uncertainty_mean': self.uncertainty_mean,
            'uncertainty_std': self.uncertainty_std,
            'mc_uncertainty_mean': self.mc_uncertainty_mean,
            'mc_uncertainty_std': self.mc_uncertainty_std,
            'update_count': self.update_count,
        }
        return base_state

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict including running statistics."""
        # Extract running stats before loading
        running_stats = state_dict.pop('_running_stats', None)
        super().load_state_dict(state_dict, strict=strict)
        if running_stats:
            self.surprise_mean = running_stats['surprise_mean']
            self.surprise_std = running_stats['surprise_std']
            self.uncertainty_mean = running_stats['uncertainty_mean']
            self.uncertainty_std = running_stats['uncertainty_std']
            self.mc_uncertainty_mean = running_stats.get('mc_uncertainty_mean', 0.0)
            self.mc_uncertainty_std = running_stats.get('mc_uncertainty_std', 1.0)
            self.update_count = running_stats['update_count']


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
