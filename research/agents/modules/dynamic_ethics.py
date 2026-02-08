"""
Dynamic Ethical Weights
========================
Context-dependent ethical framework weighting.

Instead of fixed weights (0.25 each), learns to adapt framework importance
based on the current situation:
- High-consequence situations -> more consequentialist weight
- Clear rule violations -> more deontological weight
- Character-defining moments -> more virtue ethics weight
- Relational/care situations -> more care ethics weight

References:
- Moral Foundations Theory (Haidt & Graham, 2007)
- Context-dependent ethics (Sunstein, 2005)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ContextualWeightNetwork(nn.Module):
    """
    Learns to assign ethical framework weights based on context.

    Takes state as input, outputs a softmax distribution over
    ethical frameworks, allowing the system to emphasize different
    moral reasoning approaches in different situations.
    """

    def __init__(
        self,
        state_size: int,
        n_frameworks: int = 4,
        hidden_size: int = 64,
        temperature: float = 1.0,
        min_weight: float = 0.05,
    ):
        super().__init__()

        self.n_frameworks = n_frameworks
        self.temperature = temperature
        self.min_weight = min_weight

        self.weight_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_frameworks),
        )

        # Track weight history for analysis
        self.weight_history = []

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute context-dependent framework weights.

        Args:
            state: Current state observation (batch, state_size)

        Returns:
            weights: Framework weights (batch, n_frameworks), sum to 1
        """
        logits = self.weight_net(state) / self.temperature
        weights = F.softmax(logits, dim=-1)

        # Apply minimum weight constraint
        weights = weights * (1 - self.n_frameworks * self.min_weight) + self.min_weight

        return weights

    def get_attention_weights(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get weights with the raw logits for interpretability.

        Returns:
            weights: Softmax weights
            logits: Raw logits (for analysis)
        """
        logits = self.weight_net(state) / self.temperature
        weights = F.softmax(logits, dim=-1)
        weights = weights * (1 - self.n_frameworks * self.min_weight) + self.min_weight
        return weights, logits


class DynamicMultiFrameworkEthics(nn.Module):
    """
    Multi-framework ethics with dynamic context-dependent weighting.

    Extends the base MultiFrameworkEthics by replacing fixed weights
    with a learned contextual weighting network.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        weight_hidden_size: int = 64,
        temperature: float = 1.0,
        framework_names: Optional[list] = None,
    ):
        super().__init__()

        self.framework_names = framework_names or [
            "consequentialist", "deontological", "virtue", "care"
        ]
        n_frameworks = len(self.framework_names)

        input_size = state_size + action_size

        # Individual framework evaluation heads
        self.framework_heads = nn.ModuleDict()
        for name in self.framework_names:
            self.framework_heads[name] = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )

        # Dynamic weighting network
        self.weight_network = ContextualWeightNetwork(
            state_size=state_size,
            n_frameworks=n_frameworks,
            hidden_size=weight_hidden_size,
            temperature=temperature,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Evaluate ethical score with dynamic weights.

        Args:
            state: (batch, state_size)
            action: (batch, action_size)

        Returns:
            aggregate_score: (batch, 1)
            framework_scores: Dict of framework -> (batch, 1)
            weights: (batch, n_frameworks) - the dynamic weights used
        """
        x = torch.cat([state, action], dim=-1)

        # Evaluate each framework
        scores = {}
        score_tensor = []
        for name in self.framework_names:
            score = self.framework_heads[name](x)
            scores[name] = score
            score_tensor.append(score)

        score_tensor = torch.cat(score_tensor, dim=-1)  # (batch, n_frameworks)

        # Get dynamic weights
        weights = self.weight_network(state)  # (batch, n_frameworks)

        # Weighted aggregation
        aggregate = (score_tensor * weights).sum(dim=-1, keepdim=True)

        return aggregate, scores, weights

    def get_weight_explanation(
        self, state: torch.Tensor
    ) -> Dict[str, float]:
        """
        Get human-readable weight explanation for a state.

        Returns:
            Dict mapping framework names to their weights
        """
        with torch.no_grad():
            weights = self.weight_network(state)

        explanation = {}
        for i, name in enumerate(self.framework_names):
            explanation[name] = float(weights[0, i].item())

        return explanation

    def compute_weight_diversity_loss(
        self,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Regularization loss encouraging weight diversity.

        Prevents the network from always using the same framework
        by penalizing low entropy in the weight distribution.
        """
        entropy = -torch.sum(weights * torch.log(weights + 1e-12), dim=-1)
        max_entropy = np.log(len(self.framework_names))
        # Negative because we want to MAXIMIZE entropy
        return -(entropy / max_entropy).mean()
