"""
Module A: Ethical Constraints for MARL
======================================
Multi-framework ethical reasoning with Lagrangian constraint optimization.

Key innovation: Formalizes ethics as learnable constraints rather than
reward shaping, enabling principled trade-offs between individual and social objectives.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum


class EthicalFramework(Enum):
    """Ethical frameworks for multi-perspective evaluation."""
    CONSEQUENTIALIST = "consequentialist"  # Outcome-based
    DEONTOLOGICAL = "deontological"        # Rule/duty-based
    VIRTUE = "virtue"                       # Character-based
    CARE = "care"                           # Relationship-based


@dataclass
class EthicalAssessment:
    """Result of ethical evaluation."""
    scores: Dict[str, float]  # Framework -> score
    aggregate_score: float
    constraint_violations: List[str]
    recommended_action_mask: Optional[np.ndarray] = None


class EthicalConstraints:
    """
    Defines and tracks ethical constraints.
    """

    def __init__(
        self,
        harm_threshold: float = 0.1,
        fairness_threshold: float = 0.3,
        min_cooperation: float = 0.2,
    ):
        self.harm_threshold = harm_threshold
        self.fairness_threshold = fairness_threshold
        self.min_cooperation = min_cooperation

        # Track violations for Lagrangian updates
        self.violation_history = {
            "harm": [],
            "fairness": [],
            "cooperation": []
        }

    def check_harm(self, action_effects: Dict) -> Tuple[bool, float]:
        """Check if action causes harm above threshold."""
        harm = action_effects.get("harm_caused", 0.0)
        violated = harm > self.harm_threshold
        violation_amount = max(0, harm - self.harm_threshold)
        return violated, violation_amount

    def check_fairness(self, rewards: np.ndarray) -> Tuple[bool, float]:
        """Check if reward distribution is fair (Gini coefficient)."""
        if len(rewards) < 2:
            return False, 0.0

        # Calculate Gini coefficient
        sorted_rewards = np.sort(rewards)
        n = len(sorted_rewards)
        cumsum = np.cumsum(sorted_rewards)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_rewards))) / (n * np.sum(sorted_rewards) + 1e-8) - (n + 1) / n

        violated = gini > self.fairness_threshold
        violation_amount = max(0, gini - self.fairness_threshold)
        return violated, violation_amount

    def check_cooperation(self, cooperation_rate: float) -> Tuple[bool, float]:
        """Check if cooperation is above minimum threshold."""
        violated = cooperation_rate < self.min_cooperation
        violation_amount = max(0, self.min_cooperation - cooperation_rate)
        return violated, violation_amount

    def evaluate_all(
        self,
        action_effects: Dict,
        rewards: np.ndarray,
        cooperation_rate: float
    ) -> Dict[str, Tuple[bool, float]]:
        """Evaluate all constraints."""
        return {
            "harm": self.check_harm(action_effects),
            "fairness": self.check_fairness(rewards),
            "cooperation": self.check_cooperation(cooperation_rate)
        }


class MultiFrameworkEthics(nn.Module):
    """
    Multi-framework ethical evaluation network.

    Learns to evaluate actions through multiple ethical lenses
    and aggregate into a single ethical score.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        framework_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()

        self.framework_weights = framework_weights or {
            "consequentialist": 0.25,
            "deontological": 0.25,
            "virtue": 0.25,
            "care": 0.25
        }

        input_size = state_size + action_size

        # Separate heads for each framework
        self.consequentialist_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.deontological_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.virtue_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.care_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate ethical score for state-action pair.

        Args:
            state: (batch, state_size)
            action: (batch, action_size) - one-hot or continuous

        Returns:
            aggregate_score: (batch, 1)
            framework_scores: Dict of (batch, 1) per framework
        """
        x = torch.cat([state, action], dim=-1)

        scores = {
            "consequentialist": self.consequentialist_head(x),
            "deontological": self.deontological_head(x),
            "virtue": self.virtue_head(x),
            "care": self.care_head(x)
        }

        # Weighted aggregation
        aggregate = sum(
            self.framework_weights[k] * v
            for k, v in scores.items()
        )

        return aggregate, scores


class LagrangianOptimizer:
    """
    Lagrangian multiplier optimizer for constraint satisfaction.

    Dynamically adjusts penalty weights based on constraint violations.
    """

    def __init__(
        self,
        constraint_names: List[str],
        initial_lambda: float = 0.1,
        lambda_lr: float = 0.01,
        lambda_max: float = 10.0
    ):
        self.constraint_names = constraint_names
        self.lambdas = {name: initial_lambda for name in constraint_names}
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max

        # Track for logging
        self.violation_ema = {name: 0.0 for name in constraint_names}
        self.ema_alpha = 0.1

    def compute_penalty(self, violations: Dict[str, float], device: str = "cpu") -> torch.Tensor:
        """
        Compute total constraint penalty.

        Args:
            violations: Dict of constraint_name -> violation_amount
            device: Device for the output tensor

        Returns:
            Total penalty (scalar tensor)
        """
        total = 0.0
        for name, violation in violations.items():
            if name in self.lambdas:
                total += self.lambdas[name] * violation
        return torch.tensor(total, dtype=torch.float32, device=device)

    def update(self, violations: Dict[str, float]):
        """
        Update Lagrange multipliers based on violations.

        Uses dual gradient ascent: increase lambda when violated,
        decrease when satisfied.
        """
        for name, violation in violations.items():
            if name in self.lambdas:
                # Update EMA
                self.violation_ema[name] = (
                    self.ema_alpha * violation +
                    (1 - self.ema_alpha) * self.violation_ema[name]
                )

                # Gradient step on lambda
                self.lambdas[name] += self.lambda_lr * violation
                self.lambdas[name] = np.clip(self.lambdas[name], 0, self.lambda_max)

    def get_stats(self) -> Dict[str, float]:
        """Get current lambda values and violation EMAs."""
        stats = {}
        for name in self.constraint_names:
            stats[f"lambda_{name}"] = self.lambdas[name]
            stats[f"violation_ema_{name}"] = self.violation_ema[name]
        return stats


class EthicsModule(nn.Module):
    """
    Complete ethics module integrating evaluation and constraints.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config,
    ):
        super().__init__()

        self.config = config

        # Multi-framework evaluator
        self.evaluator = MultiFrameworkEthics(
            state_size=state_size,
            action_size=action_size,
            framework_weights={
                "consequentialist": config.consequentialist_weight,
                "deontological": config.deontological_weight,
                "virtue": config.virtue_weight,
                "care": config.care_weight,
            }
        )

        # Constraint checker
        self.constraints = EthicalConstraints(
            harm_threshold=config.harm_threshold,
            fairness_threshold=config.fairness_threshold,
        )

        # Lagrangian optimizer
        self.lagrangian = LagrangianOptimizer(
            constraint_names=["harm", "fairness", "cooperation"],
            initial_lambda=config.lagrange_init,
            lambda_lr=config.lagrange_lr,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluate ethical score."""
        return self.evaluator(state, action)

    def compute_constraint_loss(
        self,
        action_effects: Dict,
        rewards: np.ndarray,
        cooperation_rate: float
    ) -> torch.Tensor:
        """Compute Lagrangian penalty for constraint violations."""
        violations = self.constraints.evaluate_all(
            action_effects, rewards, cooperation_rate
        )

        violation_amounts = {
            name: amount for name, (_, amount) in violations.items()
        }

        # Update multipliers
        self.lagrangian.update(violation_amounts)

        # Compute penalty
        return self.lagrangian.compute_penalty(violation_amounts)


def create_ethics_module(config, state_size: int, action_size: int) -> EthicsModule:
    """Create ethics module from config."""
    return EthicsModule(
        state_size=state_size,
        action_size=action_size,
        config=config
    )
