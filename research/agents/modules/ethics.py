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


class AdaptiveLagrangianOptimizer(nn.Module):
    """
    Adaptive Lagrangian optimizer with learnable multipliers.

    Stooke et al., 2020 - "Responsive Safety in Reinforcement Learning"

    Key improvements over basic Lagrangian:
    - Learnable lambda via gradient descent (end-to-end)
    - PID-based constraint satisfaction
    - Soft constraint thresholds with margin
    """

    def __init__(
        self,
        constraint_names: List[str],
        initial_lambda: float = 0.1,
        lambda_lr: float = 0.01,
        lambda_max: float = 10.0,
        pid_kp: float = 1.0,
        pid_ki: float = 0.1,
        pid_kd: float = 0.01,
        use_log_lambda: bool = True,
    ):
        super().__init__()

        self.constraint_names = constraint_names
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        self.use_log_lambda = use_log_lambda

        # PID coefficients
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd

        # Learnable lambda parameters (in log space for positivity)
        if use_log_lambda:
            init_log = np.log(initial_lambda + 1e-8)
            self.log_lambdas = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(init_log))
                for name in constraint_names
            })
        else:
            self.raw_lambdas = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(initial_lambda))
                for name in constraint_names
            })

        # PID state tracking
        self.violation_integral = {name: 0.0 for name in constraint_names}
        self.prev_violation = {name: 0.0 for name in constraint_names}
        self.violation_ema = {name: 0.0 for name in constraint_names}
        self.ema_alpha = 0.1

    @property
    def lambdas(self) -> Dict[str, torch.Tensor]:
        """Get current lambda values."""
        if self.use_log_lambda:
            return {
                name: torch.exp(param).clamp(max=self.lambda_max)
                for name, param in self.log_lambdas.items()
            }
        else:
            return {
                name: torch.relu(param).clamp(max=self.lambda_max)
                for name, param in self.raw_lambdas.items()
            }

    def compute_penalty(
        self,
        violations: Dict[str, torch.Tensor],
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Compute total constraint penalty with learnable lambdas.

        Args:
            violations: Dict of constraint_name -> violation_amount (tensor)
            device: Device for computation

        Returns:
            Total penalty (differentiable tensor)
        """
        total = torch.tensor(0.0, device=device)
        lambdas = self.lambdas

        for name, violation in violations.items():
            if name in lambdas:
                # Convert to tensor if needed
                if not isinstance(violation, torch.Tensor):
                    violation = torch.tensor(violation, dtype=torch.float32, device=device)

                # Move lambda to correct device
                lam = lambdas[name].to(device)
                total = total + lam * violation

        return total

    def compute_pid_adjustment(self, constraint_name: str, violation: float) -> float:
        """
        Compute PID-based lambda adjustment.

        Args:
            constraint_name: Name of the constraint
            violation: Current violation amount

        Returns:
            Adjustment to apply to lambda
        """
        # Proportional term
        p_term = self.pid_kp * violation

        # Integral term (accumulated violation)
        self.violation_integral[constraint_name] += violation
        i_term = self.pid_ki * self.violation_integral[constraint_name]

        # Derivative term (change in violation)
        d_term = self.pid_kd * (violation - self.prev_violation[constraint_name])
        self.prev_violation[constraint_name] = violation

        return p_term + i_term + d_term

    def update_non_differentiable(self, violations: Dict[str, float]):
        """
        Update for non-differentiable constraint tracking.

        Use this for logging and PID-based adjustments when
        gradients aren't flowing through the constraint loss.
        """
        for name, violation in violations.items():
            if name in self.violation_ema:
                # Update EMA for monitoring
                self.violation_ema[name] = (
                    self.ema_alpha * violation +
                    (1 - self.ema_alpha) * self.violation_ema[name]
                )

                # PID-based manual adjustment (optional)
                adjustment = self.compute_pid_adjustment(name, violation)

                if self.use_log_lambda and name in self.log_lambdas:
                    with torch.no_grad():
                        self.log_lambdas[name].add_(self.lambda_lr * adjustment)
                        # Clamp in log space
                        max_log = np.log(self.lambda_max)
                        self.log_lambdas[name].clamp_(max=max_log)

    def get_stats(self) -> Dict[str, float]:
        """Get current lambda values and violation EMAs."""
        stats = {}
        lambdas = self.lambdas

        for name in self.constraint_names:
            if name in lambdas:
                stats[f"lambda_{name}"] = lambdas[name].item()
            stats[f"violation_ema_{name}"] = self.violation_ema.get(name, 0.0)
            stats[f"violation_integral_{name}"] = self.violation_integral.get(name, 0.0)

        return stats

    def reset_pid_state(self):
        """Reset PID integral and derivative state."""
        for name in self.constraint_names:
            self.violation_integral[name] = 0.0
            self.prev_violation[name] = 0.0


class EthicsModule(nn.Module):
    """
    Complete ethics module integrating evaluation and constraints.

    Features:
    - Multi-framework ethical evaluation
    - Constraint checking (harm, fairness, cooperation)
    - Adaptive Lagrangian optimization with PID control
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config,
        use_adaptive_lagrangian: bool = True,
    ):
        super().__init__()

        self.config = config
        self.use_adaptive_lagrangian = use_adaptive_lagrangian

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

        # Lagrangian optimizer (adaptive or basic)
        if use_adaptive_lagrangian:
            self.lagrangian = AdaptiveLagrangianOptimizer(
                constraint_names=["harm", "fairness", "cooperation"],
                initial_lambda=config.lagrange_init,
                lambda_lr=config.lagrange_lr,
                pid_kp=getattr(config, 'pid_kp', 1.0),
                pid_ki=getattr(config, 'pid_ki', 0.1),
                pid_kd=getattr(config, 'pid_kd', 0.01),
            )
        else:
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
        cooperation_rate: float,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Compute Lagrangian penalty for constraint violations.

        For adaptive Lagrangian, returns differentiable loss.
        """
        violations = self.constraints.evaluate_all(
            action_effects, rewards, cooperation_rate
        )

        violation_amounts = {
            name: amount for name, (_, amount) in violations.items()
        }

        # Update multipliers (non-differentiable path)
        if self.use_adaptive_lagrangian:
            self.lagrangian.update_non_differentiable(violation_amounts)
        else:
            self.lagrangian.update(violation_amounts)

        # Compute penalty
        return self.lagrangian.compute_penalty(violation_amounts, device=device)

    def compute_differentiable_constraint_loss(
        self,
        violations: Dict[str, torch.Tensor],
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Compute differentiable constraint loss for end-to-end training.

        Args:
            violations: Dict of constraint violations as tensors
            device: Device for computation

        Returns:
            Differentiable penalty loss
        """
        return self.lagrangian.compute_penalty(violations, device=device)

    def get_lagrangian_stats(self) -> Dict[str, float]:
        """Get statistics about Lagrangian multipliers."""
        return self.lagrangian.get_stats()


def create_ethics_module(config, state_size: int, action_size: int) -> EthicsModule:
    """Create ethics module from config."""
    return EthicsModule(
        state_size=state_size,
        action_size=action_size,
        config=config
    )
