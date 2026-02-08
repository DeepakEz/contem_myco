"""
Interpretable Ethical Reasoning
================================
Attention-based ethical evaluation with natural language explanations.

Provides:
- Attention mechanisms showing which ethical framework most influenced each decision
- Feature attribution for ethical scores
- Natural language explanation generation
- Decision audit trail
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EthicalExplanation:
    """Human-readable explanation of an ethical assessment."""
    aggregate_score: float
    framework_scores: Dict[str, float]
    framework_weights: Dict[str, float]
    dominant_framework: str
    key_features: List[Tuple[str, float]]
    explanation_text: str
    concerns: List[str]
    confidence: float


class AttentiveEthicsEvaluator(nn.Module):
    """
    Ethical evaluation with attention-based interpretability.

    Uses multi-head attention to identify which state features
    are most relevant for each ethical framework's assessment.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        n_attention_heads: int = 4,
        framework_names: Optional[List[str]] = None,
    ):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.framework_names = framework_names or [
            "consequentialist", "deontological", "virtue", "care"
        ]
        n_frameworks = len(self.framework_names)
        input_size = state_size + action_size

        # Shared feature projection
        self.feature_proj = nn.Linear(input_size, hidden_size)

        # Multi-head attention for each framework
        self.framework_attention = nn.ModuleDict()
        self.framework_heads = nn.ModuleDict()

        for name in self.framework_names:
            # Attention layer: computes attention weights over input features
            self.framework_attention[name] = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=n_attention_heads,
                batch_first=True,
            )
            # Score head
            self.framework_heads[name] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid(),
            )

        # Store attention weights for interpretation
        self._last_attention_weights = {}

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Evaluate ethical score with attention-based interpretation.

        Args:
            state: (batch, state_size)
            action: (batch, action_size)
            return_attention: Whether to return attention weight maps

        Returns:
            scores: Dict of framework -> (batch, 1)
            attention_weights: Optional dict of framework -> (batch, n_heads, seq_len, seq_len)
        """
        x = torch.cat([state, action], dim=-1)

        # Project to hidden space
        # Reshape to sequence: each input dimension is a "token"
        features = self.feature_proj(x)  # (batch, hidden_size)
        # Create sequence of feature vectors for attention
        features_seq = features.unsqueeze(1)  # (batch, 1, hidden_size)

        scores = {}
        attention_maps = {} if return_attention else None

        for name in self.framework_names:
            # Self-attention
            attn_out, attn_weights = self.framework_attention[name](
                features_seq, features_seq, features_seq,
                need_weights=True,
            )
            # attn_out: (batch, 1, hidden_size)

            score = self.framework_heads[name](attn_out.squeeze(1))
            scores[name] = score

            self._last_attention_weights[name] = attn_weights.detach()

            if return_attention:
                attention_maps[name] = attn_weights.detach()

        return scores, attention_maps

    def get_feature_importance(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance for each framework via gradient-based attribution.

        Args:
            state: Single state observation
            action: Single action
            feature_names: Optional names for input features

        Returns:
            Dict of framework -> [(feature_name, importance_score)]
        """
        x = torch.cat([state, action], dim=-1)
        x = x.detach().requires_grad_(True)

        n_features = x.shape[-1]
        if feature_names is None:
            state_names = [f"s_{i}" for i in range(self.state_size)]
            action_names = [f"a_{i}" for i in range(self.action_size)]
            feature_names = state_names + action_names

        importances = {}

        for name in self.framework_names:
            if x.grad is not None:
                x.grad.zero_()

            features = self.feature_proj(x)
            features_seq = features.unsqueeze(1)
            attn_out, _ = self.framework_attention[name](
                features_seq, features_seq, features_seq,
            )
            score = self.framework_heads[name](attn_out.squeeze(1))
            score.backward(retain_graph=True)

            if x.grad is not None:
                grad_importance = x.grad.abs().squeeze().detach().cpu().numpy()
                # Normalize
                total = grad_importance.sum() + 1e-12
                grad_importance = grad_importance / total

                top_features = sorted(
                    zip(feature_names[:n_features], grad_importance),
                    key=lambda t: t[1],
                    reverse=True,
                )
                importances[name] = top_features[:10]
            else:
                importances[name] = []

        return importances

    def generate_explanation(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        framework_weights: Optional[Dict[str, float]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> EthicalExplanation:
        """
        Generate a complete ethical explanation for a state-action pair.

        Args:
            state: Single state observation (1, state_size)
            action: Single action (1, action_size)
            framework_weights: Optional weights for aggregation
            feature_names: Optional names for input features

        Returns:
            EthicalExplanation with scores, attribution, and text
        """
        if framework_weights is None:
            framework_weights = {n: 1.0 / len(self.framework_names)
                                for n in self.framework_names}

        with torch.no_grad():
            scores, attention = self.forward(state, action, return_attention=True)

        # Extract scores
        score_values = {
            name: float(score.item()) for name, score in scores.items()
        }

        # Aggregate score
        aggregate = sum(
            framework_weights.get(name, 0.25) * score_values[name]
            for name in self.framework_names
        )

        # Dominant framework
        dominant = max(score_values, key=lambda k: score_values[k] * framework_weights.get(k, 0.25))

        # Feature importance
        importances = self.get_feature_importance(state, action, feature_names)
        key_features = importances.get(dominant, [])[:5]

        # Generate text explanation
        explanation_parts = []

        # Overall assessment
        if aggregate > 0.7:
            explanation_parts.append("This action is ethically favorable.")
        elif aggregate > 0.4:
            explanation_parts.append("This action has mixed ethical implications.")
        else:
            explanation_parts.append("This action raises significant ethical concerns.")

        # Framework-specific insights
        framework_descriptions = {
            "consequentialist": "outcome-based",
            "deontological": "duty/rule-based",
            "virtue": "character-based",
            "care": "relationship/care-based",
        }

        for name in self.framework_names:
            score = score_values[name]
            desc = framework_descriptions.get(name, name)
            weight = framework_weights.get(name, 0.25)

            if score > 0.7:
                explanation_parts.append(
                    f"From a {desc} perspective (weight={weight:.2f}): "
                    f"Strongly positive (score={score:.2f})."
                )
            elif score < 0.3:
                explanation_parts.append(
                    f"From a {desc} perspective (weight={weight:.2f}): "
                    f"Concerning (score={score:.2f})."
                )

        explanation_parts.append(
            f"The {dominant} framework had the strongest influence on this assessment."
        )

        # Concerns
        concerns = []
        for name, score in score_values.items():
            if score < 0.3:
                concerns.append(
                    f"Low {name} score ({score:.2f}) suggests "
                    f"potential {name} issues."
                )

        # Confidence based on agreement between frameworks
        scores_array = np.array(list(score_values.values()))
        confidence = float(1.0 - np.std(scores_array))

        return EthicalExplanation(
            aggregate_score=aggregate,
            framework_scores=score_values,
            framework_weights=framework_weights,
            dominant_framework=dominant,
            key_features=key_features,
            explanation_text=" ".join(explanation_parts),
            concerns=concerns,
            confidence=confidence,
        )


class EthicsAuditTrail:
    """
    Records ethical decision audit trail for post-hoc analysis.

    Stores all ethical evaluations, explanations, and decisions
    for later review by human evaluators or automated analysis.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.trail: List[Dict] = []

    def record(
        self,
        step: int,
        agent_id: int,
        state: np.ndarray,
        action: int,
        explanation: EthicalExplanation,
        chosen: bool,
    ):
        """Record an ethical evaluation in the audit trail."""
        entry = {
            "step": step,
            "agent_id": agent_id,
            "aggregate_score": explanation.aggregate_score,
            "framework_scores": explanation.framework_scores,
            "framework_weights": explanation.framework_weights,
            "dominant_framework": explanation.dominant_framework,
            "concerns": explanation.concerns,
            "confidence": explanation.confidence,
            "action_chosen": chosen,
        }
        self.trail.append(entry)

        if len(self.trail) > self.max_size:
            self.trail = self.trail[-self.max_size:]

    def get_statistics(self) -> Dict:
        """Get aggregate statistics from the audit trail."""
        if not self.trail:
            return {}

        scores = [e["aggregate_score"] for e in self.trail]
        concerns_count = sum(len(e["concerns"]) for e in self.trail)
        frameworks_dominant = {}
        for e in self.trail:
            dom = e["dominant_framework"]
            frameworks_dominant[dom] = frameworks_dominant.get(dom, 0) + 1

        return {
            "n_evaluations": len(self.trail),
            "mean_ethical_score": float(np.mean(scores)),
            "std_ethical_score": float(np.std(scores)),
            "min_ethical_score": float(np.min(scores)),
            "total_concerns_raised": concerns_count,
            "concerns_per_evaluation": concerns_count / len(self.trail),
            "dominant_framework_distribution": frameworks_dominant,
            "mean_confidence": float(np.mean([e["confidence"] for e in self.trail])),
        }

    def export(self, path: str):
        """Export audit trail to JSON."""
        import json
        with open(path, "w") as f:
            json.dump({
                "trail": self.trail,
                "statistics": self.get_statistics(),
            }, f, indent=2)
