"""
Network Dharma Compiler - Section 3.1
=====================================

Compiles ethical directives and contemplative constraints from colony state.
Part of the Overmind Dharma + RL Integration (Section 3).

The Dharma Compiler takes:
- Base colony analysis (survival, resources, threats)
- Contemplative assessment (wisdom, ethics, suffering)

And produces:
- Ethical directives (constraints on actions)
- Contemplative priorities (what matters now)
- Intervention recommendations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DharmaDirectiveType(Enum):
    """Types of dharma directives the overmind can issue"""
    REDUCE_SUFFERING = "reduce_suffering"
    PROMOTE_COOPERATION = "promote_cooperation"
    ENHANCE_WISDOM = "enhance_wisdom"
    MAINTAIN_BALANCE = "maintain_balance"
    PROTECT_VULNERABLE = "protect_vulnerable"
    ENCOURAGE_MINDFULNESS = "encourage_mindfulness"
    PREVENT_HARM = "prevent_harm"
    FOSTER_COMPASSION = "foster_compassion"


@dataclass
class DharmaDirective:
    """A single ethical directive from the dharma compiler"""
    directive_type: DharmaDirectiveType
    priority: float  # 0.0 to 1.0
    target_metric: str  # What metric this affects (e.g., 'suffering_level')
    target_value: float  # Desired value for the metric
    rationale: str  # Human-readable explanation
    constraints: Dict[str, Any] = field(default_factory=dict)  # Additional constraints


@dataclass
class ContemplativeObservation:
    """
    Enhanced observation structure for contemplative overmind.
    Implements Section 3.1: Observation Struct
    """
    # Colony survival metrics
    population_size: int
    average_energy: float
    average_health: float
    average_age: float
    survival_rate: float

    # Contemplative state metrics
    average_mindfulness: float
    collective_wisdom: float
    ethical_alignment: float
    suffering_level: float  # 0.0 = no suffering, 1.0 = extreme

    # Wisdom grid metrics
    network_coherence: float
    signal_diversity: float
    total_signals: float
    meditation_sync_strength: float

    # Social metrics
    cooperation_rate: float
    conflict_rate: float
    wisdom_sharing_rate: float

    # Temporal context
    step: int
    time_of_day: float  # 0.0 to 1.0 (for circadian rhythms)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for RL training"""
        return np.array([
            float(self.population_size) / 100.0,  # Normalize assuming max 100
            self.average_energy,
            self.average_health,
            self.average_age / 1000.0,  # Normalize assuming max age 1000
            self.survival_rate,
            self.average_mindfulness,
            self.collective_wisdom,
            self.ethical_alignment,
            self.suffering_level,
            self.network_coherence,
            self.signal_diversity,
            self.total_signals / 100.0,  # Normalize
            self.meditation_sync_strength,
            self.cooperation_rate,
            self.conflict_rate,
            self.wisdom_sharing_rate,
            self.time_of_day
        ], dtype=np.float32)

    @property
    def observation_dim(self) -> int:
        """Dimension of observation vector"""
        return 17  # Number of features in to_array()


class NetworkDharmaCompiler:
    """
    Compiles dharma directives from colony state and contemplative assessment.
    Implements Section 3.1: Network Dharma Compiler
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Thresholds for triggering different directives
        self.suffering_threshold = self.config.get('suffering_threshold', 0.6)
        self.cooperation_threshold = self.config.get('cooperation_threshold', 0.4)
        self.wisdom_threshold = self.config.get('wisdom_threshold', 3.0)
        self.balance_threshold = self.config.get('balance_threshold', 0.3)

        self.directive_history: List[DharmaDirective] = []

    def compile(self, observation: ContemplativeObservation) -> List[DharmaDirective]:
        """
        Main compilation method: analyzes observation and produces directives.

        Args:
            observation: Current contemplative observation of colony state

        Returns:
            List of dharma directives sorted by priority
        """
        directives = []

        # 1. REDUCE_SUFFERING: High priority if suffering detected
        if observation.suffering_level > self.suffering_threshold:
            directives.append(DharmaDirective(
                directive_type=DharmaDirectiveType.REDUCE_SUFFERING,
                priority=min(1.0, observation.suffering_level),
                target_metric='suffering_level',
                target_value=0.3,  # Aim to reduce to 30%
                rationale=f"Suffering level {observation.suffering_level:.2f} exceeds threshold {self.suffering_threshold}",
                constraints={'max_energy_cost': 0.8}  # Don't exhaust resources helping
            ))

        # 2. PROMOTE_COOPERATION: If cooperation is low
        if observation.cooperation_rate < self.cooperation_threshold:
            directives.append(DharmaDirective(
                directive_type=DharmaDirectiveType.PROMOTE_COOPERATION,
                priority=0.7,
                target_metric='cooperation_rate',
                target_value=0.6,
                rationale=f"Cooperation rate {observation.cooperation_rate:.2f} below threshold",
                constraints={'focus_on_nearby_agents': True}
            ))

        # 3. ENHANCE_WISDOM: If wisdom is low relative to population
        expected_wisdom = observation.population_size * 0.5  # Expect 0.5 wisdom per agent
        if observation.collective_wisdom < expected_wisdom * 0.5:
            directives.append(DharmaDirective(
                directive_type=DharmaDirectiveType.ENHANCE_WISDOM,
                priority=0.6,
                target_metric='collective_wisdom',
                target_value=expected_wisdom,
                rationale=f"Collective wisdom {observation.collective_wisdom:.1f} below expected {expected_wisdom:.1f}",
                constraints={'encourage_meditation': True}
            ))

        # 4. MAINTAIN_BALANCE: If any extreme imbalances detected
        imbalance_score = self._calculate_imbalance(observation)
        if imbalance_score > self.balance_threshold:
            directives.append(DharmaDirective(
                directive_type=DharmaDirectiveType.MAINTAIN_BALANCE,
                priority=0.5,
                target_metric='system_balance',
                target_value=0.0,  # Zero imbalance
                rationale=f"System imbalance detected: {imbalance_score:.2f}",
                constraints={'rebalance_resources': True}
            ))

        # 5. PROTECT_VULNERABLE: If survival rate is dropping
        if observation.survival_rate < 0.7 and observation.population_size > 5:
            directives.append(DharmaDirective(
                directive_type=DharmaDirectiveType.PROTECT_VULNERABLE,
                priority=0.8,
                target_metric='survival_rate',
                target_value=0.9,
                rationale=f"Survival rate {observation.survival_rate:.2f} is concerning",
                constraints={'prioritize_low_health': True}
            ))

        # 6. ENCOURAGE_MINDFULNESS: If mindfulness is low
        if observation.average_mindfulness < 0.5:
            directives.append(DharmaDirective(
                directive_type=DharmaDirectiveType.ENCOURAGE_MINDFULNESS,
                priority=0.4,
                target_metric='average_mindfulness',
                target_value=0.7,
                rationale=f"Mindfulness {observation.average_mindfulness:.2f} below healthy level",
                constraints={'trigger_meditation_sync': observation.meditation_sync_strength < 0.3}
            ))

        # Sort by priority (highest first)
        directives.sort(key=lambda d: d.priority, reverse=True)

        # Store in history
        self.directive_history.extend(directives)
        if len(self.directive_history) > 1000:
            self.directive_history = self.directive_history[-1000:]  # Keep last 1000

        return directives

    def _calculate_imbalance(self, observation: ContemplativeObservation) -> float:
        """Calculate system imbalance score"""
        imbalances = []

        # Energy/health imbalance
        energy_health_diff = abs(observation.average_energy - observation.average_health)
        imbalances.append(energy_health_diff)

        # Cooperation/conflict imbalance
        social_imbalance = observation.conflict_rate / max(0.01, observation.cooperation_rate)
        imbalances.append(min(1.0, social_imbalance))

        # Wisdom/mindfulness imbalance (should grow together)
        if observation.collective_wisdom > 0:
            wisdom_mindfulness_ratio = observation.average_mindfulness / (observation.collective_wisdom / max(1, observation.population_size))
            wisdom_imbalance = abs(1.0 - wisdom_mindfulness_ratio)
            imbalances.append(min(1.0, wisdom_imbalance))

        return np.mean(imbalances)

    def get_directive_summary(self) -> Dict[str, Any]:
        """Get summary of recent directives"""
        if not self.directive_history:
            return {'total_directives': 0, 'recent_priorities': {}}

        recent = self.directive_history[-50:]  # Last 50 directives
        priority_by_type = {}
        for directive in recent:
            dtype = directive.directive_type.value
            if dtype not in priority_by_type:
                priority_by_type[dtype] = []
            priority_by_type[dtype].append(directive.priority)

        return {
            'total_directives': len(self.directive_history),
            'recent_priorities': {k: np.mean(v) for k, v in priority_by_type.items()},
            'most_common': max(priority_by_type.items(), key=lambda x: len(x[1]))[0] if priority_by_type else None
        }


# Placeholder for RL Policy (to be implemented in training script)
class ContemplativeOvermindPolicy:
    """
    RL policy for contemplative overmind actions.
    Placeholder for Section 3.1: Contemplative Overmind Policy

    This will be trained using the Gym environment defined in Section 3.2
    """

    def __init__(self, observation_dim: int, action_dim: int):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        logger.info(f"Contemplative Overmind Policy initialized: obs_dim={observation_dim}, action_dim={action_dim}")

    def select_action(self, observation: ContemplativeObservation, directives: List[DharmaDirective]) -> int:
        """
        Select overmind action based on observation and dharma directives.
        Currently uses rule-based logic; will be replaced with trained RL policy.
        """
        if not directives:
            return 0  # NO_ACTION

        # Rule-based selection based on highest priority directive
        top_directive = directives[0]

        # Map directive types to action indices (placeholder mapping)
        directive_to_action = {
            DharmaDirectiveType.REDUCE_SUFFERING: 1,  # TRIGGER_COLLECTIVE_MEDITATION
            DharmaDirectiveType.PROMOTE_COOPERATION: 2,  # PROPAGATE_ETHICAL_INSIGHT
            DharmaDirectiveType.ENHANCE_WISDOM: 3,  # ADJUST_COMPASSION_GRADIENT
            DharmaDirectiveType.MAINTAIN_BALANCE: 4,  # INITIATE_WISDOM_SHARING
            DharmaDirectiveType.PROTECT_VULNERABLE: 1,  # TRIGGER_COLLECTIVE_MEDITATION
            DharmaDirectiveType.ENCOURAGE_MINDFULNESS: 1,  # TRIGGER_COLLECTIVE_MEDITATION
        }

        return directive_to_action.get(top_directive.directive_type, 0)

    def update(self, observation, action, reward, next_observation, done):
        """Update policy (placeholder for RL training)"""
        # Will be implemented with actual RL algorithm (PPO, SAC, etc.)
        pass


if __name__ == "__main__":
    # Example usage
    print("Network Dharma Compiler - Section 3.1 Scaffold")
    print("=" * 60)

    # Create sample observation
    obs = ContemplativeObservation(
        population_size=10,
        average_energy=0.5,
        average_health=0.6,
        average_age=50.0,
        survival_rate=0.8,
        average_mindfulness=0.4,
        collective_wisdom=3.5,
        ethical_alignment=0.7,
        suffering_level=0.7,  # High suffering!
        network_coherence=0.5,
        signal_diversity=0.3,
        total_signals=25.0,
        meditation_sync_strength=0.2,
        cooperation_rate=0.3,  # Low cooperation!
        conflict_rate=0.1,
        wisdom_sharing_rate=0.4,
        step=100,
        time_of_day=0.5
    )

    # Compile directives
    compiler = NetworkDharmaCompiler()
    directives = compiler.compile(obs)

    print(f"\nGenerated {len(directives)} dharma directives:")
    for i, directive in enumerate(directives, 1):
        print(f"\n{i}. {directive.directive_type.value.upper()}")
        print(f"   Priority: {directive.priority:.2f}")
        print(f"   Rationale: {directive.rationale}")
        print(f"   Target: {directive.target_metric} → {directive.target_value}")

    print(f"\nObservation vector shape: {obs.to_array().shape}")
    print(f"Observation dim: {obs.observation_dim}")
