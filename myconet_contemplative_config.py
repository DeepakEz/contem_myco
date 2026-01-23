"""
Unified Contemplative Configuration Module
Single source of truth for all contemplative system parameters
"""

from dataclasses import dataclass, field
from typing import List, Optional
from copy import deepcopy
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContemplativeConfig:
    """
    Unified configuration for contemplative processing across all modules.
    Combines parameters from main.py and entities.py into single source of truth.
    """

    # ===== CORE CONTEMPLATIVE PARAMETERS =====
    enable_contemplative_processing: bool = True
    mindfulness_update_frequency: int = 20
    wisdom_signal_strength: float = 0.3
    collective_meditation_threshold: float = 0.8
    ethical_reasoning_depth: int = 1
    contemplative_memory_capacity: int = 100
    wisdom_sharing_radius: int = 1
    compassion_sensitivity: float = 0.4

    # ===== EXTENDED PARAMETERS (from entities version) =====
    wisdom_sharing_threshold: float = 0.3
    ethical_learning_rate: float = 0.01
    mindfulness_decay_rate: float = 0.02
    wisdom_accumulation_rate: float = 0.05
    cooperation_tendency: float = 0.5
    exploration_bias: float = 0.3
    risk_tolerance: float = 0.4
    mutation_rate: float = 0.1
    mutation_strength: float = 0.05

    # ===== BRAIN & NEURAL PARAMETERS =====
    brain_hidden_size: int = 128
    brain_learning_rate: float = 0.001
    brain_dropout: float = 0.1

    # ===== WISDOM SIGNAL PARAMETERS =====
    signal_diffusion_rate: float = 0.1
    signal_decay_rate: float = 0.05
    signal_propagation_distance: int = 5

    def validate(self) -> bool:
        """
        Validate configuration parameters are within acceptable ranges.
        Returns True if valid, raises ValueError if invalid.
        """
        # Probability/rate checks (must be 0-1)
        if not 0 <= self.wisdom_signal_strength <= 1:
            raise ValueError(f"wisdom_signal_strength must be in [0,1], got {self.wisdom_signal_strength}")

        if not 0 <= self.collective_meditation_threshold <= 1:
            raise ValueError(f"collective_meditation_threshold must be in [0,1], got {self.collective_meditation_threshold}")

        if not 0 <= self.wisdom_sharing_threshold <= 1:
            raise ValueError(f"wisdom_sharing_threshold must be in [0,1], got {self.wisdom_sharing_threshold}")

        if not 0 <= self.compassion_sensitivity <= 1:
            raise ValueError(f"compassion_sensitivity must be in [0,1], got {self.compassion_sensitivity}")

        # Learning rate checks
        if self.ethical_learning_rate <= 0:
            raise ValueError(f"ethical_learning_rate must be positive, got {self.ethical_learning_rate}")

        if self.brain_learning_rate <= 0:
            raise ValueError(f"brain_learning_rate must be positive, got {self.brain_learning_rate}")

        # Integer parameter checks
        if self.mindfulness_update_frequency <= 0:
            raise ValueError(f"mindfulness_update_frequency must be positive, got {self.mindfulness_update_frequency}")

        if self.ethical_reasoning_depth < 0:
            raise ValueError(f"ethical_reasoning_depth must be non-negative, got {self.ethical_reasoning_depth}")

        if self.contemplative_memory_capacity <= 0:
            raise ValueError(f"contemplative_memory_capacity must be positive, got {self.contemplative_memory_capacity}")

        if self.wisdom_sharing_radius < 0:
            raise ValueError(f"wisdom_sharing_radius must be non-negative, got {self.wisdom_sharing_radius}")

        return True

    def to_dict(self):
        """Convert configuration to dictionary for serialization"""
        return {
            'enable_contemplative_processing': self.enable_contemplative_processing,
            'mindfulness_update_frequency': self.mindfulness_update_frequency,
            'wisdom_signal_strength': self.wisdom_signal_strength,
            'collective_meditation_threshold': self.collective_meditation_threshold,
            'ethical_reasoning_depth': self.ethical_reasoning_depth,
            'contemplative_memory_capacity': self.contemplative_memory_capacity,
            'wisdom_sharing_radius': self.wisdom_sharing_radius,
            'compassion_sensitivity': self.compassion_sensitivity,
            'wisdom_sharing_threshold': self.wisdom_sharing_threshold,
            'ethical_learning_rate': self.ethical_learning_rate,
            'mindfulness_decay_rate': self.mindfulness_decay_rate,
            'wisdom_accumulation_rate': self.wisdom_accumulation_rate,
            'cooperation_tendency': self.cooperation_tendency,
            'exploration_bias': self.exploration_bias,
            'risk_tolerance': self.risk_tolerance,
            'mutation_rate': self.mutation_rate,
            'mutation_strength': self.mutation_strength,
            'brain_hidden_size': self.brain_hidden_size,
            'brain_learning_rate': self.brain_learning_rate,
            'brain_dropout': self.brain_dropout,
            'signal_diffusion_rate': self.signal_diffusion_rate,
            'signal_decay_rate': self.signal_decay_rate,
            'signal_propagation_distance': self.signal_propagation_distance,
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Create configuration from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def mutate(self) -> 'ContemplativeConfig':
        """
        Create a mutated copy of this configuration for evolutionary algorithms.
        Uses mutation_rate and mutation_strength to determine mutation behavior.
        """
        new_config = deepcopy(self)

        # List of mutable parameters with their types and bounds
        mutable_params = {
            'compassion_sensitivity': ('float', 0.0, 1.0),
            'wisdom_sharing_threshold': ('float', 0.0, 1.0),
            'ethical_learning_rate': ('float', 0.001, 0.1),
            'mindfulness_decay_rate': ('float', 0.001, 0.1),
            'wisdom_accumulation_rate': ('float', 0.001, 0.1),
            'cooperation_tendency': ('float', 0.0, 1.0),
            'exploration_bias': ('float', 0.0, 1.0),
            'risk_tolerance': ('float', 0.0, 1.0),
            'mindfulness_update_frequency': ('int', 1, 100),
            'wisdom_sharing_radius': ('int', 1, 10),
        }

        for param, (param_type, min_val, max_val) in mutable_params.items():
            if random.random() < self.mutation_rate:
                current_value = getattr(new_config, param)

                if param_type == 'float':
                    mutation = np.random.normal(0, self.mutation_strength)
                    new_value = np.clip(current_value + mutation, min_val, max_val)
                    setattr(new_config, param, float(new_value))

                elif param_type == 'int':
                    mutation = int(np.random.normal(0, 2))
                    new_value = np.clip(current_value + mutation, min_val, max_val)
                    setattr(new_config, param, int(new_value))

        # Re-validate after mutation
        new_config.validate()
        return new_config


@dataclass
class WisdomSignalConfig:
    """Configuration for wisdom signal propagation and behavior"""

    signal_types: List[str] = field(default_factory=lambda: [
        "ETHICAL_INSIGHT",
        "SUFFERING_DETECTION",
        "COMPASSION_GRADIENT",
        "CONTEMPLATIVE_DEPTH",
        "MEDITATION_SYNC"
    ])

    diffusion_rate: float = 0.1
    decay_rate: float = 0.05
    propagation_distance: int = 5
    base_diffusion_rate: float = 0.3
    base_decay_rate: float = 0.02
    intensity_threshold: float = 0.1
    max_intensity: float = 10.0

    def validate(self) -> bool:
        """Validate wisdom signal configuration parameters"""
        if self.diffusion_rate < 0:
            raise ValueError(f"diffusion_rate must be non-negative, got {self.diffusion_rate}")

        if self.decay_rate < 0:
            raise ValueError(f"decay_rate must be non-negative, got {self.decay_rate}")

        if self.propagation_distance <= 0:
            raise ValueError(f"propagation_distance must be positive, got {self.propagation_distance}")

        if self.max_intensity <= 0:
            raise ValueError(f"max_intensity must be positive, got {self.max_intensity}")

        return True


# Convenience function for creating default configs
def create_default_contemplative_config(**overrides) -> ContemplativeConfig:
    """
    Create a ContemplativeConfig with default values, optionally overriding specific parameters.

    Usage:
        config = create_default_contemplative_config(compassion_sensitivity=0.8)
    """
    config = ContemplativeConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    config.validate()
    return config


def create_default_wisdom_signal_config(**overrides) -> WisdomSignalConfig:
    """Create a WisdomSignalConfig with default values and optional overrides"""
    config = WisdomSignalConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown wisdom signal configuration parameter: {key}")
    config.validate()
    return config


@dataclass
class ResearchParameters:
    """
    Centralized research parameters extracted from hardcoded values.

    All magic numbers that affect research outcomes are documented here with:
    - Scientific justification where applicable
    - Valid ranges for sensitivity analysis
    - References to relevant literature

    IMPORTANT: Changes to these parameters should be documented and justified
    for academic reproducibility.
    """

    # ===== AGENT BEHAVIOR THRESHOLDS =====
    # Energy/health initialization range [min, max]
    # Based on typical survival simulation initial conditions
    agent_energy_init_min: float = 0.3
    agent_energy_init_max: float = 0.9
    agent_health_init_min: float = 0.4
    agent_health_init_max: float = 1.0

    # Contemplation probability when energy > threshold
    contemplation_energy_threshold: float = 0.8
    contemplation_health_threshold: float = 0.7
    contemplation_probability: float = 0.05  # 5% base chance

    # Mindfulness-related thresholds
    high_mindfulness_threshold: float = 0.6  # Above this = "mindful" state
    deep_contemplation_threshold: float = 0.8  # Above this = deep state
    meditation_sync_threshold: float = 0.15  # Rate for meditation logging

    # ===== RESOURCE CONSUMPTION RATES =====
    # These affect survival dynamics significantly
    food_consumption_rate: float = 0.3  # Max food consumed per action
    water_consumption_rate: float = 0.3  # Max water consumed per action
    food_success_threshold: float = 0.7  # random() > this = success
    water_success_threshold: float = 0.8  # random() > this = success
    hazard_encounter_threshold: float = 0.9  # random() > this = hazard

    # ===== INTERVENTION & DECISION THRESHOLDS =====
    # Overmind intervention triggers
    crisis_intervention_threshold: float = 0.7  # Crisis level to trigger
    mesh_consensus_crisis_threshold: float = 0.6  # Request consensus above
    mesh_consensus_cooperation_threshold: float = 0.8  # Or cooperation below

    # Decision confidence thresholds
    low_confidence_threshold: float = 0.4
    high_confidence_threshold: float = 0.8
    default_confidence: float = 0.5

    # ===== LEARNING & ADAPTATION RATES =====
    # Neural network and learning parameters
    adaptation_learning_rate: float = 0.01
    threshold_adaptation_rate: float = 0.01
    wisdom_retention_rate: float = 0.7  # How much wisdom is retained

    # Exploration vs exploitation
    epsilon_initial: float = 0.1  # Initial exploration rate
    epsilon_min: float = 0.01  # Minimum exploration rate
    epsilon_decay: float = 0.995  # Decay rate per step

    # ===== COOPERATION & SOCIAL DYNAMICS =====
    cooperation_boost_rate: float = 0.1  # Cooperation increase from rituals
    conflict_reduction_rate: float = 0.15  # Conflict decrease from rituals
    wisdom_sharing_effect: float = 0.3  # Effect magnitude of sharing

    # Group dynamics
    min_ritual_participants: int = 5  # Minimum for group bonus
    group_bonus_rate: float = 0.02  # Bonus per additional participant
    max_group_bonus: float = 0.3  # Cap on group bonus

    # ===== EVALUATION METRICS =====
    # Weights for multi-objective evaluation
    survival_weight: float = 0.25
    wisdom_weight: float = 0.25
    ethical_weight: float = 0.25
    efficiency_weight: float = 0.25

    # Significance thresholds
    significance_archival_threshold: float = 0.7  # Min significance to archive
    high_significance_threshold: float = 0.8  # Log as "high significance"
    decay_warning_threshold: float = 0.6  # Warn about insight decay

    # ===== SIGNAL PROPAGATION =====
    signal_intensity_normalization: float = 0.2  # Grid normalization factor
    signal_proximity_threshold: float = 0.2  # Fraction of grid for "nearby"
    signal_strength_minimum: float = 0.1  # Minimum detectable signal

    def validate(self) -> bool:
        """Validate research parameters are within acceptable ranges."""
        # Validate probability ranges [0, 1]
        prob_params = [
            'agent_energy_init_min', 'agent_energy_init_max',
            'agent_health_init_min', 'agent_health_init_max',
            'contemplation_probability', 'high_mindfulness_threshold',
            'food_success_threshold', 'water_success_threshold',
            'crisis_intervention_threshold', 'default_confidence',
            'epsilon_initial', 'epsilon_min', 'wisdom_retention_rate',
            'significance_archival_threshold'
        ]
        for param in prob_params:
            value = getattr(self, param)
            if not 0 <= value <= 1:
                raise ValueError(f"{param} must be in [0, 1], got {value}")

        # Validate positive rates
        rate_params = [
            'adaptation_learning_rate', 'threshold_adaptation_rate',
            'cooperation_boost_rate', 'conflict_reduction_rate',
            'group_bonus_rate', 'epsilon_decay'
        ]
        for param in rate_params:
            value = getattr(self, param)
            if value <= 0:
                raise ValueError(f"{param} must be positive, got {value}")

        # Validate weights sum to 1 (approximately)
        weight_sum = (self.survival_weight + self.wisdom_weight +
                     self.ethical_weight + self.efficiency_weight)
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Evaluation weights sum to {weight_sum}, not 1.0")

        return True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ResearchParameters':
        """Create from dictionary."""
        return cls(**{k: v for k, v in config_dict.items()
                     if k in cls.__dataclass_fields__})

    def get_sensitivity_ranges(self) -> dict:
        """
        Get suggested ranges for sensitivity analysis.

        Returns dict of parameter -> (min, max, recommended_steps)
        for systematic parameter sweeps.
        """
        return {
            'crisis_intervention_threshold': (0.5, 0.9, 5),
            'high_mindfulness_threshold': (0.4, 0.8, 5),
            'cooperation_boost_rate': (0.05, 0.2, 4),
            'wisdom_retention_rate': (0.5, 0.9, 5),
            'significance_archival_threshold': (0.5, 0.9, 5),
            'survival_weight': (0.1, 0.4, 4),
            'wisdom_weight': (0.1, 0.4, 4),
        }


def create_default_research_parameters(**overrides) -> ResearchParameters:
    """Create ResearchParameters with default values and optional overrides."""
    config = ResearchParameters()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown research parameter: {key}")
    config.validate()
    return config


# Global singleton for research parameters (can be overridden)
_research_params: Optional[ResearchParameters] = None


def get_research_parameters() -> ResearchParameters:
    """Get the global research parameters instance."""
    global _research_params
    if _research_params is None:
        _research_params = ResearchParameters()
    return _research_params


def set_research_parameters(params: ResearchParameters):
    """Set the global research parameters instance."""
    global _research_params
    params.validate()
    _research_params = params
