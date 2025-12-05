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
