"""
MycoNet 3.0 Configuration Module
================================

Central configuration for all hyperparameters following Field Architecture theory.
Implements the complete specification from the MycoNet 3.0 blueprint.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
import json
import numpy as np


class WisdomSignalType(Enum):
    """Types of wisdom signals that propagate through the network."""
    SUFFERING_ALERT = auto()
    WISDOM_BEACON = auto()
    COMPASSION_GRADIENT = auto()
    MEDITATION_SYNC = auto()
    ETHICAL_INSIGHT = auto()
    HELP_REQUEST = auto()
    DANGER_WARNING = auto()
    COOPERATION_CALL = auto()


class InsightType(Enum):
    """Types of discrete knowledge units."""
    ETHICAL = auto()
    PRACTICAL = auto()
    SYSTEMIC = auto()
    COMPASSIONATE = auto()
    STRATEGIC = auto()


@dataclass
class EnvironmentConfig:
    """Configuration for the 2D grid environment."""
    grid_size: Tuple[int, int] = (64, 64)
    num_resources: int = 20
    resource_regeneration_rate: float = 0.01
    signal_diffusion_rate: float = 0.1
    signal_decay_rate: float = 0.05
    obstacle_density: float = 0.05
    stochastic_event_probability: float = 0.01


@dataclass
class AgentConfig:
    """Configuration for individual MycoAgents."""
    # Neural network dimensions
    hidden_dim: int = 128
    memory_size: int = 50

    # Genome and evolution
    genome_dim: int = 256

    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor

    # Cognitive parameters
    mindfulness_threshold: float = 0.6
    wisdom_threshold: float = 0.7
    contemplation_frequency: int = 10

    # Initial state
    initial_energy: float = 1.0
    initial_health: float = 1.0

    # Memory
    short_term_memory_size: int = 10
    working_memory_size: int = 5
    wisdom_memory_capacity: int = 100


@dataclass
class UPRTFieldConfig:
    """Configuration for the Unified Pattern Resonance Field."""
    # Lattice structure
    field_resolution: int = 64

    # Lagrangian parameters
    coupling_constant: float = 0.5  # g in Lagrangian
    field_mass: float = 1.0  # m in Lagrangian

    # Potential function parameters
    potential_scale: float = 0.1
    potential_quartic: float = 0.01

    # Dynamics
    dt: float = 0.1
    max_field_magnitude: float = 10.0

    # Topological detection
    winding_number_threshold: float = 0.5


@dataclass
class OvermindConfig:
    """Configuration for the Contemplative Overmind."""
    enabled: bool = True

    # Intervention thresholds
    intervention_threshold: float = 0.7
    meditation_threshold: float = 0.6
    rx_threshold: float = 0.7  # Recoverability threshold

    # Metric weights for interventions
    coherence_weight: float = 1.0
    entropy_weight: float = 0.5
    ethical_penalty: float = 10.0

    # Surrogate model
    use_surrogate: bool = True
    surrogate_hidden_dim: int = 128
    surrogate_update_freq: int = 100

    # Symbolic reasoning
    use_symbolic_bridge: bool = True
    symbolic_reflection_frequency: int = 100


@dataclass
class QREAConfig:
    """Configuration for QREA v3.0 components."""
    # Hypernetwork
    use_hypernetwork: bool = True
    hypernetwork_hidden_dim: int = 512

    # Evolution parameters
    population_size: int = 50
    evolution_generations: int = 100
    mutation_rate: float = 0.1
    mutation_strength: float = 0.05
    crossover_rate: float = 0.7
    elite_fraction: float = 0.1
    tournament_size: int = 5

    # Differentiable surrogates
    use_differentiable_surrogate: bool = True
    surrogate_architecture: str = "fno"  # "fno", "gnn", or "unet"


@dataclass
class DharmaConfig:
    """Configuration for the Dharma Compiler (ethical constraints)."""
    # Core ethical thresholds
    rx_moral_threshold: float = 0.7  # Minimum recoverability index
    max_entropy_export: float = 0.3  # Maximum entropy exported to environment

    # Fairness parameters
    gini_threshold: float = 0.4  # Maximum Gini coefficient for resource distribution

    # Constraint weights
    suffering_prevention_weight: float = 0.9
    coherence_promotion_weight: float = 0.7
    fairness_weight: float = 0.8


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""
    # Inner loop (RL)
    rl_episodes: int = 500
    rl_max_steps: int = 1000
    rl_algorithm: str = "ppo"  # "ppo", "sac", or "dqn"

    # Middle loop (surrogate calibration)
    surrogate_calibration_episodes: int = 50
    surrogate_batch_size: int = 32

    # Outer loop (evolution)
    evolution_generations: int = 50
    evolution_episodes_per_genome: int = 3

    # Multi-objective fitness weights
    task_reward_weight: float = 1.0
    coherence_bonus: float = 0.5
    information_bonus: float = 0.3
    ethical_penalty: float = 1.0
    rx_penalty: float = 0.5

    # Checkpointing
    checkpoint_frequency: int = 10
    log_frequency: int = 5


@dataclass
class MetricsConfig:
    """Configuration for Field Architecture metrics computation."""
    # Entropy computation
    entropy_bins: int = 50
    entropy_multiscale_levels: int = 3

    # Coherence (Kuramoto)
    phase_extraction_method: str = "hilbert"  # "hilbert" or "gradient"

    # Integrated information approximation
    phi_partition_samples: int = 100

    # Recoverability
    rx_baseline_window: int = 100
    rx_perturbation_threshold: float = 0.3

    # Resonance half-life
    tau_fitting_window: int = 50
    tau_min_samples: int = 20


@dataclass
class MycoNetConfig:
    """Central configuration for all MycoNet 3.0 hyperparameters."""

    # Sub-configurations
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    uprt_field: UPRTFieldConfig = field(default_factory=UPRTFieldConfig)
    overmind: OvermindConfig = field(default_factory=OvermindConfig)
    qrea: QREAConfig = field(default_factory=QREAConfig)
    dharma: DharmaConfig = field(default_factory=DharmaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # Global settings
    num_agents: int = 20
    random_seed: int = 42
    device: str = "auto"  # "cpu", "cuda", or "auto"

    # Logging and visualization
    log_level: str = "INFO"
    tensorboard_enabled: bool = True
    visualization_enabled: bool = True
    visualization_frequency: int = 10

    def validate(self) -> bool:
        """Validate all configuration parameters."""
        errors = []

        # Environment validation
        if self.environment.grid_size[0] <= 0 or self.environment.grid_size[1] <= 0:
            errors.append("Grid size must be positive")

        if not 0 <= self.environment.signal_diffusion_rate <= 1:
            errors.append("Signal diffusion rate must be in [0, 1]")

        # Agent validation
        if self.agent.learning_rate <= 0:
            errors.append("Learning rate must be positive")

        if not 0 <= self.agent.gamma <= 1:
            errors.append("Gamma (discount factor) must be in [0, 1]")

        # UPRT Field validation
        if self.uprt_field.field_resolution <= 0:
            errors.append("Field resolution must be positive")

        # Dharma validation
        if not 0 <= self.dharma.rx_moral_threshold <= 1:
            errors.append("RX moral threshold must be in [0, 1]")

        if errors:
            for error in errors:
                print(f"Configuration error: {error}")
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Enum):
                return obj.name
            elif isinstance(obj, (list, tuple)):
                return [dataclass_to_dict(item) for item in obj]
            else:
                return obj
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MycoNetConfig':
        """Create configuration from dictionary."""
        config = cls()

        # Update sub-configs
        if 'environment' in config_dict:
            for k, v in config_dict['environment'].items():
                if hasattr(config.environment, k):
                    setattr(config.environment, k, v)

        if 'agent' in config_dict:
            for k, v in config_dict['agent'].items():
                if hasattr(config.agent, k):
                    setattr(config.agent, k, v)

        if 'uprt_field' in config_dict:
            for k, v in config_dict['uprt_field'].items():
                if hasattr(config.uprt_field, k):
                    setattr(config.uprt_field, k, v)

        if 'overmind' in config_dict:
            for k, v in config_dict['overmind'].items():
                if hasattr(config.overmind, k):
                    setattr(config.overmind, k, v)

        if 'qrea' in config_dict:
            for k, v in config_dict['qrea'].items():
                if hasattr(config.qrea, k):
                    setattr(config.qrea, k, v)

        if 'dharma' in config_dict:
            for k, v in config_dict['dharma'].items():
                if hasattr(config.dharma, k):
                    setattr(config.dharma, k, v)

        if 'training' in config_dict:
            for k, v in config_dict['training'].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)

        if 'metrics' in config_dict:
            for k, v in config_dict['metrics'].items():
                if hasattr(config.metrics, k):
                    setattr(config.metrics, k, v)

        # Update global settings
        for key in ['num_agents', 'random_seed', 'device', 'log_level',
                    'tensorboard_enabled', 'visualization_enabled',
                    'visualization_frequency']:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'MycoNetConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Preset configurations
def create_minimal_config() -> MycoNetConfig:
    """Create minimal configuration for quick testing."""
    config = MycoNetConfig()
    config.num_agents = 5
    config.environment.grid_size = (32, 32)
    config.environment.num_resources = 10
    config.training.rl_episodes = 50
    config.training.rl_max_steps = 100
    config.qrea.evolution_generations = 10
    config.overmind.use_surrogate = False
    return config


def create_basic_config() -> MycoNetConfig:
    """Create basic configuration for standard experiments."""
    config = MycoNetConfig()
    config.num_agents = 20
    config.environment.grid_size = (64, 64)
    config.training.rl_episodes = 200
    return config


def create_advanced_config() -> MycoNetConfig:
    """Create advanced configuration for comprehensive experiments."""
    config = MycoNetConfig()
    config.num_agents = 50
    config.environment.grid_size = (128, 128)
    config.environment.num_resources = 50
    config.training.rl_episodes = 500
    config.qrea.evolution_generations = 100
    return config


def create_scalability_test_config(num_agents: int) -> MycoNetConfig:
    """Create configuration for scalability testing."""
    config = MycoNetConfig()
    config.num_agents = num_agents

    # Scale grid size with agent count
    grid_size = int(np.sqrt(num_agents) * 10)
    config.environment.grid_size = (grid_size, grid_size)
    config.environment.num_resources = num_agents

    return config
