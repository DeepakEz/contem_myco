"""
Research Configuration System
==============================
Centralized configuration for reproducible experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import yaml
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    name: str = "mpe_simple_spread"
    num_agents: int = 4
    num_landmarks: int = 4
    max_steps: int = 100
    local_obs_radius: float = 0.3  # Partial observability
    mixed_motive: bool = True  # Individual + team rewards
    individual_reward_weight: float = 0.3

    # Perturbations for robustness testing
    obs_noise_std: float = 0.0
    action_noise_std: float = 0.0
    reward_noise_std: float = 0.0


@dataclass
class DiffusionConfig:
    """Module B: Stigmergic diffusion field."""
    enabled: bool = True
    grid_size: int = 32
    num_channels: int = 4  # danger, resource, coordination, ethics
    diffusion_rate: float = 0.1
    decay_rate: float = 0.05
    deposit_strength: float = 1.0
    sense_radius: int = 3


@dataclass
class EthicsConfig:
    """Module A: Ethical constraints."""
    enabled: bool = True
    constraint_type: str = "lagrangian"  # lagrangian, reward_shaping, hard

    # Multi-framework weights (sum to 1)
    consequentialist_weight: float = 0.25
    deontological_weight: float = 0.25
    virtue_weight: float = 0.25
    care_weight: float = 0.25

    # Constraint thresholds
    harm_threshold: float = 0.1
    fairness_threshold: float = 0.3  # Max Gini

    # Lagrangian parameters
    lagrange_lr: float = 0.01
    lagrange_init: float = 0.1


@dataclass
class MindfulnessConfig:
    """Module C: Mindfulness for robustness."""
    enabled: bool = True

    # Uncertainty estimation
    ensemble_size: int = 3
    dropout_rate: float = 0.1

    # Gating mechanism
    surprise_threshold: float = 0.5
    conservative_entropy_bonus: float = 0.1

    # Action smoothing
    action_smoothing: float = 0.3


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    algorithm: str = "mappo"
    total_timesteps: int = 5_000_000
    n_envs: int = 8
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4

    # PPO specific
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"

    # Logging
    log_interval: int = 10
    save_interval: int = 100_000
    eval_interval: int = 50_000
    eval_episodes: int = 20


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = "contemplative_marl"
    seed: int = 42
    num_seeds: int = 30  # For statistical significance

    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    ethics: EthicsConfig = field(default_factory=EthicsConfig)
    mindfulness: MindfulnessConfig = field(default_factory=MindfulnessConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Ablation flags
    ablation_mode: Optional[str] = None  # None, "no_ethics", "no_diffusion", "no_mindfulness"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "seed": self.seed,
            "num_seeds": self.num_seeds,
            "env": self.env.__dict__,
            "diffusion": self.diffusion.__dict__,
            "ethics": self.ethics.__dict__,
            "mindfulness": self.mindfulness.__dict__,
            "training": self.training.__dict__,
            "ablation_mode": self.ablation_mode,
        }

    def save(self, path: str):
        """Save configuration to file."""
        path = Path(path)
        with open(path, 'w') as f:
            if path.suffix == '.yaml':
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            else:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from file."""
        path = Path(path)
        with open(path, 'r') as f:
            if path.suffix == '.yaml':
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        config = cls()
        config.name = data.get("name", config.name)
        config.seed = data.get("seed", config.seed)
        config.num_seeds = data.get("num_seeds", config.num_seeds)
        config.ablation_mode = data.get("ablation_mode", None)

        if "env" in data:
            config.env = EnvironmentConfig(**data["env"])
        if "diffusion" in data:
            config.diffusion = DiffusionConfig(**data["diffusion"])
        if "ethics" in data:
            config.ethics = EthicsConfig(**data["ethics"])
        if "mindfulness" in data:
            config.mindfulness = MindfulnessConfig(**data["mindfulness"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])

        return config


# Preset configurations for experiments
def get_baseline_config() -> ExperimentConfig:
    """MAPPO baseline without any modules."""
    config = ExperimentConfig(name="baseline_mappo")
    config.diffusion.enabled = False
    config.ethics.enabled = False
    config.mindfulness.enabled = False
    return config


def get_full_config() -> ExperimentConfig:
    """Full contemplative MARL with all modules."""
    return ExperimentConfig(name="contemplative_full")


def get_ablation_configs() -> Dict[str, ExperimentConfig]:
    """Generate ablation configurations."""
    configs = {}

    # Full model
    configs["full"] = get_full_config()

    # Single module ablations
    for module in ["diffusion", "ethics", "mindfulness"]:
        config = get_full_config()
        config.name = f"ablation_no_{module}"
        config.ablation_mode = f"no_{module}"
        getattr(config, module).enabled = False
        configs[f"no_{module}"] = config

    # Baseline
    configs["baseline"] = get_baseline_config()

    return configs


def get_scaling_configs(agent_counts: List[int] = [4, 8, 16, 32]) -> Dict[str, ExperimentConfig]:
    """Generate scaling experiment configurations."""
    configs = {}
    for n in agent_counts:
        config = get_full_config()
        config.name = f"scaling_{n}_agents"
        config.env.num_agents = n
        config.env.num_landmarks = n
        configs[f"{n}_agents"] = config
    return configs


def get_robustness_configs() -> Dict[str, ExperimentConfig]:
    """Generate robustness test configurations."""
    configs = {}

    # Observation noise
    for noise in [0.1, 0.2, 0.3]:
        config = get_full_config()
        config.name = f"robustness_obs_noise_{noise}"
        config.env.obs_noise_std = noise
        configs[f"obs_noise_{noise}"] = config

    # Action noise
    for noise in [0.1, 0.2]:
        config = get_full_config()
        config.name = f"robustness_act_noise_{noise}"
        config.env.action_noise_std = noise
        configs[f"act_noise_{noise}"] = config

    return configs
