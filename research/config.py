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
        import copy

        def dataclass_to_dict(obj):
            """Recursively convert dataclass to dict."""
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return copy.deepcopy(obj)

        return {
            "name": self.name,
            "seed": self.seed,
            "num_seeds": self.num_seeds,
            "env": dataclass_to_dict(self.env),
            "diffusion": dataclass_to_dict(self.diffusion),
            "ethics": dataclass_to_dict(self.ethics),
            "mindfulness": dataclass_to_dict(self.mindfulness),
            "training": dataclass_to_dict(self.training),
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


def get_factorial_ablation_configs() -> Dict[str, ExperimentConfig]:
    """
    Generate full 2^3 factorial ablation configurations.

    Tests all 8 combinations of ethics x diffusion x mindfulness
    to enable proper factorial ANOVA analysis with interaction effects.

    Returns:
        Dict of 8 configs mapping condition name to ExperimentConfig
    """
    configs = {}
    modules = ["ethics", "diffusion", "mindfulness"]

    for ethics_on in [True, False]:
        for diffusion_on in [True, False]:
            for mindfulness_on in [True, False]:
                # Generate descriptive name
                active = []
                if ethics_on:
                    active.append("E")
                if diffusion_on:
                    active.append("D")
                if mindfulness_on:
                    active.append("M")

                if active:
                    name = "+".join(active)
                else:
                    name = "no_modules"

                # Map to human-readable keys
                if ethics_on and diffusion_on and mindfulness_on:
                    key = "full"
                elif not any([ethics_on, diffusion_on, mindfulness_on]):
                    key = "no_modules"
                elif ethics_on and not diffusion_on and not mindfulness_on:
                    key = "ethics_only"
                elif not ethics_on and diffusion_on and not mindfulness_on:
                    key = "diffusion_only"
                elif not ethics_on and not diffusion_on and mindfulness_on:
                    key = "mindfulness_only"
                elif ethics_on and diffusion_on and not mindfulness_on:
                    key = "ethics_diffusion"
                elif ethics_on and not diffusion_on and mindfulness_on:
                    key = "ethics_mindfulness"
                elif not ethics_on and diffusion_on and mindfulness_on:
                    key = "diffusion_mindfulness"
                else:
                    key = name

                config = ExperimentConfig(name=f"factorial_{key}")
                config.ethics.enabled = ethics_on
                config.diffusion.enabled = diffusion_on
                config.mindfulness.enabled = mindfulness_on
                configs[key] = config

    return configs


def get_adversarial_configs(
    epsilons: List[float] = None,
) -> Dict[str, ExperimentConfig]:
    """
    Generate configs for adversarial robustness testing.

    Creates configs at multiple perturbation levels for both
    the full contemplative agent and the no-module baseline.
    """
    epsilons = epsilons or [0.01, 0.05, 0.1, 0.2, 0.3]
    configs = {}

    for eps in epsilons:
        # Full contemplative under attack
        config_full = get_full_config()
        config_full.name = f"adversarial_full_eps{eps}"
        config_full.env.obs_noise_std = eps
        configs[f"full_eps{eps}"] = config_full

        # Baseline under attack
        config_base = get_baseline_config()
        config_base.name = f"adversarial_baseline_eps{eps}"
        config_base.env.obs_noise_std = eps
        configs[f"baseline_eps{eps}"] = config_base

    return configs


def get_diffusion_only_config() -> ExperimentConfig:
    """Agent with ONLY stigmergic diffusion — no ethics, no mindfulness."""
    config = ExperimentConfig(name="diffusion_only")
    config.diffusion.enabled = True
    config.ethics.enabled = False
    config.mindfulness.enabled = False
    return config


def get_diffusion_paper_configs() -> Dict[str, ExperimentConfig]:
    """
    Configs for the stigmergic diffusion paper.

    Total: 2 main + 9 ablation variants = 11 configs (non-redundant).
    Default diffusion uses: 4 channels, 0.05 decay, 32 grid, 1.0 deposit, radius 3.
    Each ablation varies ONE parameter away from default.
    """
    configs = {}

    # --- Main comparison ---

    # Baseline: no communication
    configs["no_comm"] = get_baseline_config()
    configs["no_comm"].name = "no_communication"

    # Our method: stigmergic diffusion (default settings)
    configs["diffusion"] = get_diffusion_only_config()

    # --- Diffusion ablations (vary one param at a time) ---

    # Channel count: default=4, test 1, 2, 8
    for n_ch in [1, 2, 8]:
        c = get_diffusion_only_config()
        c.name = f"diffusion_{n_ch}ch"
        c.diffusion.num_channels = n_ch
        configs[f"diffusion_{n_ch}ch"] = c

    # Decay rate: default=0.05, test 0.01 (slow) and 0.2 (fast)
    for decay, label in [(0.01, "slow_decay"), (0.2, "fast_decay")]:
        c = get_diffusion_only_config()
        c.name = f"diffusion_{label}"
        c.diffusion.decay_rate = decay
        configs[f"diffusion_{label}"] = c

    # Grid resolution: default=32, test 16 and 64
    for grid, label in [(16, "grid16"), (64, "grid64")]:
        c = get_diffusion_only_config()
        c.name = f"diffusion_{label}"
        c.diffusion.grid_size = grid
        configs[f"diffusion_{label}"] = c

    # Deposit strength: default=1.0, test 0.5 and 2.0
    for strength, label in [(0.5, "weak_deposit"), (2.0, "strong_deposit")]:
        c = get_diffusion_only_config()
        c.name = f"diffusion_{label}"
        c.diffusion.deposit_strength = strength
        configs[f"diffusion_{label}"] = c

    return configs


def get_scalability_configs(
    agent_counts: List[int] = None,
) -> Dict[str, ExperimentConfig]:
    """
    Scalability experiment: vary agent count for each communication method.

    Key hypothesis: Stigmergic diffusion scales O(1) per agent while
    explicit messaging (CommNet, TarMAC) scales O(n).

    Returns configs for: no_comm, diffusion at each agent count.
    Baselines (CommNet, TarMAC) are handled separately in run_experiment.
    """
    agent_counts = agent_counts or [4, 8, 16, 32]
    configs = {}

    for n in agent_counts:
        # No communication baseline
        c_base = get_baseline_config()
        c_base.name = f"no_comm_{n}agents"
        c_base.env.num_agents = n
        c_base.env.num_landmarks = n
        configs[f"no_comm_{n}"] = c_base

        # Stigmergic diffusion
        c_diff = get_diffusion_only_config()
        c_diff.name = f"diffusion_{n}agents"
        c_diff.env.num_agents = n
        c_diff.env.num_landmarks = n
        configs[f"diffusion_{n}"] = c_diff

    return configs


def get_transfer_configs(
    source_envs: List[str] = None,
    target_envs: List[str] = None,
) -> Dict[str, Dict[str, ExperimentConfig]]:
    """
    Generate configs for transfer learning experiments.

    Returns nested dict: {source_env: {target_env: config}}
    """
    source_envs = source_envs or ["mpe_simple_spread", "mpe_simple_tag"]
    target_envs = target_envs or ["mpe_simple_adversary", "mpe_simple_spread"]

    configs = {}
    for source in source_envs:
        configs[source] = {}
        for target in target_envs:
            if source == target:
                continue
            config = get_full_config()
            config.name = f"transfer_{source}_to_{target}"
            config.env.name = target
            configs[source][target] = config

    return configs
