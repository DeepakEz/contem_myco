"""
Contemplative MARL Research
============================
ICLR-grade research infrastructure for studying ethical constraints,
stigmergic communication, and mindfulness in multi-agent RL.

Modules:
- agents: ContemplativeAgent with ethics, diffusion, mindfulness
- environments: MPE wrapper with mixed-motive rewards
- training: MAPPO trainer
- analysis: Statistical analysis and visualization

Usage:
    python -m research.run_experiment --ablation --seeds 30
"""

from .config import (
    ExperimentConfig,
    get_full_config,
    get_baseline_config,
    get_ablation_configs,
)

__version__ = "0.1.0"
__all__ = [
    "ExperimentConfig",
    "get_full_config",
    "get_baseline_config",
    "get_ablation_configs",
]
