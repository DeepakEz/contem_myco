#!/usr/bin/env python3
"""
Contemplative MARL Experiment Runner
=====================================
Main entry point for running experiments.

Usage:
    python -m research.run_experiment --config full --seeds 30
    python -m research.run_experiment --ablation --seeds 30
    python -m research.run_experiment --scaling --agents 4 8 16 32
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.config import (
    ExperimentConfig,
    get_full_config,
    get_baseline_config,
    get_ablation_configs,
    get_scaling_configs,
    get_robustness_configs,
)
from research.agents import MultiAgentSystem
from research.environments import MixedMotiveMPE, SocialDilemmaEnv
from research.training import MAPPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(
    config: ExperimentConfig,
    seed: int,
    output_dir: Path,
    device: str = "cpu",
) -> dict:
    """Run a single experiment with given config and seed."""
    set_seed(seed)

    logger.info(f"Running experiment: {config.name} with seed {seed}")

    # Create environment
    try:
        env = MixedMotiveMPE(
            env_name=config.env.name.replace("mpe_", ""),
            num_agents=config.env.num_agents,
            num_landmarks=config.env.num_landmarks,
            max_steps=config.env.max_steps,
            individual_reward_weight=config.env.individual_reward_weight,
            obs_noise_std=config.env.obs_noise_std,
            action_noise_std=config.env.action_noise_std,
        )
    except ImportError:
        logger.warning("PettingZoo not available, using SocialDilemmaEnv")
        env = SocialDilemmaEnv(
            num_agents=config.env.num_agents,
            num_rounds=config.env.max_steps,
        )

    # Create agent system
    agent_system = MultiAgentSystem(
        num_agents=config.env.num_agents,
        obs_size=env.obs_size,
        action_size=env.action_size,
        config=config,
        share_params=True,
    )

    # Create trainer
    log_dir = output_dir / f"seed_{seed}"
    trainer = MAPPOTrainer(
        env=env,
        agent_system=agent_system,
        config=config,
        device=device,
        log_dir=str(log_dir),
    )

    # Train
    stats = trainer.train(
        total_timesteps=config.training.total_timesteps,
        log_interval=config.training.log_interval,
        save_interval=config.training.save_interval,
    )

    # Evaluate
    eval_results = trainer.evaluate(n_episodes=config.training.eval_episodes)

    # Close environment
    env.close()

    # Compile results
    results = {
        'config': config.name,
        'seed': seed,
        'final_reward': eval_results['mean_reward'],
        'final_reward_std': eval_results['std_reward'],
        'mean_social_welfare': eval_results.get('mean_social_welfare', 0),
        'mean_gini': eval_results.get('mean_gini', 0),
        'mean_cooperation': eval_results.get('mean_cooperation', 0),
        'training_rewards': stats.episode_rewards[-100:],  # Last 100 episodes
        'policy_losses': stats.policy_losses[-100:],
    }

    # Save results
    with open(log_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    logger.info(f"Completed seed {seed}: reward={results['final_reward']:.2f}")

    return results


def run_experiment_suite(
    configs: dict,
    num_seeds: int,
    output_dir: Path,
    device: str = "cpu",
) -> dict:
    """Run experiments for multiple configs and seeds."""
    all_results = {}

    for config_name, config in configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running configuration: {config_name}")
        logger.info(f"{'='*60}")

        config_results = []
        config_dir = output_dir / config_name

        for seed in range(num_seeds):
            try:
                result = run_single_experiment(
                    config=config,
                    seed=seed,
                    output_dir=config_dir,
                    device=device,
                )
                config_results.append(result)
            except Exception as e:
                logger.error(f"Experiment failed for {config_name} seed {seed}: {e}")
                continue

        # Aggregate results
        if config_results:
            all_results[config_name] = {
                'mean_reward': np.mean([r['final_reward'] for r in config_results]),
                'std_reward': np.std([r['final_reward'] for r in config_results]),
                'mean_social_welfare': np.mean([r['mean_social_welfare'] for r in config_results]),
                'mean_gini': np.mean([r['mean_gini'] for r in config_results]),
                'mean_cooperation': np.mean([r['mean_cooperation'] for r in config_results]),
                'num_seeds': len(config_results),
                'individual_results': config_results,
            }

    return all_results


def print_results_table(results: dict):
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"{'Config':<25} {'Reward':<15} {'Welfare':<15} {'Gini':<10} {'Coop':<10}")
    print("-" * 80)

    for config_name, data in results.items():
        print(f"{config_name:<25} "
              f"{data['mean_reward']:>6.2f} ± {data['std_reward']:<6.2f} "
              f"{data['mean_social_welfare']:>10.2f} "
              f"{data['mean_gini']:>8.3f} "
              f"{data['mean_cooperation']:>8.3f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Contemplative MARL experiments")

    # Experiment type
    parser.add_argument('--config', type=str, default='full',
                       choices=['full', 'baseline'],
                       help='Which configuration to run')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study')
    parser.add_argument('--scaling', action='store_true',
                       help='Run scaling experiments')
    parser.add_argument('--robustness', action='store_true',
                       help='Run robustness experiments')

    # Experiment parameters
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of random seeds')
    parser.add_argument('--agents', type=int, nargs='+', default=[4, 8, 16],
                       help='Agent counts for scaling experiments')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps')

    # Hardware
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')

    # Output
    parser.add_argument('--output', type=str, default='experiments',
                       help='Output directory')

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Determine which experiments to run
    if args.ablation:
        configs = get_ablation_configs()
        logger.info("Running ablation study")
    elif args.scaling:
        configs = get_scaling_configs(args.agents)
        logger.info(f"Running scaling experiments: {args.agents}")
    elif args.robustness:
        configs = get_robustness_configs()
        logger.info("Running robustness experiments")
    else:
        if args.config == 'full':
            configs = {'full': get_full_config()}
        else:
            configs = {'baseline': get_baseline_config()}

    # Update timesteps
    for config in configs.values():
        config.training.total_timesteps = args.timesteps

    # Run experiments
    results = run_experiment_suite(
        configs=configs,
        num_seeds=args.seeds,
        output_dir=output_dir,
        device=args.device,
    )

    # Save aggregate results
    with open(output_dir / 'aggregate_results.json', 'w') as f:
        # Remove individual results for cleaner output
        clean_results = {
            k: {kk: vv for kk, vv in v.items() if kk != 'individual_results'}
            for k, v in results.items()
        }
        json.dump(clean_results, f, indent=2)

    # Print table
    print_results_table(results)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
