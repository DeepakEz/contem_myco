#!/usr/bin/env python3
"""
Experiment Runner - Reusable Framework
=======================================
Configurable experiment runner for comparing agents across environments.

Features:
- Swappable environments (Resilience, Society)
- Swappable agent types (Reactive/Contemplative, Baseline/Myco)
- Automated logging and metrics collection
- Reproducible experiments with seeds
- Export results for analysis
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
import time

# Import environments
from resilience_env import ResilienceEnv
from society_env import SocietyEnv, PolicyType

# Import agents
from resilience_agent import create_agent as create_resilience_agent
from policy_agent import create_policy_agent

# Import MycoAgent core
from unified_logger import get_logger
from compute_profiler import get_registry as get_profiler_registry

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    experiment_name: str
    environment_type: str  # 'resilience' or 'society'
    agent_type: str  # 'reactive', 'contemplative', 'baseline', 'myco'
    num_steps: int = 200
    num_runs: int = 3
    seed: Optional[int] = None

    # Environment-specific configs
    env_config: Dict[str, Any] = None

    # Agent-specific configs
    agent_config: Dict[str, Any] = None

    # Output
    output_dir: str = "results"

    def __post_init__(self):
        if self.env_config is None:
            self.env_config = {}
        if self.agent_config is None:
            self.agent_config = {}


class ExperimentRunner:
    """
    Reusable experiment runner

    Handles:
    - Environment setup
    - Agent creation
    - Simulation execution
    - Metrics collection
    - Results export
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = get_logger(
            config.experiment_name,
            output_dir=str(self.output_dir)
        )

        # Store metadata
        self.logger.set_metadata('experiment_name', config.experiment_name)
        self.logger.set_metadata('environment_type', config.environment_type)
        self.logger.set_metadata('agent_type', config.agent_type)
        self.logger.set_metadata('num_steps', config.num_steps)
        self.logger.set_metadata('num_runs', config.num_runs)

        # Results storage
        self.run_results = []

    def run_experiment(self) -> Dict[str, Any]:
        """
        Run full experiment (multiple runs)

        Returns:
            Aggregated results
        """
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        logger.info(f"  Environment: {self.config.environment_type}")
        logger.info(f"  Agent: {self.config.agent_type}")
        logger.info(f"  Runs: {self.config.num_runs}")
        logger.info(f"  Steps per run: {self.config.num_steps}")

        for run_idx in range(self.config.num_runs):
            run_seed = self.config.seed + run_idx if self.config.seed else None

            logger.info(f"\n--- Run {run_idx + 1}/{self.config.num_runs} (seed={run_seed}) ---")

            run_result = self._run_single_experiment(run_idx, run_seed)
            self.run_results.append(run_result)

            logger.info(f"Run {run_idx + 1} completed:")
            logger.info(f"  Final metrics: {run_result['final_metrics']}")

        # Aggregate results
        aggregated = self._aggregate_results()

        # Export
        self._export_results(aggregated)

        logger.info(f"\nExperiment completed: {self.config.experiment_name}")
        logger.info(f"Results saved to: {self.output_dir}")

        return aggregated

    def _run_single_experiment(self, run_idx: int, seed: Optional[int]) -> Dict[str, Any]:
        """Run a single experiment instance"""

        # Create environment
        env = self._create_environment(seed)

        # Create agents
        if self.config.environment_type == 'resilience':
            agents = self._create_resilience_agents()
        elif self.config.environment_type == 'society':
            agents = self._create_policy_agent()
        else:
            raise ValueError(f"Unknown environment type: {self.config.environment_type}")

        # Reset environment
        obs = env.reset()

        # Tracking
        step_metrics = []
        total_reward = 0.0

        # Run simulation
        for step in range(self.config.num_steps):
            # Select actions
            if self.config.environment_type == 'resilience':
                actions = self._select_resilience_actions(agents, obs, env)
                obs, rewards, dones, info = env.step(actions)

                step_reward = sum(rewards.values())

                # Log agent actions
                for agent_id, agent in agents.items():
                    if agent_id in rewards:
                        self.logger.log_agent_action(
                            step=step,
                            agent_id=agent_id,
                            action=actions.get(agent_id, 'none'),
                            ethical_score=agent.get_decision_info().get('avg_ethical_score', 0.5)
                                        if hasattr(agent, 'get_decision_info') else 0.5,
                            mindfulness=agent.get_decision_info().get('mindfulness', 0.5)
                                      if hasattr(agent, 'get_decision_info') else 0.5,
                            reward=rewards[agent_id]
                        )

            elif self.config.environment_type == 'society':
                agent = agents
                policy = agent.select_policy(obs, step)
                obs, step_reward, done, info = env.step(policy)

                # Log policy decision
                self.logger.log_agent_action(
                    step=step,
                    agent_id=0,
                    action=policy.value,
                    ethical_score=agent.get_decision_info().get('avg_ethical_score', 0.5),
                    mindfulness=0.5,  # Societies don't have mindfulness
                    reward=step_reward
                )

                dones = {'__all__': done}

            total_reward += step_reward

            # Log environment state
            self.logger.log_environment_state(step, info)

            # Log key metrics
            self._log_step_metrics(step, info)

            # Store metrics
            step_metrics.append(info)

            # Check if done
            if dones.get('__all__', False):
                logger.info(f"Simulation ended at step {step}")
                break

            # Periodic rendering
            if step % 50 == 0:
                logger.debug(f"\nStep {step}:")
                logger.debug(env.render())

        # Final statistics
        final_metrics = self._compute_final_metrics(step_metrics)

        return {
            'run_idx': run_idx,
            'seed': seed,
            'total_steps': len(step_metrics),
            'total_reward': total_reward,
            'step_metrics': step_metrics,
            'final_metrics': final_metrics
        }

    def _create_environment(self, seed: Optional[int]):
        """Create environment instance"""
        if self.config.environment_type == 'resilience':
            return ResilienceEnv(
                seed=seed,
                **self.config.env_config
            )

        elif self.config.environment_type == 'society':
            return SocietyEnv(
                seed=seed,
                **self.config.env_config
            )

        else:
            raise ValueError(f"Unknown environment: {self.config.environment_type}")

    def _create_resilience_agents(self) -> Dict[int, Any]:
        """Create resilience agents"""
        num_agents = self.config.env_config.get('num_agents', 10)

        agents = {}
        for agent_id in range(num_agents):
            agents[agent_id] = create_resilience_agent(
                agent_id=agent_id,
                agent_type=self.config.agent_type,
                config=self.config.agent_config
            )

        return agents

    def _create_policy_agent(self) -> Any:
        """Create policy agent"""
        return create_policy_agent(
            agent_type=self.config.agent_type,
            agent_id=0,
            config=self.config.agent_config
        )

    def _select_resilience_actions(
        self,
        agents: Dict[int, Any],
        observations: Dict[int, Any],
        env: ResilienceEnv
    ) -> Dict[int, str]:
        """Select actions for resilience agents"""
        actions = {}

        env_info = env.get_statistics()

        for agent_id, agent in agents.items():
            agent_state = env.agents.get(agent_id)

            if agent_state and agent_state.is_alive:
                obs = observations.get(agent_id, np.zeros(20))

                state_dict = {
                    'health': agent_state.health,
                    'energy': agent_state.energy,
                    'suffering_level': agent_state.suffering_level,
                    'has_food': agent_state.has_food,
                    'has_shelter': agent_state.has_shelter,
                    'has_medical': agent_state.has_medical
                }

                action = agent.select_action(obs, state_dict, env_info)
                actions[agent_id] = action

        return actions

    def _log_step_metrics(self, step: int, info: Dict[str, Any]):
        """Log key metrics for this step"""
        if self.config.environment_type == 'resilience':
            self.logger.log_result(step, 'casualties', info.get('casualties', 0))
            self.logger.log_result(step, 'avg_suffering', info.get('avg_suffering', 0.0))
            self.logger.log_result(step, 'total_rescues', info.get('total_rescues', 0))
            self.logger.log_result(step, 'alive_agents', info.get('alive_agents', 0))

        elif self.config.environment_type == 'society':
            self.logger.log_result(step, 'inequality', info.get('inequality', 0.0))
            self.logger.log_result(step, 'avg_trust', info.get('avg_trust', 0.0))
            self.logger.log_result(step, 'avg_suffering', info.get('avg_suffering', 0.0))
            self.logger.log_result(step, 'crime_rate', info.get('crime_rate', 0.0))

    def _compute_final_metrics(self, step_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute final aggregate metrics"""
        if not step_metrics:
            return {}

        final = {}

        if self.config.environment_type == 'resilience':
            final['total_casualties'] = step_metrics[-1].get('casualties', 0)
            final['avg_suffering'] = np.mean([m.get('avg_suffering', 0) for m in step_metrics])
            final['total_rescues'] = step_metrics[-1].get('total_rescues', 0)
            final['survival_rate'] = step_metrics[-1].get('alive_agents', 0) / step_metrics[0].get('alive_agents', 1)

        elif self.config.environment_type == 'society':
            final['final_inequality'] = step_metrics[-1].get('inequality', 0.0)
            final['avg_trust'] = np.mean([m.get('avg_trust', 0) for m in step_metrics])
            final['avg_suffering'] = np.mean([m.get('avg_suffering', 0) for m in step_metrics])
            final['total_crimes'] = step_metrics[-1].get('total_crimes', 0)
            final['final_homeless_rate'] = step_metrics[-1].get('homeless_rate', 0.0)

        return final

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all runs"""
        if not self.run_results:
            return {}

        # Compute means and stds
        aggregated = {
            'config': asdict(self.config),
            'num_runs': len(self.run_results),
            'mean_metrics': {},
            'std_metrics': {},
            'all_runs': self.run_results
        }

        # Extract final metrics from all runs
        if self.run_results:
            final_metrics_keys = self.run_results[0]['final_metrics'].keys()

            for key in final_metrics_keys:
                values = [run['final_metrics'][key] for run in self.run_results]
                aggregated['mean_metrics'][key] = np.mean(values)
                aggregated['std_metrics'][key] = np.std(values)

        # Compute profiler stats
        profiler_registry = get_profiler_registry()
        profiler_summary = profiler_registry.get_aggregate_summary()
        aggregated['compute_metrics'] = profiler_summary

        return aggregated

    def _export_results(self, aggregated: Dict[str, Any]):
        """Export results to files"""
        # Export aggregated JSON
        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(aggregated, f, indent=2)

        logger.info(f"Exported results to {results_file}")

        # Export profiler metrics
        profiler_registry = get_profiler_registry()
        profiler_dir = self.output_dir / "profiler"
        profiler_registry.export_all(str(profiler_dir))

        # Finalize logger
        self.logger.finalize()


# ===== CONVENIENCE FUNCTIONS =====

def run_resilience_experiment(
    agent_type: str,
    num_steps: int = 200,
    num_runs: int = 3,
    seed: Optional[int] = None,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Convenience function to run resilience experiment

    Args:
        agent_type: 'reactive' or 'contemplative'
        num_steps: Steps per run
        num_runs: Number of runs
        seed: Random seed
        output_dir: Output directory

    Returns:
        Aggregated results
    """
    config = ExperimentConfig(
        experiment_name=f"resilience_{agent_type}",
        environment_type='resilience',
        agent_type=agent_type,
        num_steps=num_steps,
        num_runs=num_runs,
        seed=seed,
        env_config={'num_agents': 10, 'grid_size': 20},
        output_dir=output_dir
    )

    runner = ExperimentRunner(config)
    return runner.run_experiment()


def run_society_experiment(
    agent_type: str,
    num_steps: int = 200,
    num_runs: int = 3,
    seed: Optional[int] = None,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Convenience function to run society experiment

    Args:
        agent_type: 'baseline' or 'myco'
        num_steps: Steps per run
        num_runs: Number of runs
        seed: Random seed
        output_dir: Output directory

    Returns:
        Aggregated results
    """
    config = ExperimentConfig(
        experiment_name=f"society_{agent_type}",
        environment_type='society',
        agent_type=agent_type,
        num_steps=num_steps,
        num_runs=num_runs,
        seed=seed,
        env_config={'num_citizens': 100},
        output_dir=output_dir
    )

    runner = ExperimentRunner(config)
    return runner.run_experiment()


if __name__ == "__main__":
    # Quick test
    print("Experiment Runner Test:")
    print("  Testing resilience environment with reactive agents...")

    # Small test experiment
    test_config = ExperimentConfig(
        experiment_name="test_resilience_reactive",
        environment_type='resilience',
        agent_type='reactive',
        num_steps=10,
        num_runs=1,
        seed=42,
        env_config={'num_agents': 5, 'grid_size': 20},
        output_dir="test_results"
    )

    runner = ExperimentRunner(test_config)
    results = runner.run_experiment()

    print(f"\n  Test completed!")
    print(f"  Final metrics: {results['mean_metrics']}")
    print("\n✓ Experiment Runner initialized successfully")
