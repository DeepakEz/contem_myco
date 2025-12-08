"""
MycoNet 3.0 Main Orchestrator
=============================

Entry point for running MycoNet 3.0 simulations.
Provides CLI interface and programmatic API.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from .config import (
    MycoNetConfig,
    create_minimal_config,
    create_basic_config,
    create_advanced_config,
    create_scalability_test_config
)
from .environment import Environment
from .uprt_field import UPRTField
from .myco_agent import MycoAgent
from .overmind import Overmind
from .field_metrics import MetricsComputer, ColonyMetrics
from .hypernetwork import GenomeHyperNet, EvolutionEngine, TargetArchitecture
from .training_pipeline import TrainingPipeline, create_training_pipeline
from .evaluation import Evaluator, create_evaluator, EvaluationResult
from .scenarios import ScenarioRunner, ScenarioType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MycoNetSimulation:
    """
    Main simulation orchestrator for MycoNet 3.0.

    Coordinates all system components for training, evaluation, and visualization.
    """

    def __init__(self, config: MycoNetConfig = None):
        """
        Initialize the simulation.

        Args:
            config: Configuration object. Uses default if None.
        """
        self.config = config or MycoNetConfig()

        if not self.config.validate():
            raise ValueError("Invalid configuration")

        # Set random seed
        np.random.seed(self.config.random_seed)

        # Initialize core components
        logger.info("Initializing MycoNet 3.0 simulation...")

        self.environment = Environment(self.config)
        self.uprt_field = UPRTField(self.config)
        self.overmind = Overmind(self.config)
        self.metrics_computer = MetricsComputer(self.config)

        # Initialize agents
        self.agents: List[MycoAgent] = []
        self._initialize_agents()

        # Training and evaluation
        self.training_pipeline: Optional[TrainingPipeline] = None
        self.evaluator = create_evaluator(self.config)

        # State tracking
        self.step_count = 0
        self.episode_count = 0
        self.state_history: List[np.ndarray] = []
        self.field_history: List[np.ndarray] = []
        self.metrics_history: List[ColonyMetrics] = []

        logger.info(f"Simulation initialized with {len(self.agents)} agents")

    def _initialize_agents(self):
        """Create and initialize agents."""
        self.agents = []

        for i in range(self.config.num_agents):
            agent = MycoAgent(self.config, agent_id=i)
            position = self.environment.get_random_valid_position()
            agent.reset(position=position)
            self.agents.append(agent)

        logger.info(f"Created {len(self.agents)} agents")

    def reset(self):
        """Reset simulation to initial state."""
        self.environment.reset()
        self.uprt_field.reset()

        for agent in self.agents:
            position = self.environment.get_random_valid_position()
            agent.reset(position=position)

        self.step_count = 0
        self.state_history = []
        self.field_history = []
        self.metrics_history = []

        logger.info("Simulation reset")

    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation step.

        Returns:
            Dictionary with step results and metrics
        """
        step_start = time.time()

        # Collect current states
        agent_states = np.array([a.get_state_vector() for a in self.agents])
        self.state_history.append(agent_states)

        # Each agent observes and acts
        observations = {}
        actions = {}

        for agent in self.agents:
            if not agent.alive:
                continue

            # Get observation
            obs = self._get_observation(agent)
            observations[agent.agent_id] = obs

            # Select action
            action, _, _ = agent.select_action(obs)
            actions[agent.agent_id] = action

        # Execute actions
        rewards = {}
        for agent in self.agents:
            if not agent.alive:
                continue

            action = actions.get(agent.agent_id)
            if action is not None:
                reward = self._execute_action(agent, action)
                rewards[agent.agent_id] = reward

        # Update environment
        self.environment.propagate_signals()
        self.environment.regenerate_resources()

        # Update UPRT field
        current_states = np.array([a.get_state_vector() for a in self.agents])
        self.uprt_field.step(current_states)
        self.field_history.append(self.uprt_field.get_field_state())

        # Overmind reflection
        if self.step_count % 10 == 0:
            field_state = self.uprt_field.get_field_state()
            self.overmind.step([a.get_state_vector() for a in self.agents], field_state)

        # Compute metrics
        if len(self.state_history) >= 10:
            recent_states = np.array(self.state_history[-10:])
            field_state = self.uprt_field.get_field_state()
            metrics = self.metrics_computer.compute_all(recent_states, field_state)
            self.metrics_history.append(metrics)
        else:
            metrics = None

        self.step_count += 1
        step_time = time.time() - step_start

        return {
            'step': self.step_count,
            'alive_agents': sum(1 for a in self.agents if a.alive),
            'total_reward': sum(rewards.values()),
            'metrics': metrics,
            'step_time': step_time
        }

    def _get_observation(self, agent: MycoAgent) -> np.ndarray:
        """Get observation for an agent."""
        local_terrain = self.environment.get_local_view(agent.position, radius=2)
        local_signals = self.environment.get_local_signals(agent.position, radius=2)
        agent_state = agent.get_state_vector()[:10]

        return np.concatenate([
            local_terrain.flatten(),
            local_signals.flatten(),
            agent_state
        ]).astype(np.float32)

    def _execute_action(self, agent: MycoAgent, action: int) -> float:
        """Execute an action for an agent."""
        reward = 0.0

        # Movement actions (0-3)
        if action < 4:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dx, dy = directions[action]
            new_pos = (agent.position[0] + dx, agent.position[1] + dy)

            if self.environment.is_valid_position(new_pos):
                agent.position = new_pos
                agent.energy -= 0.01

                resource = self.environment.get_resource_at(new_pos)
                if resource > 0:
                    reward += resource * 0.1

        # Signal emission (4-11)
        elif action < 12:
            signal_type = action - 4
            self.environment.emit_signal(agent.position, signal_type, agent.energy * 0.5)
            agent.energy -= 0.02

        # Harvest (12)
        elif action == 12:
            harvested = self.environment.harvest_resource(agent.position)
            agent.energy += harvested
            reward += harvested

        # Meditate (13)
        elif action == 13:
            agent.mindfulness_state['focus_coherence'] += 0.1
            agent.mindfulness_state['focus_coherence'] = min(
                1.0, agent.mindfulness_state['focus_coherence']
            )
            agent.energy -= 0.005

        # Share wisdom (14)
        elif action == 14:
            agent.energy -= 0.03
            reward += 0.05

        # Base cost
        agent.energy -= 0.001
        if agent.energy <= 0:
            agent.alive = False
            reward -= 1.0

        return reward

    def run_episode(self, max_steps: int = None) -> Dict[str, Any]:
        """
        Run a complete episode.

        Args:
            max_steps: Maximum steps per episode

        Returns:
            Episode summary
        """
        if max_steps is None:
            max_steps = self.config.training.rl_max_steps

        self.reset()
        episode_start = time.time()

        total_reward = 0.0
        step_results = []

        for step in range(max_steps):
            result = self.step()
            step_results.append(result)
            total_reward += result['total_reward']

            # Check termination
            if result['alive_agents'] == 0:
                break

        episode_time = time.time() - episode_start
        self.episode_count += 1

        # Final metrics
        final_metrics = self.metrics_history[-1] if self.metrics_history else None

        return {
            'episode': self.episode_count,
            'steps': self.step_count,
            'total_reward': total_reward,
            'final_alive': sum(1 for a in self.agents if a.alive),
            'metrics': final_metrics,
            'episode_time': episode_time
        }

    def train(self, num_generations: int = None,
              num_episodes: int = None) -> Dict[str, Any]:
        """
        Run full training procedure.

        Args:
            num_generations: Override evolution generations
            num_episodes: Override RL episodes

        Returns:
            Training results
        """
        logger.info("Starting training...")

        # Override config if specified
        if num_generations:
            self.config.training.evolution_generations = num_generations
        if num_episodes:
            self.config.training.rl_episodes = num_episodes

        # Create training pipeline
        self.training_pipeline = create_training_pipeline(self.config)

        # Run training
        results = self.training_pipeline.train()

        logger.info("Training complete")
        return results

    def evaluate(self, scenarios: List[ScenarioType] = None) -> EvaluationResult:
        """
        Run comprehensive evaluation.

        Args:
            scenarios: Specific scenarios to run (all if None)

        Returns:
            Evaluation result
        """
        logger.info("Running evaluation...")

        # Collect data from episode runs
        self.reset()

        for _ in range(5):  # Run 5 evaluation episodes
            self.run_episode()

        # Prepare data
        state_history = np.array(self.state_history) if self.state_history else np.zeros((1, self.config.num_agents, 10))
        field_states = np.array(self.field_history) if self.field_history else np.zeros((1, 64, 64))

        # Run evaluation
        result = self.evaluator.evaluate_colony(state_history, field_states)

        # Run scenario evaluation if specified
        if scenarios:
            scenario_results = self.evaluator.run_scenario_evaluation(
                self.environment, self.agents, scenarios
            )
            result.scenario_results = scenario_results

        logger.info(f"Evaluation complete: score={result.overall_score:.3f}")
        return result

    def run_demo(self, num_steps: int = 100):
        """
        Run a demonstration simulation with logging.

        Args:
            num_steps: Number of steps to run
        """
        logger.info(f"Running demo for {num_steps} steps...")

        self.reset()

        for step in range(num_steps):
            result = self.step()

            if step % 10 == 0:
                metrics = result.get('metrics')
                if metrics:
                    logger.info(
                        f"Step {step}: alive={result['alive_agents']}, "
                        f"coherence={metrics.coherence:.3f}, "
                        f"RX={metrics.recoverability_index:.3f}"
                    )
                else:
                    logger.info(f"Step {step}: alive={result['alive_agents']}")

        logger.info("Demo complete")

    def save_state(self, filepath: str):
        """Save current simulation state."""
        state = {
            'config': self.config.to_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'agent_states': [a.get_state_vector().tolist() for a in self.agents],
            'field_state': self.uprt_field.get_field_state().tolist()
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"State saved to {filepath}")

    def load_state(self, filepath: str):
        """Load simulation state."""
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)

        # Restore config
        self.config = MycoNetConfig.from_dict(state['config'])

        # Restore counters
        self.step_count = state['step_count']
        self.episode_count = state['episode_count']

        # Restore agents
        for i, agent_state in enumerate(state['agent_states']):
            if i < len(self.agents):
                self.agents[i].load_state_vector(np.array(agent_state))

        # Restore field
        self.uprt_field.set_field_state(np.array(state['field_state']))

        logger.info(f"State loaded from {filepath}")


def run_simulation(config: MycoNetConfig = None,
                  mode: str = 'demo',
                  **kwargs) -> Any:
    """
    Convenience function to run simulation.

    Args:
        config: Configuration object
        mode: 'demo', 'train', or 'evaluate'
        **kwargs: Additional arguments for the mode

    Returns:
        Results based on mode
    """
    sim = MycoNetSimulation(config)

    if mode == 'demo':
        num_steps = kwargs.get('num_steps', 100)
        sim.run_demo(num_steps)
        return sim

    elif mode == 'train':
        num_generations = kwargs.get('num_generations')
        num_episodes = kwargs.get('num_episodes')
        return sim.train(num_generations, num_episodes)

    elif mode == 'evaluate':
        scenarios = kwargs.get('scenarios')
        return sim.evaluate(scenarios)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='MycoNet 3.0: Bio-Inspired Multi-Agent Cognitive Architecture'
    )

    parser.add_argument(
        '--mode', type=str, default='demo',
        choices=['demo', 'train', 'evaluate', 'benchmark'],
        help='Simulation mode'
    )

    parser.add_argument(
        '--config', type=str, default='basic',
        choices=['minimal', 'basic', 'advanced', 'custom'],
        help='Configuration preset'
    )

    parser.add_argument(
        '--config-file', type=str, default=None,
        help='Path to custom config JSON file'
    )

    parser.add_argument(
        '--num-agents', type=int, default=None,
        help='Number of agents'
    )

    parser.add_argument(
        '--num-steps', type=int, default=100,
        help='Number of steps for demo mode'
    )

    parser.add_argument(
        '--num-generations', type=int, default=None,
        help='Number of evolution generations for training'
    )

    parser.add_argument(
        '--num-episodes', type=int, default=None,
        help='Number of RL episodes for training'
    )

    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path for results'
    )

    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create configuration
    if args.config_file:
        config = MycoNetConfig.load(args.config_file)
    elif args.config == 'minimal':
        config = create_minimal_config()
    elif args.config == 'basic':
        config = create_basic_config()
    elif args.config == 'advanced':
        config = create_advanced_config()
    else:
        config = MycoNetConfig()

    # Override with command line args
    if args.num_agents:
        config.num_agents = args.num_agents
    config.random_seed = args.seed

    logger.info(f"MycoNet 3.0 - Mode: {args.mode}, Config: {args.config}")
    logger.info(f"Agents: {config.num_agents}, Seed: {config.random_seed}")

    # Run simulation
    try:
        if args.mode == 'demo':
            sim = run_simulation(config, 'demo', num_steps=args.num_steps)

            # Save final state if output specified
            if args.output:
                sim.save_state(args.output)

        elif args.mode == 'train':
            results = run_simulation(
                config, 'train',
                num_generations=args.num_generations,
                num_episodes=args.num_episodes
            )

            if args.output:
                import json
                # Convert numpy arrays to lists for JSON serialization
                def convert_to_serializable(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(item) for item in obj]
                    return obj

                with open(args.output, 'w') as f:
                    json.dump(convert_to_serializable(results), f, indent=2)
                logger.info(f"Training results saved to {args.output}")

        elif args.mode == 'evaluate':
            result = run_simulation(config, 'evaluate')

            # Generate and print report
            sim = MycoNetSimulation(config)
            report = sim.evaluator.generate_report(result, args.output)
            print(report)

        elif args.mode == 'benchmark':
            # Run all scenarios as benchmark
            sim = MycoNetSimulation(config)
            scenarios = list(ScenarioType)
            result = sim.evaluate(scenarios)

            report = sim.evaluator.generate_report(result, args.output)
            print(report)

        logger.info("Simulation complete")

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise


if __name__ == '__main__':
    main()
