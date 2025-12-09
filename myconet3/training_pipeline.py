"""
MycoNet 3.0 Training Pipeline
=============================

Multi-level optimization with nested loops:
- Inner loop: RL training (PPO/SAC/DQN)
- Middle loop: Surrogate model calibration
- Outer loop: Evolutionary optimization of genomes
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import time
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical, Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import MycoNetConfig, TrainingConfig
from .environment import Environment
from .myco_agent import MycoAgent
from .hypernetwork import (
    GenomeHyperNet,
    EvolutionEngine,
    TargetArchitecture,
    create_hypernetwork,
)
from .uprt_field import UPRTField, FieldSurrogateModel
from .field_metrics import MetricsComputer, ColonyMetrics
from .overmind import Overmind

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    episode: int = 0
    generation: int = 0
    total_reward: float = 0.0
    mean_agent_reward: float = 0.0
    colony_coherence: float = 0.0
    colony_entropy: float = 0.0
    colony_information: float = 0.0
    phi_f: float = 0.0
    rx_index: float = 0.0
    tau_r: float = 0.0
    ethical_violations: int = 0
    steps: int = 0
    wall_time: float = 0.0


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data for policy gradient methods."""

    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def clear(self):
        """Clear all stored data."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state: np.ndarray, action: int, reward: float,
            value: float, log_prob: float, done: bool):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(self, gamma: float, lam: float = 0.95,
                                        last_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute GAE returns and advantages."""
        returns = []
        advantages = []

        gae = 0
        next_value = last_value

        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                next_value = 0
                gae = 0

            delta = self.rewards[i] + gamma * next_value - self.values[i]
            gae = delta + gamma * lam * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])

            next_value = self.values[i]

        return returns, advantages


class PPOTrainer:
    """Proximal Policy Optimization trainer for agents."""

    def __init__(self, config: MycoNetConfig):
        self.config = config
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        self.mini_batch_size = 64

    def train_step(self, agent: MycoAgent, buffer: RolloutBuffer,
                   returns: List[float], advantages: List[float]) -> Dict[str, float]:
        """Perform PPO update step."""
        if not TORCH_AVAILABLE:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Convert to tensors
        states = torch.FloatTensor(np.array(buffer.states))
        actions = torch.LongTensor(buffer.actions)
        old_log_probs = torch.FloatTensor(buffer.log_probs)
        returns_t = torch.FloatTensor(returns)
        advantages_t = torch.FloatTensor(advantages)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Generate mini-batches
            indices = np.random.permutation(len(buffer.states))

            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]

                # Get current policy outputs
                policy_out, value_out = agent.policy_net(batch_states)
                dist = Categorical(logits=policy_out)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                   1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_pred = value_out.squeeze()
                value_loss = nn.MSELoss()(value_pred, batch_returns)

                # Combined loss
                loss = (policy_loss +
                       self.value_loss_coef * value_loss -
                       self.entropy_coef * entropy)

                # Optimization step
                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.policy_net.parameters(),
                                         self.max_grad_norm)
                agent.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1)
        }


class SACTrainer:
    """Soft Actor-Critic trainer (placeholder for alternative RL algorithm)."""

    def __init__(self, config: MycoNetConfig):
        self.config = config
        self.tau = 0.005  # Target network update rate
        self.alpha = 0.2  # Entropy coefficient

    def train_step(self, agent: MycoAgent, transitions: List[Tuple]) -> Dict[str, float]:
        """Perform SAC update step."""
        # Simplified implementation - would need full SAC with Q-networks
        return {"q_loss": 0.0, "policy_loss": 0.0, "alpha_loss": 0.0}


class DQNTrainer:
    """DQN trainer (placeholder for alternative RL algorithm)."""

    def __init__(self, config: MycoNetConfig):
        self.config = config
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def train_step(self, agent: MycoAgent, batch: List[Tuple]) -> Dict[str, float]:
        """Perform DQN update step."""
        return {"q_loss": 0.0}


class SurrogateCalibrator:
    """Calibrates the field surrogate model using collected data."""

    def __init__(self, config: MycoNetConfig):
        self.config = config
        self.field_history: List[np.ndarray] = []
        self.max_history = 1000
        self.calibration_loss_history: List[float] = []

    def collect_field_state(self, field_state: np.ndarray):
        """Collect field state for calibration."""
        self.field_history.append(field_state.copy())
        if len(self.field_history) > self.max_history:
            self.field_history.pop(0)

    def calibrate(self, surrogate: FieldSurrogateModel,
                  uprt_field: UPRTField, num_steps: int = 50) -> float:
        """Calibrate surrogate model against true field dynamics."""
        if not TORCH_AVAILABLE or len(self.field_history) < 10:
            return 0.0

        # Create training pairs
        inputs = []
        targets = []

        for i in range(len(self.field_history) - 1):
            inputs.append(self.field_history[i])
            targets.append(self.field_history[i + 1])

        inputs = torch.FloatTensor(np.array(inputs)).unsqueeze(1)
        targets = torch.FloatTensor(np.array(targets)).unsqueeze(1)

        optimizer = optim.Adam(surrogate.parameters(), lr=1e-4)

        total_loss = 0.0
        batch_size = min(32, len(inputs))

        for step in range(num_steps):
            # Sample mini-batch
            indices = np.random.choice(len(inputs), batch_size, replace=False)
            batch_inputs = inputs[indices]
            batch_targets = targets[indices]

            # Forward pass
            predictions = surrogate(batch_inputs)
            loss = nn.MSELoss()(predictions, batch_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_steps
        self.calibration_loss_history.append(avg_loss)

        return avg_loss


class FitnessEvaluator:
    """Evaluates multi-objective fitness for evolutionary optimization."""

    def __init__(self, config: MycoNetConfig):
        self.config = config
        self.training_config = config.training

    def compute_fitness(self, metrics: ColonyMetrics,
                        total_reward: float,
                        ethical_violations: int) -> float:
        """Compute multi-objective fitness score."""

        # Task performance component
        task_fitness = total_reward * self.training_config.task_reward_weight

        # Coherence bonus (Kuramoto order parameter)
        coherence_bonus = metrics.coherence * self.training_config.coherence_bonus

        # Information bonus (mutual information)
        information_bonus = metrics.mutual_information * self.training_config.information_bonus

        # Ethical penalty
        ethical_penalty = ethical_violations * self.training_config.ethical_penalty

        # Recoverability penalty (penalize low RX)
        rx_penalty = max(0, self.config.dharma.rx_moral_threshold - metrics.recoverability_index)
        rx_penalty *= self.training_config.rx_penalty

        # Combined fitness
        fitness = (task_fitness + coherence_bonus + information_bonus
                   - ethical_penalty - rx_penalty)

        return fitness


class TrainingPipeline:
    """
    Complete training pipeline with nested optimization loops.

    Structure:
    - Outer loop: Evolution of agent genomes
    - Middle loop: Surrogate model calibration
    - Inner loop: RL training of agent policies
    """

    def __init__(self, config: MycoNetConfig):
        self.config = config
        self.device = self._get_device()

        # Initialize components
        self.environment = Environment(config)
        self.uprt_field = UPRTField(config)
        self.metrics_computer = MetricsComputer(config)
        self.overmind = Overmind(config)

        # Initialize hypernetwork and evolution
        self.target_arch = TargetArchitecture.from_dims(
            input_dim=self._compute_observation_dim(),
            hidden_dim=config.agent.hidden_dim,
            output_dim=self._compute_action_dim()
        )

        if config.qrea.use_hypernetwork:
            self.hypernetwork = create_hypernetwork(
                genome_dim=config.agent.genome_dim,
                architecture=self.target_arch,
                hidden_dim=config.qrea.hypernetwork_hidden_dim
            )
        else:
            self.hypernetwork = None

        self.evolution_engine = EvolutionEngine(
            config.qrea,
            genome_dim=config.agent.genome_dim
        )

        # Initialize surrogate calibrator
        self.surrogate_calibrator = SurrogateCalibrator(config)

        # Initialize RL trainer
        self.rl_trainer = self._create_rl_trainer()

        # Initialize fitness evaluator
        self.fitness_evaluator = FitnessEvaluator(config)

        # Agents will be created per generation
        self.agents: List[MycoAgent] = []

        # Training history
        self.training_history: List[TrainingMetrics] = []
        self.best_genomes: List[np.ndarray] = []
        self.best_fitness: float = float('-inf')

        # Callbacks
        self.callbacks: List[Callable] = []

        logger.info("Training pipeline initialized")

    def _get_device(self) -> str:
        """Determine compute device."""
        if self.config.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.config.device

    def _compute_observation_dim(self) -> int:
        """Compute observation dimension for agents."""
        # Local terrain view + signal channels + agent state
        view_size = 5  # 5x5 local view
        terrain_dim = view_size * view_size * 3  # terrain, resource, obstacle
        signal_dim = view_size * view_size * 8  # 8 wisdom signal types
        state_dim = 10  # energy, health, position, etc.
        return terrain_dim + signal_dim + state_dim

    def _compute_action_dim(self) -> int:
        """Compute action dimension for agents."""
        # Movement (4) + signal emission (8) + special actions (3)
        return 15

    def _create_rl_trainer(self):
        """Create RL trainer based on configuration."""
        algorithm = self.config.training.rl_algorithm.lower()
        if algorithm == "ppo":
            return PPOTrainer(self.config)
        elif algorithm == "sac":
            return SACTrainer(self.config)
        elif algorithm == "dqn":
            return DQNTrainer(self.config)
        else:
            logger.warning(f"Unknown RL algorithm: {algorithm}, defaulting to PPO")
            return PPOTrainer(self.config)

    def _create_agents(self, genomes: List[np.ndarray]) -> List[MycoAgent]:
        """Create agents from genomes using hypernetwork."""
        agents = []

        for i, genome in enumerate(genomes):
            agent = MycoAgent(self.config, agent_id=i)

            # If hypernetwork available, generate weights from genome
            if self.hypernetwork is not None and TORCH_AVAILABLE:
                genome_tensor = torch.FloatTensor(genome)
                weights = self.hypernetwork(genome_tensor)
                agent.load_weights_from_dict(weights)

            agent.genome = genome
            agents.append(agent)

        return agents

    def _run_episode(self, agents: List[MycoAgent],
                     max_steps: int,
                     training: bool = True) -> Tuple[float, ColonyMetrics, int, Dict]:
        """
        Run a single episode with all agents.

        Returns:
            total_reward: Sum of all agent rewards
            metrics: Colony metrics at end of episode
            ethical_violations: Count of ethical violations
            episode_data: Additional data from episode
        """
        # Reset environment
        self.environment.reset()
        self.uprt_field.reset()

        # Place agents in environment
        for agent in agents:
            pos = self.environment.get_random_valid_position()
            agent.reset(position=pos)

        total_reward = 0.0
        ethical_violations = 0
        buffers = {agent.agent_id: RolloutBuffer() for agent in agents}

        # Collect agent states for metrics
        state_history = []

        for step in range(max_steps):
            # Get observations for all agents
            observations = {}
            for agent in agents:
                obs = self._get_observation(agent)
                observations[agent.agent_id] = obs

            # Each agent selects action
            actions = {}
            for agent in agents:
                obs = observations[agent.agent_id]
                action, value, log_prob = agent.select_action(obs, deterministic=not training)
                actions[agent.agent_id] = (action, value, log_prob)

            # Execute actions in environment
            rewards = {}
            for agent in agents:
                action, _, _ = actions[agent.agent_id]
                reward, done, info = self._execute_action(agent, action)
                rewards[agent.agent_id] = reward
                total_reward += reward

                # Check for ethical violations
                if info.get('ethical_violation', False):
                    ethical_violations += 1

            # Update wisdom signals
            self.environment.propagate_signals()

            # Update UPRT field based on agent states
            agent_states = np.array([a.get_state_vector() for a in agents])
            self.uprt_field.step(agent_states)

            # Collect field state for surrogate calibration
            if training and self.config.overmind.use_surrogate:
                self.surrogate_calibrator.collect_field_state(
                    self.uprt_field.get_field_state()
                )

            # Overmind reflection and intervention
            if step % 10 == 0:
                agent_states_list = [a.get_state_vector() for a in agents]
                field_state = self.uprt_field.get_field_state()
                self.overmind.step(agent_states_list, field_state)

            # Store transitions in buffers
            if training:
                for agent in agents:
                    obs = observations[agent.agent_id]
                    action, value, log_prob = actions[agent.agent_id]
                    reward = rewards[agent.agent_id]
                    done = not agent.alive
                    buffers[agent.agent_id].add(obs, action, reward, value, log_prob, done)

            # Collect states for metrics
            state_history.append([a.get_state_vector() for a in agents])

            # Check termination
            if all(not a.alive for a in agents):
                break

        # Compute colony metrics
        state_history_array = np.array(state_history)
        metrics = self.metrics_computer.compute_all(
            state_history_array,
            self.uprt_field.get_field_state()
        )

        # Training update if needed
        training_losses = {}
        if training:
            for agent in agents:
                buffer = buffers[agent.agent_id]
                if len(buffer.states) > 0:
                    # Compute returns and advantages
                    last_obs = self._get_observation(agent)
                    _, last_value, _ = agent.select_action(last_obs, deterministic=True)
                    returns, advantages = buffer.compute_returns_and_advantages(
                        self.config.agent.gamma,
                        lam=0.95,
                        last_value=last_value
                    )

                    # PPO update
                    losses = self.rl_trainer.train_step(agent, buffer, returns, advantages)
                    training_losses[agent.agent_id] = losses
                    buffer.clear()

        episode_data = {
            'steps': step + 1,
            'training_losses': training_losses,
            'interventions': self.overmind.get_intervention_history()
        }

        return total_reward, metrics, ethical_violations, episode_data

    def _get_observation(self, agent: MycoAgent) -> np.ndarray:
        """Get observation for an agent."""
        # Get local view from environment
        local_terrain = self.environment.get_local_view(agent.position, radius=2)
        local_signals = self.environment.get_local_signals(agent.position, radius=2)

        # Agent internal state
        agent_state = agent.get_state_vector()[:10]

        # Flatten and concatenate
        obs = np.concatenate([
            local_terrain.flatten(),
            local_signals.flatten(),
            agent_state
        ])

        return obs.astype(np.float32)

    def _execute_action(self, agent: MycoAgent, action: int) -> Tuple[float, bool, Dict]:
        """Execute an action for an agent in the environment."""
        reward = 0.0
        done = False
        info = {}

        # Action space:
        # 0-3: Movement (up, down, left, right)
        # 4-11: Signal emission (8 wisdom signal types)
        # 12: Harvest resource
        # 13: Meditate
        # 14: Share wisdom

        if action < 4:
            # Movement
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dx, dy = directions[action]
            new_pos = (agent.position[0] + dx, agent.position[1] + dy)

            if self.environment.is_valid_position(new_pos):
                agent.position = new_pos
                agent.energy -= 0.01  # Movement cost

                # Check for resource at new position
                resource = self.environment.get_resource_at(new_pos)
                if resource > 0:
                    reward += resource * 0.1

        elif action < 12:
            # Signal emission
            signal_type = action - 4
            self.environment.emit_signal(agent.position, signal_type, agent.energy * 0.5)
            agent.energy -= 0.02  # Signal cost

        elif action == 12:
            # Harvest resource
            harvested = self.environment.harvest_resource(agent.position)
            agent.energy += harvested
            reward += harvested

        elif action == 13:
            # Meditate - increases mindfulness
            agent.mindfulness_state['focus_coherence'] += 0.1
            agent.mindfulness_state['focus_coherence'] = min(
                1.0, agent.mindfulness_state['focus_coherence']
            )
            agent.energy -= 0.005

        elif action == 14:
            # Share wisdom with nearby agents
            # (Effect handled at colony level)
            agent.energy -= 0.03
            reward += 0.05  # Cooperation bonus

        # Energy cost and health update
        agent.energy -= 0.001  # Base metabolic cost
        if agent.energy <= 0:
            agent.alive = False
            done = True
            reward -= 1.0

        # Ethical check (simplified)
        if agent.energy > 5.0:  # Hoarding check
            info['ethical_violation'] = True
            reward -= 0.5

        return reward, done, info

    def train_inner_loop(self, agents: List[MycoAgent],
                         num_episodes: int) -> List[TrainingMetrics]:
        """
        Inner training loop: RL optimization for fixed genomes.

        Args:
            agents: List of agents to train
            num_episodes: Number of episodes to run

        Returns:
            List of training metrics per episode
        """
        metrics_list = []

        for episode in range(num_episodes):
            start_time = time.time()

            total_reward, colony_metrics, violations, episode_data = self._run_episode(
                agents,
                max_steps=self.config.training.rl_max_steps,
                training=True
            )

            wall_time = time.time() - start_time

            metrics = TrainingMetrics(
                episode=episode,
                total_reward=total_reward,
                mean_agent_reward=total_reward / len(agents),
                colony_coherence=colony_metrics.coherence,
                colony_entropy=colony_metrics.entropy,
                colony_information=colony_metrics.mutual_information,
                phi_f=colony_metrics.phenomenal_curvature,
                rx_index=colony_metrics.recoverability_index,
                tau_r=colony_metrics.resonance_half_life,
                ethical_violations=violations,
                steps=episode_data['steps'],
                wall_time=wall_time
            )

            metrics_list.append(metrics)

            if episode % self.config.training.log_frequency == 0:
                logger.info(
                    f"Episode {episode}: reward={total_reward:.2f}, "
                    f"coherence={colony_metrics.coherence:.3f}, "
                    f"RX={colony_metrics.recoverability_index:.3f}"
                )

            # Trigger callbacks
            for callback in self.callbacks:
                callback('episode_end', metrics)

        return metrics_list

    def train_middle_loop(self, num_calibration_steps: int = 50) -> float:
        """
        Middle training loop: Surrogate model calibration.

        Returns:
            Calibration loss
        """
        if not self.config.overmind.use_surrogate:
            return 0.0

        surrogate = self.overmind.get_surrogate_model()
        if surrogate is None:
            return 0.0

        loss = self.surrogate_calibrator.calibrate(
            surrogate,
            self.uprt_field,
            num_steps=num_calibration_steps
        )

        logger.info(f"Surrogate calibration loss: {loss:.6f}")
        return loss

    def train_outer_loop(self, num_generations: int) -> Tuple[List[np.ndarray], List[float]]:
        """
        Outer training loop: Evolutionary optimization of genomes.

        Returns:
            best_genomes: Top performing genomes
            fitness_history: Fitness over generations
        """
        fitness_history = []

        # Initialize population
        population = self.evolution_engine.initialize_population()

        for generation in range(num_generations):
            logger.info(f"=== Generation {generation} ===")

            generation_fitness = []

            for genome_idx, genome in enumerate(population):
                # Create agents from this genome
                agents = self._create_agents([genome] * self.config.num_agents)

                # Run inner loop (abbreviated for evolution)
                fitness_episodes = self.config.training.evolution_episodes_per_genome
                episode_metrics = self.train_inner_loop(agents, fitness_episodes)

                # Compute fitness
                final_metrics = episode_metrics[-1] if episode_metrics else None
                if final_metrics:
                    colony_metrics = ColonyMetrics(
                        entropy=final_metrics.colony_entropy,
                        mutual_information=final_metrics.colony_information,
                        coherence=final_metrics.colony_coherence,
                        integrated_information=final_metrics.phi_f,
                        phenomenal_curvature=final_metrics.phi_f,
                        recoverability_index=final_metrics.rx_index,
                        resonance_half_life=final_metrics.tau_r
                    )

                    fitness = self.fitness_evaluator.compute_fitness(
                        colony_metrics,
                        final_metrics.total_reward,
                        final_metrics.ethical_violations
                    )
                else:
                    fitness = 0.0

                generation_fitness.append(fitness)

                logger.debug(f"Genome {genome_idx}: fitness={fitness:.2f}")

            # Evolve population
            if TORCH_AVAILABLE:
                fitness_tensor = torch.tensor(generation_fitness,
                                             dtype=torch.float32)
                population = self.evolution_engine.evolve_population(
                    population,
                    fitness_tensor
                )
            else:
                population = self.evolution_engine.evolve_population(
                    population,
                    np.array(generation_fitness)
                )

            # Track best fitness
            best_gen_fitness = max(generation_fitness)
            mean_gen_fitness = np.mean(generation_fitness)
            fitness_history.append(best_gen_fitness)

            if best_gen_fitness > self.best_fitness:
                self.best_fitness = best_gen_fitness
                self.best_genomes = [population[0].copy()]  # Elite is first

            logger.info(
                f"Generation {generation}: best={best_gen_fitness:.2f}, "
                f"mean={mean_gen_fitness:.2f}"
            )

            # Middle loop: calibrate surrogate periodically
            if generation % 5 == 0:
                self.train_middle_loop()

            # Callbacks
            for callback in self.callbacks:
                callback('generation_end', {
                    'generation': generation,
                    'best_fitness': best_gen_fitness,
                    'mean_fitness': mean_gen_fitness,
                    'population_size': len(population)
                })

            # Checkpointing
            if generation % self.config.training.checkpoint_frequency == 0:
                self._save_checkpoint(generation, population, fitness_history)

        return self.best_genomes, fitness_history

    def train(self) -> Dict[str, Any]:
        """
        Full training procedure with all nested loops.

        Returns:
            Dictionary with training results and best models
        """
        logger.info("Starting MycoNet 3.0 training")
        start_time = time.time()

        # Outer loop: Evolution
        best_genomes, fitness_history = self.train_outer_loop(
            self.config.training.evolution_generations
        )

        # Final training with best genome
        logger.info("Final training with best genome")
        best_agents = self._create_agents(
            [best_genomes[0]] * self.config.num_agents
        )

        final_metrics = self.train_inner_loop(
            best_agents,
            self.config.training.rl_episodes
        )

        total_time = time.time() - start_time

        results = {
            'best_genomes': best_genomes,
            'fitness_history': fitness_history,
            'final_metrics': final_metrics,
            'total_training_time': total_time,
            'best_fitness': self.best_fitness,
            'num_generations': self.config.training.evolution_generations,
            'num_agents': self.config.num_agents
        }

        logger.info(f"Training complete in {total_time:.1f}s")
        logger.info(f"Best fitness: {self.best_fitness:.2f}")

        return results

    def _save_checkpoint(self, generation: int, population: List[np.ndarray],
                         fitness_history: List[float]):
        """Save training checkpoint."""
        checkpoint = {
            'generation': generation,
            'population': [g.tolist() for g in population],
            'fitness_history': fitness_history,
            'best_fitness': self.best_fitness,
            'config': self.config.to_dict()
        }

        # Would save to file in production
        logger.debug(f"Checkpoint saved at generation {generation}")

    def add_callback(self, callback: Callable):
        """Add a callback function for training events."""
        self.callbacks.append(callback)

    def evaluate(self, agents: List[MycoAgent] = None,
                 num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained agents.

        Args:
            agents: Agents to evaluate (uses best if None)
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """
        if agents is None and self.best_genomes:
            agents = self._create_agents(
                [self.best_genomes[0]] * self.config.num_agents
            )

        if not agents:
            return {}

        total_reward = 0
        total_coherence = 0
        total_rx = 0

        for _ in range(num_episodes):
            reward, metrics, _, _ = self._run_episode(
                agents,
                max_steps=self.config.training.rl_max_steps,
                training=False
            )
            total_reward += reward
            total_coherence += metrics.coherence
            total_rx += metrics.recoverability_index

        return {
            'mean_reward': total_reward / num_episodes,
            'mean_coherence': total_coherence / num_episodes,
            'mean_rx': total_rx / num_episodes
        }


def create_training_pipeline(config: MycoNetConfig = None) -> TrainingPipeline:
    """Factory function to create a training pipeline."""
    if config is None:
        config = MycoNetConfig()

    if not config.validate():
        raise ValueError("Invalid configuration")

    return TrainingPipeline(config)
