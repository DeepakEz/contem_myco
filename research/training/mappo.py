"""
MAPPO Training Loop
===================
Multi-Agent PPO with Contemplative modules integration.

Features:
- Centralized critic with decentralized actors (CTDE)
- Learning rate scheduling (cosine annealing)
- Mixed precision training (torch.cuda.amp)
- Full contemplative module integration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
import logging

# Mixed precision support
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

# torch.compile support (PyTorch 2.0+)
COMPILE_AVAILABLE = hasattr(torch, 'compile')

logger = logging.getLogger(__name__)


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    observations: List[Dict[str, np.ndarray]] = field(default_factory=list)
    next_observations: List[Dict[str, np.ndarray]] = field(default_factory=list)
    actions: List[Dict[str, int]] = field(default_factory=list)
    rewards: List[Dict[str, float]] = field(default_factory=list)
    values: List[Dict[str, float]] = field(default_factory=list)
    log_probs: List[Dict[str, float]] = field(default_factory=list)
    dones: List[Dict[str, bool]] = field(default_factory=list)
    positions: List[Dict[str, np.ndarray]] = field(default_factory=list)
    diffusion_obs: List[Dict[str, np.ndarray]] = field(default_factory=list)

    # Ethics tracking
    ethics_scores: List[Dict[str, float]] = field(default_factory=list)
    constraint_violations: List[Dict[str, float]] = field(default_factory=list)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: Dict[str, int],
        reward: Dict[str, float],
        value: Dict[str, float],
        log_prob: Dict[str, float],
        done: Dict[str, bool],
        next_obs: Optional[Dict[str, np.ndarray]] = None,
        position: Optional[Dict[str, np.ndarray]] = None,
        diffusion_obs: Optional[Dict[str, np.ndarray]] = None,
        ethics_score: Optional[Dict[str, float]] = None,
    ):
        """Add a transition to the buffer."""
        self.observations.append(obs)
        self.next_observations.append(next_obs or {})
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.positions.append(position or {})
        self.diffusion_obs.append(diffusion_obs or {})
        self.ethics_scores.append(ethics_score or {})

    def clear(self):
        """Clear the buffer."""
        self.observations.clear()
        self.next_observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.positions.clear()
        self.diffusion_obs.clear()
        self.ethics_scores.clear()
        self.constraint_violations.clear()

    def __len__(self):
        return len(self.observations)


@dataclass
class TrainingStats:
    """Statistics from training."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropy_losses: List[float] = field(default_factory=list)
    ethics_losses: List[float] = field(default_factory=list)
    mindfulness_losses: List[float] = field(default_factory=list)

    # Metrics
    social_welfare: List[float] = field(default_factory=list)
    gini_coefficients: List[float] = field(default_factory=list)
    cooperation_rates: List[float] = field(default_factory=list)
    constraint_violations: List[float] = field(default_factory=list)


class MAPPOTrainer:
    """
    Multi-Agent PPO trainer with Contemplative modules.

    Features:
    - Centralized Training with Decentralized Execution (CTDE)
    - Learning rate scheduling (cosine annealing)
    - Mixed precision training for GPU acceleration
    - torch.compile() optimization (PyTorch 2.0+)
    """

    def __init__(
        self,
        env,
        agent_system,
        config,
        device: str = "cpu",
        log_dir: Optional[str] = None,
        ethics_coef: float = 0.1,
        mindfulness_coef: float = 0.1,
        use_lr_scheduler: bool = True,
        use_mixed_precision: bool = True,
        use_compile: bool = True,
    ):
        self.env = env
        self.agent_system = agent_system
        self.config = config
        self.device = torch.device(device)
        self.log_dir = Path(log_dir) if log_dir else Path("runs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Loss coefficients for contemplative modules
        self.ethics_coef = ethics_coef
        self.mindfulness_coef = mindfulness_coef

        # Move agents to device
        if hasattr(agent_system, 'shared_agent'):
            agent_system.shared_agent.to(self.device)
        else:
            for agent in agent_system.agents:
                agent.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            agent_system.parameters(),
            lr=config.training.learning_rate,
        )

        # Learning rate scheduler (cosine annealing)
        self.use_lr_scheduler = use_lr_scheduler
        self.scheduler = None
        if use_lr_scheduler:
            total_updates = config.training.total_timesteps // config.training.n_steps
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_updates,
                eta_min=config.training.learning_rate * 0.1,  # 10% of initial LR
            )
            logger.info(f"Using cosine LR scheduling over {total_updates} updates")

        # Mixed precision training
        self.use_mixed_precision = use_mixed_precision and AMP_AVAILABLE and device != "cpu"
        self.scaler = None
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")

        # torch.compile optimization (PyTorch 2.0+)
        if use_compile and COMPILE_AVAILABLE and device != "cpu":
            try:
                agent = agent_system.shared_agent if hasattr(agent_system, 'shared_agent') else agent_system.agents[0]
                agent.ac = torch.compile(agent.ac, mode='reduce-overhead')
                logger.info("torch.compile() optimization enabled")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}")

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training stats
        self.stats = TrainingStats()
        self.global_step = 0
        self.episode_count = 0

        # Logging
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            logger.warning("TensorBoard not available")

    def collect_rollout(self, n_steps: int) -> Dict[str, Any]:
        """
        Collect rollout data for n_steps.
        """
        self.buffer.clear()

        obs, _ = self.env.reset()
        positions = self.env.get_positions() if hasattr(self.env, 'get_positions') else {}
        self.agent_system.reset()

        episode_reward = 0
        episode_length = 0
        episode_metrics = {}

        for step in range(n_steps):
            # Get actions from agent system
            with torch.no_grad():
                outputs = self.agent_system.step(obs, positions, deterministic=False)

            # Extract actions, values, log_probs
            actions = {agent: int(out.action.item() if np.isscalar(out.action) or out.action.size == 1 else out.action[0])
                      for agent, out in outputs.items()}
            values = {agent: out.value.item() for agent, out in outputs.items()}
            log_probs = {agent: out.log_prob.item() for agent, out in outputs.items()}
            ethics_scores = {agent: out.ethics_score.item() if out.ethics_score is not None else 0.0
                           for agent, out in outputs.items()}

            # Get diffusion observations
            diffusion_obs = {}
            if self.config.diffusion.enabled:
                diffusion_obs_size = 3 * self.config.diffusion.num_channels
                for agent in obs.keys():
                    pos = positions.get(agent)
                    if self.agent_system.diffusion_field is not None and pos is not None:
                        diffusion_obs[agent] = self.agent_system.diffusion_field.sense(pos)
                    else:
                        # No position available - use zeros to match expected input size
                        diffusion_obs[agent] = np.zeros(diffusion_obs_size, dtype=np.float32)

            # Step environment
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

            # Update positions
            positions = self.env.get_positions() if hasattr(self.env, 'get_positions') else {}

            # Compute dones
            dones = {agent: terminations.get(agent, False) or truncations.get(agent, False)
                    for agent in obs.keys()}

            # Add to buffer
            self.buffer.add(
                obs=obs,
                action=actions,
                reward=rewards,
                value=values,
                log_prob=log_probs,
                done=dones,
                next_obs=next_obs,
                position=positions,
                diffusion_obs=diffusion_obs,
                ethics_score=ethics_scores,
            )

            # Track episode stats
            episode_reward += sum(rewards.values())
            episode_length += 1
            self.global_step += 1

            # Check episode end
            if all(dones.values()):
                # Log episode
                self.stats.episode_rewards.append(episode_reward)
                self.stats.episode_lengths.append(episode_length)
                self.episode_count += 1

                # Get metrics
                if hasattr(self.env, 'get_metrics'):
                    metrics = self.env.get_metrics()
                    self.stats.social_welfare.append(metrics.social_welfare)
                    self.stats.gini_coefficients.append(metrics.gini_coefficient)
                    self.stats.cooperation_rates.append(metrics.cooperation_rate)
                    episode_metrics = {
                        'social_welfare': metrics.social_welfare,
                        'gini': metrics.gini_coefficient,
                        'cooperation': metrics.cooperation_rate,
                    }

                # Reset
                obs, _ = self.env.reset()
                positions = self.env.get_positions() if hasattr(self.env, 'get_positions') else {}
                self.agent_system.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs

        return {
            'steps': n_steps,
            'episodes': self.episode_count,
            'mean_reward': np.mean(self.stats.episode_rewards[-10:]) if self.stats.episode_rewards else 0,
            **episode_metrics,
        }

    def compute_gae(
        self,
        rewards: List[Dict[str, float]],
        values: List[Dict[str, float]],
        dones: List[Dict[str, bool]],
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Compute Generalized Advantage Estimation.
        """
        agents = list(rewards[0].keys())
        n_steps = len(rewards)

        advantages = {agent: np.zeros(n_steps) for agent in agents}
        returns = {agent: np.zeros(n_steps) for agent in agents}

        for agent in agents:
            last_gae = 0
            last_value = values[-1][agent]

            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    next_value = last_value
                else:
                    next_value = values[t + 1][agent]

                delta = (
                    rewards[t][agent] +
                    gamma * next_value * (1 - dones[t][agent]) -
                    values[t][agent]
                )
                advantages[agent][t] = last_gae = (
                    delta + gamma * gae_lambda * (1 - dones[t][agent]) * last_gae
                )
                returns[agent][t] = advantages[agent][t] + values[t][agent]

        return advantages, returns

    def update(self) -> Dict[str, float]:
        """
        Update policy using collected rollout.

        Supports:
        - Mixed precision training (AMP)
        - Centralized critic with global state
        - Learning rate scheduling
        """
        config = self.config.training

        # Compute advantages
        advantages, returns = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            config.gamma,
            config.gae_lambda,
        )

        # Flatten data for all agents
        agents = list(self.buffer.observations[0].keys())
        n_steps = len(self.buffer)
        n_agents = len(agents)

        all_obs = []
        all_next_obs = []
        all_actions = []
        all_actions_onehot = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        all_diffusion_obs = []
        all_rewards = []
        all_global_states = []

        # Get action size for one-hot encoding
        agent_ref = self.agent_system.shared_agent if hasattr(self.agent_system, 'shared_agent') else self.agent_system.agents[0]
        action_size = agent_ref.action_size

        for t in range(n_steps):
            # Build global state (concatenated observations from all agents)
            global_state = np.concatenate([self.buffer.observations[t][a] for a in agents])

            for agent in agents:
                all_obs.append(self.buffer.observations[t][agent])
                all_actions.append(self.buffer.actions[t][agent])
                all_old_log_probs.append(self.buffer.log_probs[t][agent])
                all_advantages.append(advantages[agent][t])
                all_returns.append(returns[agent][t])
                all_global_states.append(global_state)

                # One-hot encode actions for ethics module
                action_onehot = np.zeros(action_size)
                action_onehot[self.buffer.actions[t][agent]] = 1.0
                all_actions_onehot.append(action_onehot)

                # Next observations for mindfulness prediction
                if self.buffer.next_observations[t].get(agent) is not None:
                    all_next_obs.append(self.buffer.next_observations[t][agent])
                else:
                    all_next_obs.append(self.buffer.observations[t][agent])

                # Rewards for computing fairness
                all_rewards.append(self.buffer.rewards[t][agent])

                if self.buffer.diffusion_obs[t].get(agent) is not None:
                    all_diffusion_obs.append(self.buffer.diffusion_obs[t][agent])

        # Convert to tensors
        obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.tensor(np.array(all_next_obs), dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        actions_onehot_tensor = torch.tensor(np.array(all_actions_onehot), dtype=torch.float32, device=self.device)
        old_log_probs_tensor = torch.tensor(all_old_log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)
        global_state_tensor = torch.tensor(np.array(all_global_states), dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Diffusion observations
        diffusion_tensor = None
        agent_ref = self.agent_system.shared_agent if hasattr(self.agent_system, 'shared_agent') else self.agent_system.agents[0]
        if all_diffusion_obs and len(all_diffusion_obs) == n_samples:
            diffusion_tensor = torch.tensor(np.array(all_diffusion_obs), dtype=torch.float32, device=self.device)
        elif self.config.diffusion.enabled:
            # Diffusion is enabled but no observations collected (positions unavailable)
            # Provide zeros to match expected input size
            diffusion_obs_size = 3 * self.config.diffusion.num_channels
            diffusion_tensor = torch.zeros(n_samples, diffusion_obs_size, dtype=torch.float32, device=self.device)

        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_ethics_loss = 0
        total_mindfulness_loss = 0

        batch_size = config.batch_size
        n_samples = len(all_obs)

        # Get agent reference
        agent = self.agent_system.shared_agent if hasattr(self.agent_system, 'shared_agent') else self.agent_system.agents[0]

        # Check if centralized critic is available
        use_centralized = agent.ac.centralized_critic is not None

        for epoch in range(config.n_epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_obs = obs_tensor[batch_indices]
                batch_next_obs = next_obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_actions_onehot = actions_onehot_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_rewards = rewards_tensor[batch_indices]
                batch_diffusion = diffusion_tensor[batch_indices] if diffusion_tensor is not None else None
                batch_global_state = global_state_tensor[batch_indices] if use_centralized else None

                # Mixed precision context
                amp_context = autocast() if self.use_mixed_precision else torch.enable_grad()

                with amp_context:
                    _, log_probs, entropy, values = agent.ac.get_action_and_value(
                        batch_obs,
                        batch_diffusion,
                        batch_actions,
                        global_state=batch_global_state,
                    )

                    # Policy loss (clipped)
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = nn.functional.mse_loss(values, batch_returns)

                    # Entropy loss
                    entropy_loss = -entropy.mean()

                    # Ethics loss (Module A)
                    ethics_loss = torch.tensor(0.0, device=self.device)
                    if agent.ethics is not None and self.ethics_coef > 0:
                        ethics_score, _ = agent.ethics(batch_obs, batch_actions_onehot)
                        # Encourage high ethics scores (negative loss when score is high)
                        ethics_loss = -ethics_score.mean()

                        # Compute constraint penalties based on rewards (fairness)
                        # Reshape rewards to compute Gini per batch segment
                        batch_len = len(batch_rewards)
                        if batch_len >= n_agents:
                            reward_chunks = batch_rewards[:batch_len - (batch_len % n_agents)].view(-1, n_agents)
                            # Compute Gini for each chunk
                            gini_losses = []
                            for chunk in reward_chunks:
                                sorted_r, _ = torch.sort(chunk)
                                n = len(sorted_r)
                                gini_indices = torch.arange(1, n + 1, dtype=torch.float32, device=self.device)
                                gini = (2 * (gini_indices * sorted_r).sum()) / (n * sorted_r.sum() + 1e-8) - (n + 1) / n
                                gini_losses.append(torch.relu(gini - 0.3))  # Penalize Gini > 0.3
                            if gini_losses:
                                ethics_loss = ethics_loss + torch.stack(gini_losses).mean()

                    # Mindfulness loss (Module C) - train world model
                    mindfulness_loss = torch.tensor(0.0, device=self.device)
                    if agent.mindfulness is not None and self.mindfulness_coef > 0:
                        # Embed discrete actions for prediction model
                        action_embed = batch_actions_onehot
                        mindfulness_loss = agent.mindfulness.compute_prediction_loss(
                            batch_obs, action_embed, batch_next_obs
                        )

                    # Total loss
                    loss = (
                        policy_loss +
                        config.vf_coef * value_loss +
                        config.ent_coef * entropy_loss +
                        self.ethics_coef * ethics_loss +
                        self.mindfulness_coef * mindfulness_loss
                    )

                # Optimize (with mixed precision if enabled)
                self.optimizer.zero_grad()
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_ethics_loss += ethics_loss.item()
                total_mindfulness_loss += mindfulness_loss.item()

        # Step LR scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        n_updates = config.n_epochs * (n_samples // batch_size + 1)
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy_loss = total_entropy_loss / n_updates
        avg_ethics_loss = total_ethics_loss / n_updates
        avg_mindfulness_loss = total_mindfulness_loss / n_updates

        self.stats.policy_losses.append(avg_policy_loss)
        self.stats.value_losses.append(avg_value_loss)
        self.stats.entropy_losses.append(avg_entropy_loss)
        self.stats.ethics_losses.append(avg_ethics_loss)
        self.stats.mindfulness_losses.append(avg_mindfulness_loss)

        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'ethics_loss': avg_ethics_loss,
            'mindfulness_loss': avg_mindfulness_loss,
            'learning_rate': current_lr,
        }

    def train(self, total_timesteps: int, log_interval: int = 10, save_interval: int = 100000):
        """
        Main training loop.
        """
        logger.info(f"Starting training for {total_timesteps} timesteps")
        start_time = time.time()

        n_steps = self.config.training.n_steps
        iteration = 0

        while self.global_step < total_timesteps:
            iteration += 1

            # Collect rollout
            rollout_info = self.collect_rollout(n_steps)

            # Update policy
            update_info = self.update()

            # Logging
            if iteration % log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.global_step / elapsed

                logger.info(
                    f"Step {self.global_step}/{total_timesteps} | "
                    f"Episodes: {self.episode_count} | "
                    f"Mean Reward: {rollout_info['mean_reward']:.2f} | "
                    f"Policy Loss: {update_info['policy_loss']:.4f} | "
                    f"FPS: {fps:.0f}"
                )

                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('train/reward', rollout_info['mean_reward'], self.global_step)
                    self.writer.add_scalar('train/policy_loss', update_info['policy_loss'], self.global_step)
                    self.writer.add_scalar('train/value_loss', update_info['value_loss'], self.global_step)
                    self.writer.add_scalar('train/entropy', -update_info['entropy_loss'], self.global_step)
                    self.writer.add_scalar('train/ethics_loss', update_info['ethics_loss'], self.global_step)
                    self.writer.add_scalar('train/mindfulness_loss', update_info['mindfulness_loss'], self.global_step)

                    if 'social_welfare' in rollout_info:
                        self.writer.add_scalar('metrics/social_welfare', rollout_info['social_welfare'], self.global_step)
                    if 'gini' in rollout_info:
                        self.writer.add_scalar('metrics/gini', rollout_info['gini'], self.global_step)
                    if 'cooperation' in rollout_info:
                        self.writer.add_scalar('metrics/cooperation', rollout_info['cooperation'], self.global_step)

            # Save checkpoint
            if self.global_step % save_interval == 0:
                self.save(f"checkpoint_{self.global_step}")

        # Final save
        self.save("final")
        logger.info(f"Training complete. Total time: {time.time() - start_time:.1f}s")

        return self.stats

    def save(self, name: str):
        """Save checkpoint."""
        path = self.log_dir / f"{name}.pt"
        torch.save({
            'agent_state_dict': self.agent_system.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'config': self.config.to_dict(),
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent_system.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.episode_count = checkpoint['episode_count']
        logger.info(f"Loaded checkpoint from {path}")

    def evaluate(self, n_episodes: int = 20, deterministic: bool = True) -> Dict[str, float]:
        """Evaluate current policy."""
        episode_rewards = []
        episode_metrics = []

        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            positions = self.env.get_positions() if hasattr(self.env, 'get_positions') else {}
            self.agent_system.reset()

            episode_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    outputs = self.agent_system.step(obs, positions, deterministic=deterministic)

                actions = {agent: int(out.action.item() if np.isscalar(out.action) or out.action.size == 1 else out.action[0])
                          for agent, out in outputs.items()}

                obs, rewards, terminations, truncations, _ = self.env.step(actions)
                positions = self.env.get_positions() if hasattr(self.env, 'get_positions') else {}

                episode_reward += sum(rewards.values())
                done = all(terminations.get(a, False) or truncations.get(a, False) for a in obs.keys())

            episode_rewards.append(episode_reward)

            if hasattr(self.env, 'get_metrics'):
                metrics = self.env.get_metrics()
                episode_metrics.append({
                    'social_welfare': metrics.social_welfare,
                    'gini': metrics.gini_coefficient,
                    'cooperation': metrics.cooperation_rate,
                })

        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
        }

        if episode_metrics:
            results['mean_social_welfare'] = np.mean([m['social_welfare'] for m in episode_metrics])
            results['mean_gini'] = np.mean([m['gini'] for m in episode_metrics])
            results['mean_cooperation'] = np.mean([m['cooperation'] for m in episode_metrics])

        return results
