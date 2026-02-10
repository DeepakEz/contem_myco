"""
MADDPG Baseline
================
Multi-Agent Deep Deterministic Policy Gradient.

Reference: Lowe et al., 2017 - "Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments"

Key features:
- Centralized critic (observes all agents' observations and actions)
- Decentralized actors (each agent only observes its own state)
- Continuous action space (adapted for discrete via Gumbel-Softmax)
- Experience replay with multi-agent transitions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


class MADDPGActor(nn.Module):
    """
    Actor network for a single MADDPG agent.
    Maps local observation to action.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 128,
        continuous: bool = False,
    ):
        super().__init__()

        self.continuous = continuous
        self.action_size = action_size

        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute action from observation.

        For continuous: returns action values (use tanh for bounded)
        For discrete: returns logits (use Gumbel-Softmax for differentiable sampling)
        """
        x = self.network(obs)

        if self.continuous:
            return torch.tanh(x)
        else:
            return x  # Logits for Gumbel-Softmax

    def get_action(
        self,
        obs: torch.Tensor,
        explore: bool = True,
        noise_std: float = 0.1,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action with exploration.

        Returns:
            action: Selected action
            action_onehot: One-hot or continuous action for critic input
        """
        logits = self.forward(obs)

        if self.continuous:
            action = logits
            if explore:
                action = action + torch.randn_like(action) * noise_std
                action = torch.clamp(action, -1, 1)
            return action, action
        else:
            if explore:
                # Gumbel-Softmax for differentiable discrete actions
                action_onehot = F.gumbel_softmax(logits, tau=temperature, hard=True)
                action = action_onehot.argmax(dim=-1)
            else:
                action = logits.argmax(dim=-1)
                action_onehot = F.one_hot(action, self.action_size).float()
            return action, action_onehot


class MADDPGCritic(nn.Module):
    """
    Centralized critic for MADDPG.
    Takes all agents' observations and actions as input.
    """

    def __init__(
        self,
        total_obs_size: int,
        total_action_size: int,
        hidden_size: int = 256,
    ):
        super().__init__()

        input_size = total_obs_size + total_action_size

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        all_obs: torch.Tensor,
        all_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q-value from global information.

        Args:
            all_obs: Concatenated observations from all agents (batch, total_obs_size)
            all_actions: Concatenated actions from all agents (batch, total_action_size)

        Returns:
            q_value: (batch, 1)
        """
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.network(x)


class MADDPGReplayBuffer:
    """Multi-agent replay buffer storing transitions for all agents."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        dones: Dict[str, bool],
    ):
        """Store a multi-agent transition."""
        self.buffer.append({
            "obs": {k: v.copy() for k, v in obs.items()},
            "actions": {k: np.array(v).copy() for k, v in actions.items()},
            "rewards": dict(rewards),
            "next_obs": {k: v.copy() for k, v in next_obs.items()},
            "dones": dict(dones),
        })

    def sample(self, batch_size: int) -> List[Dict]:
        """Sample a batch of transitions."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class MADDPGAgent:
    """
    Multi-Agent DDPG system managing all agents.

    Centralized training with decentralized execution.
    """

    def __init__(
        self,
        num_agents: int,
        obs_size: int,
        action_size: int,
        hidden_size: int = 128,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        batch_size: int = 256,
        buffer_size: int = 100000,
        update_every: int = 100,
        continuous: bool = False,
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.device = device
        self.continuous = continuous
        self._step_count = 0

        total_obs_size = obs_size * num_agents
        total_action_size = action_size * num_agents

        # Create actor and critic for each agent
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []

        for i in range(num_agents):
            # Actor
            actor = MADDPGActor(obs_size, action_size, hidden_size, continuous).to(device)
            target_actor = MADDPGActor(obs_size, action_size, hidden_size, continuous).to(device)
            target_actor.load_state_dict(actor.state_dict())

            # Critic
            critic = MADDPGCritic(total_obs_size, total_action_size, hidden_size * 2).to(device)
            target_critic = MADDPGCritic(total_obs_size, total_action_size, hidden_size * 2).to(device)
            target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.critics.append(critic)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=actor_lr))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=critic_lr))

        # Shared replay buffer
        self.replay_buffer = MADDPGReplayBuffer(buffer_size)

        self.update_count = 0

    def act(
        self,
        obs: Dict[str, np.ndarray],
        explore: bool = True,
    ) -> Tuple[Dict[str, int], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get actions for all agents.

        Returns:
            actions: Dict of agent_id -> action
            values: Dict of agent_id -> value (empty for MADDPG)
            log_probs: Dict of agent_id -> log_prob (empty for MADDPG)
        """
        actions = {}
        agent_ids = sorted(obs.keys())

        for i, aid in enumerate(agent_ids):
            obs_tensor = torch.from_numpy(obs[aid]).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, _ = self.actors[i].get_action(obs_tensor, explore=explore)

            actions[aid] = int(action.item()) if not self.continuous else action.squeeze().cpu().numpy()

        return actions, {}, {}

    def store(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, any],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        dones: Dict[str, bool],
    ):
        """Store transition in replay buffer."""
        # Convert actions to numpy arrays
        action_arrays = {}
        for aid, a in actions.items():
            if isinstance(a, (int, np.integer)):
                action_arrays[aid] = np.eye(self.action_size)[a]
            else:
                action_arrays[aid] = np.array(a)

        self.replay_buffer.push(obs, action_arrays, rewards, next_obs, dones)

    def update(self) -> Optional[Dict[str, float]]:
        """
        Update all agents' actors and critics.

        Returns:
            Dict of loss values or None if buffer too small
        """
        self._step_count += 1
        if self._step_count % self.update_every != 0:
            return None
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        agent_ids = sorted(batch[0]["obs"].keys())

        losses = {}

        for i, aid in enumerate(agent_ids):
            # Prepare batch tensors
            obs_i = torch.stack([
                torch.from_numpy(t["obs"][aid]).float()
                for t in batch
            ]).to(self.device)

            next_obs_i = torch.stack([
                torch.from_numpy(t["next_obs"][aid]).float()
                for t in batch
            ]).to(self.device)

            rewards_i = torch.tensor(
                [t["rewards"][aid] for t in batch], dtype=torch.float32
            ).to(self.device)

            dones_i = torch.tensor(
                [float(t["dones"][aid]) for t in batch], dtype=torch.float32
            ).to(self.device)

            # All obs and actions for centralized critic
            all_obs = torch.cat([
                torch.stack([torch.from_numpy(t["obs"][a]).float() for t in batch]).to(self.device)
                for a in agent_ids
            ], dim=-1)

            all_next_obs = torch.cat([
                torch.stack([torch.from_numpy(t["next_obs"][a]).float() for t in batch]).to(self.device)
                for a in agent_ids
            ], dim=-1)

            all_actions = torch.cat([
                torch.stack([torch.from_numpy(t["actions"][a]).float() for t in batch]).to(self.device)
                for a in agent_ids
            ], dim=-1)

            # --- Critic update ---
            with torch.no_grad():
                # Get target actions from all target actors
                target_actions = []
                for j, a in enumerate(agent_ids):
                    next_o = torch.stack([
                        torch.from_numpy(t["next_obs"][a]).float() for t in batch
                    ]).to(self.device)
                    _, target_action_onehot = self.target_actors[j].get_action(
                        next_o, explore=False
                    )
                    target_actions.append(target_action_onehot)
                all_target_actions = torch.cat(target_actions, dim=-1)

                target_q = self.target_critics[i](all_next_obs, all_target_actions).squeeze(-1)
                target_value = rewards_i + self.gamma * (1 - dones_i) * target_q

            current_q = self.critics[i](all_obs, all_actions).squeeze(-1)
            critic_loss = F.mse_loss(current_q, target_value)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_optimizers[i].step()

            # --- Actor update ---
            # Get current agent's action with gradient
            _, actor_action_onehot = self.actors[i].get_action(obs_i, explore=False)

            # Replace agent i's action in all_actions
            new_all_actions_parts = []
            for j, a in enumerate(agent_ids):
                if j == i:
                    new_all_actions_parts.append(actor_action_onehot)
                else:
                    new_all_actions_parts.append(
                        torch.stack([
                            torch.from_numpy(t["actions"][a]).float() for t in batch
                        ]).to(self.device)
                    )
            new_all_actions = torch.cat(new_all_actions_parts, dim=-1)

            actor_loss = -self.critics[i](all_obs, new_all_actions).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()

            losses[f"critic_{i}"] = float(critic_loss.item())
            losses[f"actor_{i}"] = float(actor_loss.item())

        # Soft update targets
        self._soft_update()
        self.update_count += 1

        return losses

    def _soft_update(self):
        """Soft update target networks."""
        for i in range(self.num_agents):
            for target_param, param in zip(
                self.target_actors[i].parameters(), self.actors[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for target_param, param in zip(
                self.target_critics[i].parameters(), self.critics[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def parameters(self):
        """Get all parameters for saving."""
        params = []
        for actor in self.actors:
            params.extend(actor.parameters())
        for critic in self.critics:
            params.extend(critic.parameters())
        return params

    def state_dict(self):
        """Get state dict."""
        return {
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics],
            "target_actors": [a.state_dict() for a in self.target_actors],
            "target_critics": [c.state_dict() for c in self.target_critics],
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(state_dict["actors"][i])
            self.critics[i].load_state_dict(state_dict["critics"][i])
            self.target_actors[i].load_state_dict(state_dict["target_actors"][i])
            self.target_critics[i].load_state_dict(state_dict["target_critics"][i])
