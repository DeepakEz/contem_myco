"""
CommNet: Learning Multiagent Communication with Backpropagation
===============================================================
Sukhbaatar et al., 2016

Agents communicate by averaging hidden states across all agents.
Simple but effective baseline for learned communication.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class CommNetLayer(nn.Module):
    """
    Single communication layer.

    Each agent's hidden state is updated by:
    h_i' = tanh(W1 * h_i + W2 * avg(h_j for j != i))
    """

    def __init__(self, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        # Transform own hidden state
        self.W1 = nn.Linear(hidden_size, hidden_size)

        # Transform communication (average of others)
        self.W2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states (batch, num_agents, hidden_size)

        Returns:
            Updated hidden states (batch, num_agents, hidden_size)
        """
        batch_size, num_agents, _ = h.shape

        # Compute average of other agents' hidden states
        # Sum all, subtract self, divide by (n-1)
        h_sum = h.sum(dim=1, keepdim=True)  # (batch, 1, hidden)
        h_others_avg = (h_sum - h) / max(num_agents - 1, 1)  # (batch, agents, hidden)

        # Update hidden state
        h_new = torch.tanh(self.W1(h) + self.W2(h_others_avg))

        return h_new


class CommNet(nn.Module):
    """
    Full CommNet architecture.

    Observation -> Encoder -> [CommLayer] x K -> Actor/Critic heads
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 128,
        num_comm_layers: int = 2,
        continuous: bool = False,
    ):
        super().__init__()

        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_comm_layers = num_comm_layers
        self.continuous = continuous

        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Communication layers
        self.comm_layers = nn.ModuleList([
            CommNetLayer(hidden_size) for _ in range(num_comm_layers)
        ])

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_size, action_size)
            self.actor_logstd = nn.Parameter(torch.zeros(action_size))
        else:
            self.actor = nn.Linear(hidden_size, action_size)

        # Critic head
        self.critic = nn.Linear(hidden_size, 1)

    def forward(
        self,
        obs: torch.Tensor,
        return_comm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with communication.

        Args:
            obs: Observations (batch, num_agents, obs_size)
            return_comm: Whether to return communication vectors

        Returns:
            action_out: Action logits/means (batch, num_agents, action_size)
            values: Value estimates (batch, num_agents, 1)
            comm: Communication vectors if requested
        """
        batch_size, num_agents, _ = obs.shape

        # Encode observations
        h = self.encoder(obs)  # (batch, agents, hidden)

        # Communication rounds
        comm_history = [h] if return_comm else None
        for comm_layer in self.comm_layers:
            h = comm_layer(h)
            if return_comm:
                comm_history.append(h)

        # Actor output
        if self.continuous:
            action_out = self.actor_mean(h)
        else:
            action_out = self.actor(h)

        # Critic output
        values = self.critic(h)

        comm_out = torch.stack(comm_history, dim=2) if return_comm else None

        return action_out, values, comm_out

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions, log_probs, entropy, and values for all agents.

        Args:
            obs: (batch, num_agents, obs_size)
            action: Optional actions (batch, num_agents)
            deterministic: Use deterministic policy

        Returns:
            actions, log_probs, entropy, values - all (batch, num_agents)
        """
        action_out, values, _ = self.forward(obs)

        batch_size, num_agents, _ = obs.shape

        if self.continuous:
            std = torch.exp(self.actor_logstd)
            dist = torch.distributions.Normal(action_out, std)

            if action is None:
                if deterministic:
                    action = action_out
                else:
                    action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=action_out)

            if action is None:
                if deterministic:
                    action = action_out.argmax(dim=-1)
                else:
                    action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, values.squeeze(-1)


class CommNetAgent:
    """
    Wrapper for using CommNet with environment interface.
    """

    def __init__(
        self,
        num_agents: int,
        obs_size: int,
        action_size: int,
        hidden_size: int = 128,
        num_comm_layers: int = 2,
        continuous: bool = False,
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.device = torch.device(device)

        self.network = CommNet(
            obs_size=obs_size,
            action_size=action_size,
            hidden_size=hidden_size,
            num_comm_layers=num_comm_layers,
            continuous=continuous,
        ).to(self.device)

    def act(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        """
        Get actions for all agents.

        Args:
            observations: Dict of agent_id -> observation

        Returns:
            actions, values, log_probs as dicts
        """
        agent_ids = list(observations.keys())

        # Stack observations
        obs_list = [observations[aid] for aid in agent_ids]
        obs_tensor = torch.tensor(
            np.array(obs_list), dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, num_agents, obs_size)

        with torch.no_grad():
            actions, log_probs, _, values = self.network.get_action_and_value(
                obs_tensor, deterministic=deterministic
            )

        # Convert to dicts
        actions_np = actions.squeeze(0).cpu().numpy()
        values_np = values.squeeze(0).cpu().numpy()
        log_probs_np = log_probs.squeeze(0).cpu().numpy()

        action_dict = {aid: int(actions_np[i]) for i, aid in enumerate(agent_ids)}
        value_dict = {aid: float(values_np[i]) for i, aid in enumerate(agent_ids)}
        log_prob_dict = {aid: float(log_probs_np[i]) for i, aid in enumerate(agent_ids)}

        return action_dict, value_dict, log_prob_dict

    def parameters(self):
        return self.network.parameters()

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict)

    def to(self, device):
        self.device = torch.device(device)
        self.network.to(self.device)
        return self
