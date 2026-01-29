"""
TarMAC: Targeted Multi-Agent Communication
==========================================
Das et al., 2019

Agents communicate using attention mechanism to selectively
attend to relevant messages from other agents.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention for inter-agent communication."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, num_agents, hidden_size)
            key: (batch, num_agents, hidden_size)
            value: (batch, num_agents, hidden_size)
            mask: Optional attention mask

        Returns:
            output: (batch, num_agents, hidden_size)
            attention_weights: (batch, num_heads, num_agents, num_agents)
        """
        batch_size, num_agents, _ = query.shape

        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, num_agents, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, num_agents, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, num_agents, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, agents, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Mask self-attention (agent shouldn't attend to itself)
        if mask is None:
            mask = torch.eye(num_agents, device=query.device).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, agents, agents)

        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        out = torch.matmul(attention_weights, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, num_agents, self.hidden_size)
        out = self.out_proj(out)

        return out, attention_weights


class TarMACLayer(nn.Module):
    """
    Single TarMAC communication layer.

    Combines:
    1. Message generation
    2. Attention-based message aggregation
    3. Message integration with hidden state
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Message generation
        self.message_gen = nn.Linear(hidden_size, hidden_size)

        # Attention for message aggregation
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)

        # Integrate messages with hidden state
        self.integrate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        h: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h: Hidden states (batch, num_agents, hidden_size)

        Returns:
            h_new: Updated hidden states
            attention: Attention weights if requested
        """
        # Generate messages
        messages = self.message_gen(h)

        # Attend to messages from other agents
        aggregated, attention_weights = self.attention(h, messages, messages)

        # Integrate with residual connection
        combined = torch.cat([h, aggregated], dim=-1)
        h_new = self.layer_norm(h + self.integrate(combined))

        return h_new, attention_weights if return_attention else None


class TarMAC(nn.Module):
    """
    Full TarMAC architecture.

    Observation -> Encoder -> [TarMACLayer] x K -> Actor/Critic
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 128,
        num_comm_layers: int = 2,
        num_heads: int = 4,
        continuous: bool = False,
    ):
        super().__init__()

        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.continuous = continuous

        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # TarMAC communication layers
        self.comm_layers = nn.ModuleList([
            TarMACLayer(hidden_size, num_heads) for _ in range(num_comm_layers)
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
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass with attention-based communication.

        Args:
            obs: (batch, num_agents, obs_size)

        Returns:
            action_out, values, attention_weights_list
        """
        # Encode observations
        h = self.encoder(obs)

        # Communication with attention
        attention_list = [] if return_attention else None
        for comm_layer in self.comm_layers:
            h, attn = comm_layer(h, return_attention=return_attention)
            if return_attention:
                attention_list.append(attn)

        # Actor output
        if self.continuous:
            action_out = self.actor_mean(h)
        else:
            action_out = self.actor(h)

        # Critic output
        values = self.critic(h)

        return action_out, values, attention_list

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions, log_probs, entropy, values."""
        action_out, values, _ = self.forward(obs)

        if self.continuous:
            std = torch.exp(self.actor_logstd)
            dist = torch.distributions.Normal(action_out, std)

            if action is None:
                action = action_out if deterministic else dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=action_out)

            if action is None:
                action = action_out.argmax(dim=-1) if deterministic else dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, values.squeeze(-1)


class TarMACAgent:
    """Wrapper for TarMAC with environment interface."""

    def __init__(
        self,
        num_agents: int,
        obs_size: int,
        action_size: int,
        hidden_size: int = 128,
        num_comm_layers: int = 2,
        num_heads: int = 4,
        continuous: bool = False,
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.device = torch.device(device)

        self.network = TarMAC(
            obs_size=obs_size,
            action_size=action_size,
            hidden_size=hidden_size,
            num_comm_layers=num_comm_layers,
            num_heads=num_heads,
            continuous=continuous,
        ).to(self.device)

    def act(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        """Get actions for all agents."""
        agent_ids = list(observations.keys())

        obs_list = [observations[aid] for aid in agent_ids]
        obs_tensor = torch.tensor(
            np.array(obs_list), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            actions, log_probs, _, values = self.network.get_action_and_value(
                obs_tensor, deterministic=deterministic
            )

        actions_np = actions.squeeze(0).cpu().numpy()
        values_np = values.squeeze(0).cpu().numpy()
        log_probs_np = log_probs.squeeze(0).cpu().numpy()

        return (
            {aid: int(actions_np[i]) for i, aid in enumerate(agent_ids)},
            {aid: float(values_np[i]) for i, aid in enumerate(agent_ids)},
            {aid: float(log_probs_np[i]) for i, aid in enumerate(agent_ids)},
        )

    def get_attention_weights(
        self,
        observations: Dict[str, np.ndarray],
    ) -> List[np.ndarray]:
        """Get attention weights for visualization."""
        agent_ids = list(observations.keys())

        obs_list = [observations[aid] for aid in agent_ids]
        obs_tensor = torch.tensor(
            np.array(obs_list), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            _, _, attention_list = self.network.forward(obs_tensor, return_attention=True)

        return [attn.squeeze(0).cpu().numpy() for attn in attention_list]

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
