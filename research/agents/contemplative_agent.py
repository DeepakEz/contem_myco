"""
Contemplative MARL Agent
========================
Unified agent integrating ethics, diffusion, and mindfulness modules
with MAPPO as the base RL algorithm.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .modules import (
    create_ethics_module,
    create_diffusion_module,
    create_mindfulness_module,
    DiffusionField,
)


@dataclass
class AgentOutput:
    """Output from agent forward pass."""
    action: np.ndarray
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    ethics_score: Optional[torch.Tensor] = None
    mindfulness_state: Optional[Any] = None
    deposit_strengths: Optional[torch.Tensor] = None


class CentralizedCritic(nn.Module):
    """
    Centralized critic for MAPPO that uses global state.

    Following MAPPO best practices (Yu et al., 2022):
    - Actors use local observations
    - Critic uses global state (concatenated observations from all agents)
    """

    def __init__(
        self,
        global_state_size: int,
        hidden_sizes: List[int] = [256, 256],
    ):
        super().__init__()

        layers = []
        prev_size = global_state_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.Tanh(),
            ])
            prev_size = hidden_size

        self.network = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_size, 1)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_state: Concatenated observations from all agents (batch, global_state_size)

        Returns:
            value: Centralized value estimate (batch,)
        """
        features = self.network(global_state)
        return self.value_head(features).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Actor-Critic network with modular augmentations.
    Supports both decentralized and centralized critic modes.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_sizes: List[int] = [256, 256],
        continuous: bool = False,
        diffusion_obs_size: int = 0,
        use_centralized_critic: bool = False,
        num_agents: int = 1,
    ):
        super().__init__()

        self.continuous = continuous
        self.action_size = action_size
        self.use_centralized_critic = use_centralized_critic
        self.num_agents = num_agents

        # Total input size includes diffusion observations
        total_obs_size = obs_size + diffusion_obs_size
        self.local_obs_size = total_obs_size

        # Shared feature extractor for actor
        layers = []
        prev_size = total_obs_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.Tanh(),
            ])
            prev_size = hidden_size

        self.shared = nn.Sequential(*layers)

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(prev_size, action_size)
            self.actor_logstd = nn.Parameter(torch.zeros(action_size))
        else:
            self.actor = nn.Linear(prev_size, action_size)

        # Critic head - decentralized (local) critic
        self.critic = nn.Linear(prev_size, 1)

        # Centralized critic (optional) - uses global state
        self.centralized_critic = None
        if use_centralized_critic:
            global_state_size = total_obs_size * num_agents
            self.centralized_critic = CentralizedCritic(
                global_state_size=global_state_size,
                hidden_sizes=hidden_sizes,
            )

    def forward(
        self,
        obs: torch.Tensor,
        diffusion_obs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            action_logits_or_mean: Logits (discrete) or mean (continuous)
            value: State value estimate
        """
        if diffusion_obs is not None:
            x = torch.cat([obs, diffusion_obs], dim=-1)
        else:
            x = obs

        features = self.shared(x)

        if self.continuous:
            action_out = self.actor_mean(features)
        else:
            action_out = self.actor(features)

        value = self.critic(features)

        return action_out, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        diffusion_obs: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        global_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log_prob, entropy, and value.

        Args:
            obs: Local observation
            diffusion_obs: Optional diffusion field observations
            action: Optional action for log_prob computation
            deterministic: Use greedy action selection
            global_state: Optional global state for centralized critic

        Returns:
            action, log_prob, entropy, value
        """
        action_out, local_value = self.forward(obs, diffusion_obs)

        if self.continuous:
            std = torch.exp(self.actor_logstd)
            dist = Normal(action_out, std)

            if action is None:
                if deterministic:
                    action = action_out
                else:
                    action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            dist = Categorical(logits=action_out)

            if action is None:
                if deterministic:
                    action = action_out.argmax(dim=-1)
                else:
                    action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        # Use centralized critic if available and global_state provided
        if self.centralized_critic is not None and global_state is not None:
            value = self.centralized_critic(global_state)
        else:
            value = local_value.squeeze(-1)

        return action, log_prob, entropy, value


class ContemplativeAgent(nn.Module):
    """
    Full Contemplative MARL agent with all modules.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        config,
        agent_id: int = 0,
        continuous: bool = False,
    ):
        super().__init__()

        self.config = config
        self.agent_id = agent_id
        self.obs_size = obs_size
        self.action_size = action_size
        self.continuous = continuous

        # Compute additional observation size from diffusion
        # Store this so forward() uses the same value the network was built with
        self.diffusion_obs_size = 0
        if config.diffusion.enabled:
            self.diffusion_obs_size = 3 * config.diffusion.num_channels  # value + grad_x + grad_y

        # Core actor-critic
        self.ac = ActorCritic(
            obs_size=obs_size,
            action_size=action_size,
            hidden_sizes=config.training.hidden_sizes,
            continuous=continuous,
            diffusion_obs_size=self.diffusion_obs_size,
        )

        # Module A: Ethics
        self.ethics = None
        if config.ethics.enabled:
            self.ethics = create_ethics_module(
                config.ethics,
                state_size=obs_size,
                action_size=action_size
            )

        # Module B: Diffusion
        self.diffusion_encoder = None
        self.diffusion_policy = None
        if config.diffusion.enabled:
            from .modules.diffusion import DiffusionEncoder, DiffusionPolicy
            self.diffusion_encoder = DiffusionEncoder(
                num_channels=config.diffusion.num_channels,
                sense_radius=config.diffusion.sense_radius,
            )
            self.diffusion_policy = DiffusionPolicy(
                state_size=config.training.hidden_sizes[-1],
                num_channels=config.diffusion.num_channels,
            )

        # Module C: Mindfulness
        self.mindfulness = None
        if config.mindfulness.enabled:
            self.mindfulness = create_mindfulness_module(
                config.mindfulness,
                obs_size=obs_size,
                action_size=action_size
            )

    def forward(
        self,
        obs: torch.Tensor,
        diffusion_field: Optional[DiffusionField] = None,
        position: Optional[np.ndarray] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> AgentOutput:
        """
        Full forward pass with all modules.
        """
        # Get diffusion observations if enabled
        diffusion_obs = None
        deposit_strengths = None

        if self.diffusion_obs_size > 0:
            # Use stored diffusion_obs_size to ensure consistency with network architecture
            if diffusion_field is not None and position is not None:
                # Sense field
                field_obs = diffusion_field.sense(position)
                diffusion_obs = torch.from_numpy(field_obs).float().to(obs.device).unsqueeze(0)
                # Validate size matches what network expects
                if diffusion_obs.shape[-1] != self.diffusion_obs_size:
                    raise ValueError(
                        f"Diffusion observation size mismatch: got {diffusion_obs.shape[-1]}, "
                        f"expected {self.diffusion_obs_size}. Check num_channels configuration."
                    )
            else:
                # No position available - use zeros to maintain expected input size
                diffusion_obs = torch.zeros(1, self.diffusion_obs_size, dtype=obs.dtype, device=obs.device)

        # Get action and value
        action_out, log_prob, entropy, value = self.ac.get_action_and_value(
            obs, diffusion_obs, action, deterministic
        )

        # Convert action to one-hot encoding for modules that need it
        if not self.continuous:
            action_onehot = F.one_hot(
                action_out.long() if action_out.dim() == 0 else action_out.long().squeeze(),
                num_classes=self.action_size
            ).float().unsqueeze(0)
        else:
            action_onehot = action_out.unsqueeze(0) if action_out.dim() == 1 else action_out

        # Apply mindfulness gating if enabled
        if self.mindfulness is not None and not deterministic:
            mindfulness_state = self.mindfulness.compute_mindfulness_state(
                obs, action_onehot
            )
            entropy_bonus = self.mindfulness.get_entropy_bonus(
                torch.tensor([1.0 if mindfulness_state.is_conservative else 0.0], device=obs.device)
            )
            # Could modify action selection here based on uncertainty
        else:
            mindfulness_state = None

        # Compute ethics score if enabled
        ethics_score = None
        if self.ethics is not None:
            ethics_score, _ = self.ethics(obs, action_onehot)

        # Compute deposit strengths for diffusion
        if self.diffusion_policy is not None:
            # Use hidden features to decide deposits
            features = self.ac.shared(obs if diffusion_obs is None else torch.cat([obs, diffusion_obs], dim=-1))
            deposit_strengths = self.diffusion_policy(features)

        return AgentOutput(
            action=action_out.detach().cpu().numpy(),
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            ethics_score=ethics_score,
            mindfulness_state=mindfulness_state,
            deposit_strengths=deposit_strengths,
        )

    def get_value(
        self,
        obs: torch.Tensor,
        diffusion_obs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get value estimate only."""
        _, value = self.ac.forward(obs, diffusion_obs)
        return value.squeeze(-1)


class MultiAgentSystem:
    """
    Manages multiple ContemplativeAgents with shared diffusion field.
    """

    def __init__(
        self,
        num_agents: int,
        obs_size: int,
        action_size: int,
        config,
        continuous: bool = False,
        share_params: bool = True,
    ):
        self.num_agents = num_agents
        self.config = config
        self.share_params = share_params

        # Create agents
        if share_params:
            # Parameter sharing: one network for all
            self.shared_agent = ContemplativeAgent(
                obs_size=obs_size,
                action_size=action_size,
                config=config,
                continuous=continuous,
            )
            self.agents = [self.shared_agent] * num_agents
        else:
            # Independent agents
            self.agents = [
                ContemplativeAgent(
                    obs_size=obs_size,
                    action_size=action_size,
                    config=config,
                    agent_id=i,
                    continuous=continuous,
                )
                for i in range(num_agents)
            ]

        # Shared diffusion field
        self.diffusion_field = None
        if config.diffusion.enabled:
            self.diffusion_field = DiffusionField(
                grid_size=config.diffusion.grid_size,
                num_channels=config.diffusion.num_channels,
            )

    def reset(self):
        """Reset for new episode."""
        if self.diffusion_field is not None:
            self.diffusion_field.reset()

    def step(
        self,
        observations: Dict[str, np.ndarray],
        positions: Optional[Dict[str, np.ndarray]] = None,
        deterministic: bool = False,
    ) -> Dict[str, AgentOutput]:
        """
        Get actions for all agents.
        """
        outputs = {}

        for i, (agent_id, obs) in enumerate(observations.items()):
            agent = self.agents[i] if not self.share_params else self.shared_agent

            device = next(agent.parameters()).device
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            pos = positions.get(agent_id) if positions else None

            output = agent(
                obs_tensor,
                diffusion_field=self.diffusion_field,
                position=pos,
                deterministic=deterministic,
            )

            # Deposit signals if enabled
            if output.deposit_strengths is not None and pos is not None:
                strengths = output.deposit_strengths.squeeze().detach().cpu().numpy()
                for c in range(self.config.diffusion.num_channels):
                    if strengths[c] > 0.1:  # Threshold
                        self.diffusion_field.deposit(
                            pos, c, strengths[c] * self.config.diffusion.deposit_strength
                        )

            outputs[agent_id] = output

        # Step diffusion field
        if self.diffusion_field is not None:
            self.diffusion_field.step()

        return outputs

    def parameters(self):
        """Get all trainable parameters."""
        if self.share_params:
            return self.shared_agent.parameters()
        else:
            params = []
            for agent in self.agents:
                params.extend(agent.parameters())
            return params

    def state_dict(self):
        """Get state dict for saving."""
        if self.share_params:
            return self.shared_agent.state_dict()
        else:
            return {f"agent_{i}": a.state_dict() for i, a in enumerate(self.agents)}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        if self.share_params:
            self.shared_agent.load_state_dict(state_dict)
        else:
            for i, a in enumerate(self.agents):
                a.load_state_dict(state_dict[f"agent_{i}"])
