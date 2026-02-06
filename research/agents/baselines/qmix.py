"""
QMIX: Monotonic Value Function Factorisation
============================================
Rashid et al., 2018

Value decomposition method that learns individual Q-functions
and combines them with a mixing network that enforces monotonicity.

Modern improvements implemented:
- Double DQN for reduced overestimation (van Hasselt et al., 2016)
- Prioritized Experience Replay (Schaul et al., 2015)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class QNetwork(nn.Module):
    """Individual agent Q-network."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 64,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_size)

        Returns:
            q_values: (batch, action_size)
        """
        return self.network(obs)


class MixingNetwork(nn.Module):
    """
    Mixing network that combines individual Q-values.

    Enforces monotonicity: dQ_tot/dQ_i >= 0 for all i
    This is achieved by using absolute value of weights.
    """

    def __init__(
        self,
        num_agents: int,
        state_size: int,
        hidden_size: int = 32,
        hypernet_hidden: int = 64,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.hidden_size = hidden_size

        # Hypernetworks generate mixing network weights from state
        # First layer weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_size, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, num_agents * hidden_size),
        )

        # First layer bias
        self.hyper_b1 = nn.Linear(state_size, hidden_size)

        # Second layer weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_size, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, hidden_size),
        )

        # Second layer bias (state-independent)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_size, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, 1),
        )

    def forward(
        self,
        q_values: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q_values: Individual Q-values (batch, num_agents)
            state: Global state (batch, state_size)

        Returns:
            q_total: Mixed Q-value (batch, 1)
        """
        batch_size = q_values.shape[0]

        # Generate weights (use abs for monotonicity)
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.num_agents, self.hidden_size)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.hidden_size)

        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.hidden_size, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # Forward pass through mixing network
        q_values = q_values.view(batch_size, 1, self.num_agents)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)
        q_total = torch.bmm(hidden, w2) + b2

        return q_total.squeeze(-1).squeeze(-1)


class ReplayBuffer:
    """Experience replay buffer for QMIX."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs: Dict[str, np.ndarray],
        state: np.ndarray,
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        next_state: np.ndarray,
        dones: Dict[str, bool],
    ):
        """Add transition to buffer."""
        self.buffer.append((obs, state, actions, rewards, next_obs, next_state, dones))

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return list(zip(*batch))

    def __len__(self):
        return len(self.buffer)


class SumTree:
    """Sum tree data structure for efficient priority sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index for given priority sum."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Total priority sum."""
        return self.tree[0]

    def add(self, priority: float, data):
        """Add data with given priority."""
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """Update priority at given index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, any]:
        """Get (index, priority, data) for given priority sum."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Schaul et al., 2015 - "Prioritized Experience Replay"
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
    ):
        """
        Args:
            capacity: Buffer capacity
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Frames to anneal beta to 1.0
            epsilon: Small constant to prevent zero priorities
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0
        self.max_priority = 1.0

    @property
    def beta(self) -> float:
        """Current beta value (annealed from beta_start to 1.0)."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def push(
        self,
        obs: Dict[str, np.ndarray],
        state: np.ndarray,
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        next_state: np.ndarray,
        dones: Dict[str, bool],
    ):
        """Add transition with max priority."""
        self.tree.add(self.max_priority ** self.alpha,
                      (obs, state, actions, rewards, next_obs, next_state, dones))
        self.frame += 1

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch with prioritized sampling.

        Returns:
            (obs, state, actions, rewards, next_obs, next_state, dones, indices, weights)
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        # Unzip batch
        obs, state, actions, rewards, next_obs, next_state, dones = zip(*batch)

        return obs, state, actions, rewards, next_obs, next_state, dones, indices, weights

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


class QMIX(nn.Module):
    """
    Full QMIX architecture.

    Each agent has individual Q-network.
    Mixing network combines individual Q-values using global state.
    """

    def __init__(
        self,
        num_agents: int,
        obs_size: int,
        state_size: int,
        action_size: int,
        hidden_size: int = 64,
        mixing_hidden: int = 32,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.action_size = action_size

        # Individual Q-networks (parameter sharing)
        self.q_network = QNetwork(obs_size, action_size, hidden_size)

        # Mixing network
        self.mixer = MixingNetwork(
            num_agents=num_agents,
            state_size=state_size,
            hidden_size=mixing_hidden,
        )

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, num_agents, obs_size)
            state: (batch, state_size)
            actions: Optional (batch, num_agents) for Q-value selection

        Returns:
            q_values: Individual Q-values (batch, num_agents, action_size) or (batch, num_agents)
            q_total: Mixed Q-value (batch,)
        """
        batch_size = obs.shape[0]

        # Get individual Q-values
        obs_flat = obs.view(batch_size * self.num_agents, -1)
        q_all = self.q_network(obs_flat)
        q_all = q_all.view(batch_size, self.num_agents, self.action_size)

        if actions is not None:
            # Select Q-values for taken actions
            actions = actions.unsqueeze(-1)
            q_values = q_all.gather(2, actions.long()).squeeze(-1)
        else:
            q_values = q_all

        # Mix Q-values
        if actions is not None:
            q_total = self.mixer(q_values, state)
        else:
            # Return max Q for action selection
            q_max, _ = q_all.max(dim=-1)
            q_total = self.mixer(q_max, state)

        return q_values, q_total

    def get_actions(
        self,
        obs: torch.Tensor,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        """
        Get actions using epsilon-greedy.

        Args:
            obs: (batch, num_agents, obs_size)
            epsilon: Exploration rate

        Returns:
            actions: (batch, num_agents)
        """
        batch_size = obs.shape[0]

        # Get Q-values
        obs_flat = obs.view(batch_size * self.num_agents, -1)
        q_values = self.q_network(obs_flat)
        q_values = q_values.view(batch_size, self.num_agents, self.action_size)

        # Epsilon-greedy
        if np.random.random() < epsilon:
            actions = torch.randint(0, self.action_size, (batch_size, self.num_agents))
        else:
            actions = q_values.argmax(dim=-1)

        return actions


class QMIXAgent:
    """
    QMIX agent wrapper with training.

    Modern improvements:
    - Double DQN: Action selection with online network, evaluation with target
    - Prioritized Experience Replay: Sample important transitions more often
    """

    def __init__(
        self,
        num_agents: int,
        obs_size: int,
        state_size: int,
        action_size: int,
        hidden_size: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 200,
        device: str = "cpu",
        use_double_dqn: bool = True,
        use_prioritized_replay: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
    ):
        self.num_agents = num_agents
        self.action_size = action_size
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Double DQN flag
        self.use_double_dqn = use_double_dqn

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0

        # Networks
        self.qmix = QMIX(
            num_agents=num_agents,
            obs_size=obs_size,
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
        ).to(self.device)

        self.target_qmix = QMIX(
            num_agents=num_agents,
            obs_size=obs_size,
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
        ).to(self.device)
        self.target_qmix.load_state_dict(self.qmix.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.qmix.parameters(), lr=lr)

        # Replay buffer (prioritized or uniform)
        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
            )
        else:
            self.buffer = ReplayBuffer(buffer_size)

    @property
    def epsilon(self) -> float:
        """Current epsilon value."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps / self.epsilon_decay)

    def act(
        self,
        observations: Dict[str, np.ndarray],
        explore: bool = True,
    ) -> Dict[str, int]:
        """Get actions for all agents."""
        agent_ids = list(observations.keys())

        obs_list = [observations[aid] for aid in agent_ids]
        obs_tensor = torch.tensor(
            np.array(obs_list), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        epsilon = self.epsilon if explore else 0.0

        with torch.no_grad():
            actions = self.qmix.get_actions(obs_tensor, epsilon)

        actions_np = actions.squeeze(0).cpu().numpy()
        return {aid: int(actions_np[i]) for i, aid in enumerate(agent_ids)}

    def store(
        self,
        obs: Dict[str, np.ndarray],
        state: np.ndarray,
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        next_state: np.ndarray,
        dones: Dict[str, bool],
    ):
        """Store transition in buffer."""
        self.buffer.push(obs, state, actions, rewards, next_obs, next_state, dones)
        self.steps += 1

    def update(self) -> Optional[float]:
        """
        Update networks from replay buffer.

        Uses Double DQN: select actions with online network, evaluate with target.
        Supports prioritized experience replay with importance sampling weights.
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch (different return format for prioritized vs uniform)
        if self.use_prioritized_replay:
            obs_batch, state_batch, actions_batch, rewards_batch, \
                next_obs_batch, next_state_batch, dones_batch, indices, weights = \
                self.buffer.sample(self.batch_size)
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            obs_batch, state_batch, actions_batch, rewards_batch, \
                next_obs_batch, next_state_batch, dones_batch = self.buffer.sample(self.batch_size)
            indices = None
            weights = None

        # Convert to tensors
        agent_ids = list(obs_batch[0].keys())

        obs = torch.tensor(
            np.array([[o[aid] for aid in agent_ids] for o in obs_batch]),
            dtype=torch.float32, device=self.device
        )
        state = torch.tensor(np.array(state_batch), dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            np.array([[a[aid] for aid in agent_ids] for a in actions_batch]),
            dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            np.array([[r[aid] for aid in agent_ids] for r in rewards_batch]),
            dtype=torch.float32, device=self.device
        ).mean(dim=1)  # Team reward
        next_obs = torch.tensor(
            np.array([[o[aid] for aid in agent_ids] for o in next_obs_batch]),
            dtype=torch.float32, device=self.device
        )
        next_state = torch.tensor(np.array(next_state_batch), dtype=torch.float32, device=self.device)
        dones = torch.tensor(
            np.array([any(d.values()) for d in dones_batch]),
            dtype=torch.float32, device=self.device
        )

        # Current Q-values
        _, q_total = self.qmix(obs, state, actions)

        # Target Q-values with Double DQN
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: select actions with online network
                batch_size = next_obs.shape[0]
                next_obs_flat = next_obs.view(batch_size * self.num_agents, -1)
                next_q_values = self.qmix.q_network(next_obs_flat)
                next_q_values = next_q_values.view(batch_size, self.num_agents, self.action_size)
                next_actions = next_q_values.argmax(dim=-1)

                # Evaluate selected actions with target network
                _, target_q_total = self.target_qmix(next_obs, next_state, next_actions)
            else:
                # Standard DQN: max Q from target network
                _, target_q_total = self.target_qmix(next_obs, next_state)

            target = rewards + self.gamma * (1 - dones) * target_q_total

        # TD error for prioritized replay
        td_errors = (q_total - target).detach().cpu().numpy()

        # Loss (weighted for prioritized replay)
        if weights is not None:
            loss = (weights * F.mse_loss(q_total, target, reduction='none')).mean()
        else:
            loss = F.mse_loss(q_total, target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qmix.parameters(), 10)
        self.optimizer.step()

        # Update priorities in prioritized replay buffer
        if self.use_prioritized_replay and indices is not None:
            self.buffer.update_priorities(indices, td_errors)

        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_qmix.load_state_dict(self.qmix.state_dict())

        return loss.item()

    def parameters(self):
        return self.qmix.parameters()

    def state_dict(self):
        return {
            'qmix': self.qmix.state_dict(),
            'target': self.target_qmix.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
        }

    def load_state_dict(self, state_dict):
        self.qmix.load_state_dict(state_dict['qmix'])
        self.target_qmix.load_state_dict(state_dict['target'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.steps = state_dict['steps']

    def to(self, device):
        self.device = torch.device(device)
        self.qmix.to(self.device)
        self.target_qmix.to(self.device)
        return self
