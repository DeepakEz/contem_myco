"""
MycoNet 3.0 MycoAgent Module
============================

Autonomous cognitive agent with reflexive learning capabilities.

Core Architecture:
1. Perception → Internal World Model → Policy → Action
2. Reflexive monitoring (mindfulness, self-evaluation)
3. Symbolic reasoning module
4. Multi-tier memory system
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import logging
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from .config import AgentConfig, WisdomSignalType, InsightType

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions an agent can take."""
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    EAT_FOOD = 4
    COLLECT_WATER = 5
    REST = 6
    REPRODUCE = 7
    MEDITATE = 8
    SHARE_WISDOM = 9
    HELP_OTHER = 10
    EXPLORE = 11
    NO_ACTION = 12


@dataclass
class WisdomInsight:
    """Discrete knowledge unit stored in agent memory."""
    type: InsightType
    content: Dict[str, Any]
    intensity: float  # Confidence/importance [0,1]
    timestamp: int
    origin_agent_id: Optional[int] = None
    decay_rate: float = 0.01

    def decay(self, current_time: int):
        """Apply time-based decay to intensity."""
        time_elapsed = current_time - self.timestamp
        self.intensity *= np.exp(-self.decay_rate * time_elapsed)


@dataclass
class AgentState:
    """Complete state of an agent."""
    # Identity
    agent_id: int

    # Position
    x: int
    y: int

    # Vital stats
    energy: float = 1.0
    health: float = 1.0
    water: float = 1.0
    age: int = 0

    # Cognitive state
    mindfulness_level: float = 0.5
    wisdom_accumulated: float = 0.0
    prediction_error: float = 0.0

    # Action state
    last_action: Optional[ActionType] = None
    last_decision_time: int = 0
    decision_period: int = 10

    # Social state
    is_cooperating: bool = False
    in_conflict: bool = False
    resources_consumed: float = 0.0
    resources_produced: float = 0.0


if TORCH_AVAILABLE:
    class AgentPolicyNet(nn.Module):
        """
        Policy network with hierarchical structure.
        Multi-scale perception (conv layers) + decision layers.
        """

        def __init__(self, input_dim: int = 64, hidden_dim: int = 128,
                     num_actions: int = 13):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_actions = num_actions

            # Encoder network
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            # Action head
            self.action_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_actions)
            )

            # Value head (for actor-critic)
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Args:
                x: Input observation [batch, input_dim]

            Returns:
                action_logits: [batch, num_actions]
                value: [batch, 1]
            """
            features = self.encoder(x)
            action_logits = self.action_head(features)
            value = self.value_head(features)
            return action_logits, value

        def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, float]:
            """
            Sample action from policy.

            Returns:
                action: Sampled action index
                log_prob: Log probability of action
                value: Estimated value
            """
            action_logits, value = self.forward(x)
            probs = F.softmax(action_logits, dim=-1)

            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()

            log_prob = torch.log(probs[action] + 1e-8)

            return action.item(), log_prob.item(), value.item()


    class AgentWorldModel(nn.Module):
        """
        Predictive model for environment dynamics.
        Outputs: next_state_pred, uncertainty
        """

        def __init__(self, state_dim: int = 64, action_dim: int = 13,
                     hidden_dim: int = 128):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim

            # State encoder
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )

            # Action encoder
            self.action_encoder = nn.Embedding(action_dim, hidden_dim // 4)

            # Dynamics model
            combined_dim = hidden_dim // 2 + hidden_dim // 4
            self.dynamics = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            # Prediction heads
            self.state_predictor = nn.Linear(hidden_dim, state_dim)
            self.uncertainty_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()  # Ensure positive uncertainty
            )

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Predict next state and uncertainty.

            Args:
                state: Current state [batch, state_dim]
                action: Action taken [batch] (integers)

            Returns:
                next_state_pred: Predicted next state [batch, state_dim]
                uncertainty: Prediction uncertainty [batch, 1]
            """
            state_features = self.state_encoder(state)
            action_features = self.action_encoder(action)

            combined = torch.cat([state_features, action_features], dim=-1)
            dynamics_features = self.dynamics(combined)

            next_state_pred = self.state_predictor(dynamics_features)
            uncertainty = self.uncertainty_predictor(dynamics_features)

            return next_state_pred, uncertainty

        def compute_prediction_error(self, state: torch.Tensor, action: torch.Tensor,
                                     actual_next_state: torch.Tensor) -> float:
            """Compute prediction error for phenomenal curvature."""
            pred_state, _ = self.forward(state, action)
            error = F.mse_loss(pred_state, actual_next_state)
            return error.item()


    class MindfulnessMonitor:
        """
        Tracks agent's cognitive coherence.

        Metrics:
        - focus_coherence: How unified attention is
        - attention_stability: Variance of attention over time
        - distraction_events: Count of focus breaks
        """

        def __init__(self, window_size: int = 20):
            self.window_size = window_size

            # Tracking buffers
            self.attention_history: deque = deque(maxlen=window_size)
            self.action_history: deque = deque(maxlen=window_size)
            self.reward_history: deque = deque(maxlen=window_size)

            # Metrics
            self.focus_coherence: float = 0.5
            self.attention_stability: float = 0.5
            self.distraction_count: int = 0

            # Mindfulness state
            self.current_focus: Optional[str] = None
            self.focus_start_time: int = 0

        def update(self, attention_vector: np.ndarray, action: ActionType,
                   reward: float, time_step: int):
            """Update mindfulness state with new observation."""
            self.attention_history.append(attention_vector.copy())
            self.action_history.append(action)
            self.reward_history.append(reward)

            # Compute focus coherence
            if len(self.attention_history) >= 3:
                recent = np.array(list(self.attention_history)[-3:])
                self.focus_coherence = 1.0 - np.mean(np.std(recent, axis=0))

            # Compute attention stability
            if len(self.attention_history) >= 5:
                all_attention = np.array(list(self.attention_history))
                self.attention_stability = 1.0 - np.mean(np.var(all_attention, axis=0))

            # Detect distraction (action change without reward improvement)
            if len(self.action_history) >= 2:
                if (self.action_history[-1] != self.action_history[-2] and
                        len(self.reward_history) >= 2 and
                        self.reward_history[-1] <= self.reward_history[-2]):
                    self.distraction_count += 1

        def get_mindfulness_level(self) -> float:
            """Compute overall mindfulness level."""
            return 0.5 * self.focus_coherence + 0.5 * self.attention_stability

        def get_state(self) -> Dict[str, float]:
            """Get complete mindfulness state."""
            return {
                'mindfulness_level': self.get_mindfulness_level(),
                'focus_coherence': self.focus_coherence,
                'attention_stability': self.attention_stability,
                'distraction_count': self.distraction_count
            }


class MycoAgent:
    """
    Autonomous cognitive agent with reflexive learning.

    Core Architecture:
    1. Perception → Internal World Model → Policy → Action
    2. Reflexive monitoring (mindfulness, self-evaluation)
    3. Symbolic reasoning module
    4. Multi-tier memory system
    """

    def __init__(self, config, agent_id: int = 0,
                 initial_position: Tuple[int, int] = (0, 0)):
        self.id = agent_id
        self.agent_id = agent_id  # Alias for compatibility
        # Accept either MycoNetConfig or AgentConfig
        if hasattr(config, 'agent'):
            self.config = config.agent
            self.full_config = config
        else:
            self.config = config
            self.full_config = None

        # Position
        self.x, self.y = initial_position
        self._position = initial_position

        # State
        self.alive = True
        self.mindfulness_state = {
            'focus_coherence': 0.5,
            'attention_stability': 0.5,
            'distraction_level': 0.0
        }

        # Vital stats
        self.energy = self.config.initial_energy
        self.health = self.config.initial_health
        self.water = 1.0
        self.age = 0

        # Neural networks (if PyTorch available)
        self.policy_net = None
        self.world_model = None
        self.optimizer = None
        self.world_model_optimizer = None
        if TORCH_AVAILABLE:
            self.policy_net = AgentPolicyNet(
                input_dim=64,
                hidden_dim=self.config.hidden_dim,
                num_actions=len(ActionType)
            )
            self.world_model = AgentWorldModel(
                state_dim=64,
                action_dim=len(ActionType),
                hidden_dim=self.config.hidden_dim
            )

            # Optimizers for policy and world model
            self.optimizer = torch.optim.Adam(
                self.policy_net.parameters(),
                lr=self.config.learning_rate
            )
            self.world_model_optimizer = torch.optim.Adam(
                self.world_model.parameters(),
                lr=self.config.learning_rate
            )

        # Cognitive monitoring
        self.mindfulness_monitor = MindfulnessMonitor() if TORCH_AVAILABLE else None
        self.prediction_error = 0.0
        self.phenomenal_curvature = 0.0

        # Memory systems
        self.short_term_memory: deque = deque(maxlen=self.config.short_term_memory_size)
        self.working_memory: Dict[str, Any] = {}
        self.wisdom_memory: List[WisdomInsight] = []

        # Current state
        self.last_observation: Optional[Dict[str, Any]] = None
        self.last_action: Optional[ActionType] = None
        self.last_decision_time: int = 0
        self.decision_period: int = 10
        self.last_signal: Optional[str] = None

        # Social state
        self.is_cooperating = False
        self.in_conflict = False
        self.resources_consumed = 0.0
        self.resources_produced = 0.0

        # Learning state
        self.experience_buffer: List[Dict[str, Any]] = []
        self.cumulative_reward = 0.0

        # Exploration parameter
        self.exploration_rate = 0.3

        logger.debug(f"Agent {agent_id} initialized at ({self.x}, {self.y})")

    def perceive(self, env_obs: Dict[str, Any], signal_map: Dict[WisdomSignalType, float]):
        """
        Update internal state from environment observation.

        Args:
            env_obs: Observation from environment
            signal_map: Local wisdom signal intensities
        """
        # Store in short-term memory
        self.short_term_memory.append({
            'observation': env_obs,
            'signals': signal_map,
            'time': self.age
        })

        self.last_observation = env_obs

        # Update position from observation
        if 'position' in env_obs:
            self.x, self.y = env_obs['position']

        # Process signals
        self._process_wisdom_signals(signal_map)

        # Update working memory with salient features
        self._update_working_memory(env_obs)

    def _process_wisdom_signals(self, signal_map: Dict[WisdomSignalType, float]):
        """Process wisdom signals and potentially learn from them."""
        for signal_type, intensity in signal_map.items():
            if intensity > 0.5:
                # Strong signal - may trigger behavior change
                if signal_type == WisdomSignalType.SUFFERING_ALERT:
                    self.working_memory['suffering_nearby'] = True

                elif signal_type == WisdomSignalType.MEDITATION_SYNC:
                    # Boost mindfulness from external meditation
                    if self.mindfulness_monitor:
                        self.mindfulness_monitor.focus_coherence *= 1.1

                elif signal_type == WisdomSignalType.DANGER_WARNING:
                    self.working_memory['danger_alert'] = True

    def _update_working_memory(self, env_obs: Dict[str, Any]):
        """Update working memory with relevant features."""
        # Resource awareness
        if 'local_resources' in env_obs:
            resources = env_obs['local_resources']
            self.working_memory['resources_nearby'] = np.sum(resources) > 0.1

        # Social awareness
        if 'nearby_agents' in env_obs:
            self.working_memory['agents_nearby'] = len(env_obs['nearby_agents'])

        # Goal tracking
        if self.energy < 0.3:
            self.working_memory['current_goal'] = 'find_food'
        elif self.water < 0.3:
            self.working_memory['current_goal'] = 'find_water'
        else:
            self.working_memory['current_goal'] = 'explore'

    def decide_action(self) -> ActionType:
        """
        Run policy with symbolic augmentation to select action.

        Returns:
            Chosen action
        """
        self.last_decision_time = self.age

        # Build state tensor
        state_tensor = self._build_state_tensor()

        # Get base action from neural policy
        base_action = self._get_neural_action(state_tensor)

        # Apply symbolic augmentation
        final_action = self._symbolic_augment_policy(base_action)

        self.last_action = final_action
        return final_action

    def _build_state_tensor(self) -> np.ndarray:
        """Build state tensor from current observations."""
        state = np.zeros(64, dtype=np.float32)

        # Internal state (8 dims)
        state[0] = self.energy
        state[1] = self.health
        state[2] = self.water
        state[3] = self.age / 1000.0  # Normalize age
        state[4] = self.x / 64.0  # Normalize position
        state[5] = self.y / 64.0
        state[6] = self.mindfulness_monitor.get_mindfulness_level() if self.mindfulness_monitor else 0.5
        state[7] = len(self.wisdom_memory) / 100.0

        # Local environment (18 dims - 3x3 grid * 2 features)
        if self.last_observation and 'local_resources' in self.last_observation:
            resources = self.last_observation['local_resources']
            flat_resources = resources.flatten()[:18]
            state[8:8 + len(flat_resources)] = flat_resources

        # Nearby agents (10 dims)
        if self.last_observation and 'nearby_agents' in self.last_observation:
            nearby = self.last_observation['nearby_agents'][:5]
            for i, agent in enumerate(nearby):
                state[26 + i * 2] = agent.get('distance', 10) / 10.0
                state[27 + i * 2] = 1.0  # Agent present flag

        # Wisdom signals (14 dims - 7 signal types * 2)
        if self.last_observation and 'local_signals' in self.last_observation:
            signals = self.last_observation['local_signals']
            for i, (signal_type, intensity) in enumerate(signals.items()):
                if i < 7:
                    state[36 + i * 2] = intensity
                    state[37 + i * 2] = 1.0 if intensity > 0.5 else 0.0

        # Working memory features (14 dims)
        state[50] = 1.0 if self.working_memory.get('resources_nearby', False) else 0.0
        state[51] = self.working_memory.get('agents_nearby', 0) / 10.0
        state[52] = 1.0 if self.working_memory.get('suffering_nearby', False) else 0.0
        state[53] = 1.0 if self.working_memory.get('danger_alert', False) else 0.0

        return state

    def _get_neural_action(self, state: np.ndarray) -> ActionType:
        """Get action from neural policy network."""
        if self.policy_net is None or not TORCH_AVAILABLE:
            return self._random_action()

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_idx, log_prob, value = self.policy_net.get_action(
                state_tensor,
                deterministic=(np.random.random() > self.exploration_rate)
            )

        return ActionType(action_idx)

    def _random_action(self) -> ActionType:
        """Fallback random action selection."""
        # Bias towards survival actions when low on resources
        if self.energy < 0.3:
            weights = [0.15] * 4 + [0.3, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0]
        elif self.water < 0.3:
            weights = [0.15] * 4 + [0.0, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0]
        else:
            weights = [0.1] * 4 + [0.08] * 9

        weights = np.array(weights)
        weights /= weights.sum()

        action_idx = np.random.choice(len(ActionType), p=weights)
        return ActionType(action_idx)

    def _symbolic_augment_policy(self, base_action: ActionType) -> ActionType:
        """Apply symbolic rules/constraints to base action."""
        # Check for ethical overrides
        if self.working_memory.get('suffering_nearby', False):
            # Prioritize helping if we have resources
            if self.energy > 0.5 and base_action not in [ActionType.HELP_OTHER, ActionType.SHARE_WISDOM]:
                return ActionType.HELP_OTHER

        # Check for danger response
        if self.working_memory.get('danger_alert', False):
            # Override with escape action
            if base_action not in [ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH,
                                   ActionType.MOVE_EAST, ActionType.MOVE_WEST]:
                # Move away from danger (random direction for now)
                return ActionType(np.random.randint(0, 4))

        # Mindfulness-based adjustment
        if self.mindfulness_monitor and self.mindfulness_monitor.distraction_count > 5:
            # Too distracted - consider meditation
            if np.random.random() < 0.3:
                return ActionType.MEDITATE

        return base_action

    def learn(self, reward: float, next_state: Dict[str, Any]):
        """
        Update networks based on experience.

        Args:
            reward: Reward from action
            next_state: New observation after action
        """
        self.cumulative_reward += reward

        # Store experience
        experience = {
            'state': self._build_state_tensor().copy(),
            'action': self.last_action,
            'reward': reward,
            'next_state': next_state,
            'done': False
        }
        self.experience_buffer.append(experience)

        # Update mindfulness monitor
        if self.mindfulness_monitor:
            attention = self._build_state_tensor()[:16]  # Use first 16 dims as attention proxy
            self.mindfulness_monitor.update(attention, self.last_action, reward, self.age)

        # Compute prediction error if world model available
        if self.world_model is not None and TORCH_AVAILABLE:
            self._update_prediction_error(experience)

        # Prune experience buffer
        if len(self.experience_buffer) > 1000:
            self.experience_buffer = self.experience_buffer[-500:]

    def _update_prediction_error(self, experience: Dict[str, Any]):
        """Update prediction error using world model."""
        if len(self.experience_buffer) < 2:
            return

        prev = self.experience_buffer[-2]

        state = torch.tensor(prev['state'], dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([prev['action'].value], dtype=torch.long)
        next_state = torch.tensor(experience['state'], dtype=torch.float32).unsqueeze(0)

        self.prediction_error = self.world_model.compute_prediction_error(
            state, action, next_state
        )

    def compute_phenomenal_curvature(self) -> float:
        """
        Compute phenomenal curvature from prediction error history.

        Φ_f = ||∂²error/∂θ²||
        """
        if len(self.experience_buffer) < 3:
            return 0.0

        # Get recent prediction errors
        errors = []
        for exp in self.experience_buffer[-10:]:
            errors.append(exp.get('prediction_error', self.prediction_error))

        if len(errors) < 3:
            return 0.0

        # Second derivative approximation
        d2_error = np.diff(errors, n=2)
        self.phenomenal_curvature = float(np.mean(np.abs(d2_error)))

        return self.phenomenal_curvature

    def emit_signals(self, signal_grid: Any):
        """
        Generate signals based on current state.

        Args:
            signal_grid: WisdomSignalGrid to emit signals to
        """
        # Reset last emitted signal tracker
        self.last_signal = None

        # Suffering signal if low on resources
        if self.energy < 0.2 or self.health < 0.2:
            signal_grid.add_signal(
                WisdomSignalType.SUFFERING_ALERT,
                self.x, self.y,
                intensity=0.8,
                content={'agent_id': self.id, 'type': 'resource_need'}
            )
            self.last_signal = 'SUFFERING_ALERT'

        # Meditation signal if meditating
        if self.last_action == ActionType.MEDITATE:
            signal_grid.add_signal(
                WisdomSignalType.MEDITATION_SYNC,
                self.x, self.y,
                intensity=0.6,
                content={'agent_id': self.id}
            )
            self.last_signal = 'MEDITATION_SYNC'

        # Wisdom beacon if high wisdom
        if len(self.wisdom_memory) > 10:
            signal_grid.add_signal(
                WisdomSignalType.WISDOM_BEACON,
                self.x, self.y,
                intensity=min(1.0, len(self.wisdom_memory) / 50),
                content={'agent_id': self.id, 'wisdom_count': len(self.wisdom_memory)}
            )
            self.last_signal = 'WISDOM_BEACON'

    def reflect_and_store(self):
        """
        Create WisdomInsights from experiences.

        Called periodically to consolidate learning.
        """
        if len(self.experience_buffer) < 5:
            return

        # Analyze recent experiences
        recent = self.experience_buffer[-10:]
        rewards = [exp['reward'] for exp in recent]

        # Create insight if pattern detected
        if np.mean(rewards) > 0.5:
            # Positive pattern - store as practical wisdom
            insight = WisdomInsight(
                type=InsightType.PRACTICAL,
                content={
                    'pattern': 'positive_reward_sequence',
                    'mean_reward': float(np.mean(rewards)),
                    'actions': [exp['action'].name for exp in recent[-3:]]
                },
                intensity=0.7,
                timestamp=self.age,
                origin_agent_id=self.id
            )
            self._store_wisdom(insight)

        elif np.mean(rewards) < -0.3:
            # Negative pattern - store as cautionary insight
            insight = WisdomInsight(
                type=InsightType.ETHICAL,
                content={
                    'pattern': 'negative_reward_sequence',
                    'mean_reward': float(np.mean(rewards)),
                    'actions_to_avoid': [exp['action'].name for exp in recent[-3:]]
                },
                intensity=0.6,
                timestamp=self.age,
                origin_agent_id=self.id
            )
            self._store_wisdom(insight)

    def _store_wisdom(self, insight: WisdomInsight):
        """Store wisdom insight with capacity management."""
        self.wisdom_memory.append(insight)

        # Prune low-intensity insights if over capacity
        if len(self.wisdom_memory) > self.config.wisdom_memory_capacity:
            # Decay all insights
            for w in self.wisdom_memory:
                w.decay(self.age)

            # Remove low-intensity ones
            self.wisdom_memory = [w for w in self.wisdom_memory if w.intensity > 0.1]

            # If still over capacity, keep most recent
            if len(self.wisdom_memory) > self.config.wisdom_memory_capacity:
                self.wisdom_memory = sorted(
                    self.wisdom_memory,
                    key=lambda w: (w.intensity, w.timestamp),
                    reverse=True
                )[:self.config.wisdom_memory_capacity]

    def step(self, dt: float = 1.0):
        """
        Update agent's internal state (aging, resource consumption, etc.).

        Args:
            dt: Time step size
        """
        self.age += 1

        # Resource consumption
        energy_cost = 0.01 * dt
        if self.last_action in [ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH,
                                ActionType.MOVE_EAST, ActionType.MOVE_WEST]:
            energy_cost *= 2
        elif self.last_action == ActionType.MEDITATE:
            energy_cost *= 0.5  # Meditation conserves energy
            # Boost mindfulness
            if self.mindfulness_monitor:
                self.mindfulness_monitor.focus_coherence = min(
                    1.0, self.mindfulness_monitor.focus_coherence + 0.05
                )

        self.energy = max(0, self.energy - energy_cost)
        self.resources_consumed += energy_cost

        # Health decay if starving
        if self.energy < 0.1:
            self.health -= 0.02 * dt

        # Clear temporary working memory
        self.working_memory.pop('suffering_nearby', None)
        self.working_memory.pop('danger_alert', None)

    def consume_resource(self, amount: float, resource_type: str = 'food'):
        """Consume a resource to replenish energy/water."""
        if resource_type == 'food':
            self.energy = min(1.0, self.energy + amount)
        elif resource_type == 'water':
            self.water = min(1.0, self.water + amount)
        self.resources_consumed += amount

    def is_alive(self) -> bool:
        """Check if agent is still alive."""
        return self.health > 0 and self.age < 10000

    def get_state(self) -> AgentState:
        """Get complete agent state."""
        return AgentState(
            agent_id=self.id,
            x=self.x,
            y=self.y,
            energy=self.energy,
            health=self.health,
            water=self.water,
            age=self.age,
            mindfulness_level=self.mindfulness_monitor.get_mindfulness_level()
            if self.mindfulness_monitor else 0.5,
            wisdom_accumulated=len(self.wisdom_memory),
            prediction_error=self.prediction_error,
            last_action=self.last_action,
            last_decision_time=self.last_decision_time,
            decision_period=self.decision_period,
            is_cooperating=self.is_cooperating,
            in_conflict=self.in_conflict,
            resources_consumed=self.resources_consumed,
            resources_produced=self.resources_produced
        )

    def get_state_dict(self) -> Dict[str, Any]:
        """Get state as dictionary for metrics computation."""
        state = self.get_state()
        return {
            'agent_id': state.agent_id,
            'x': state.x,
            'y': state.y,
            'energy': state.energy,
            'health': state.health,
            'water': state.water,
            'age': state.age,
            'mindfulness_level': state.mindfulness_level,
            'wisdom_accumulated': state.wisdom_accumulated,
            'prediction_error': state.prediction_error,
            'last_action': state.last_action.name if state.last_action else None,
            'last_decision_time': state.last_decision_time,
            'decision_period': state.decision_period,
            'is_cooperating': state.is_cooperating,
            'in_conflict': state.in_conflict,
            'resources_consumed': state.resources_consumed,
            'resources_produced': state.resources_produced
        }

    def save(self, filepath: str):
        """Save agent state to file."""
        if TORCH_AVAILABLE:
            state_dict = {
                'id': self.id,
                'position': (self.x, self.y),
                'vitals': {
                    'energy': self.energy,
                    'health': self.health,
                    'water': self.water,
                    'age': self.age
                },
                'policy_net': self.policy_net.state_dict() if self.policy_net else None,
                'world_model': self.world_model.state_dict() if self.world_model else None,
                'wisdom_count': len(self.wisdom_memory),
                'cumulative_reward': self.cumulative_reward
            }
            torch.save(state_dict, filepath)
        else:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.get_state_dict(), f)

    @classmethod
    def load(cls, filepath: str, config: AgentConfig) -> 'MycoAgent':
        """Load agent from file."""
        if TORCH_AVAILABLE:
            state_dict = torch.load(filepath)
            agent = cls(state_dict['id'], config, state_dict['position'])
            agent.energy = state_dict['vitals']['energy']
            agent.health = state_dict['vitals']['health']
            agent.water = state_dict['vitals']['water']
            agent.age = state_dict['vitals']['age']

            if state_dict['policy_net'] and agent.policy_net:
                agent.policy_net.load_state_dict(state_dict['policy_net'])
            if state_dict['world_model'] and agent.world_model:
                agent.world_model.load_state_dict(state_dict['world_model'])

            agent.cumulative_reward = state_dict['cumulative_reward']
            return agent
        else:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls(data['agent_id'], config, (data['x'], data['y']))

    @property
    def position(self) -> Tuple[int, int]:
        """Get agent position."""
        return (self.x, self.y)

    @position.setter
    def position(self, value: Tuple[int, int]):
        """Set agent position."""
        self.x, self.y = value
        self._position = value

    def reset(self, position: Tuple[int, int] = None):
        """Reset agent to initial state."""
        if position is not None:
            self.x, self.y = position
            self._position = position

        self.energy = self.config.initial_energy
        self.health = self.config.initial_health
        self.water = 1.0
        self.age = 0
        self.alive = True

        self.prediction_error = 0.0
        self.cumulative_reward = 0.0
        self.last_action = None
        self.last_signal = None
        self.is_cooperating = False
        self.in_conflict = False
        self.resources_consumed = 0.0
        self.resources_produced = 0.0

        self.short_term_memory.clear()
        self.working_memory.clear()

    def select_action(self, observation: np.ndarray,
                     deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action given observation.

        Args:
            observation: Observation array
            deterministic: If True, select greedy action

        Returns:
            action: Selected action index
            value: Value estimate
            log_prob: Log probability of action
        """
        if self.policy_net is not None and TORCH_AVAILABLE:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                logits, value = self.policy_net(obs_tensor)

                if deterministic:
                    action = logits.argmax(dim=-1).item()
                    log_prob = 0.0
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action_tensor = dist.sample()
                    action = action_tensor.item()
                    log_prob = dist.log_prob(action_tensor).item()

                return action, value.item(), log_prob
        else:
            # Random action if no neural network
            action = np.random.randint(0, len(ActionType))
            return action, 0.0, 0.0

    def get_state_vector(self) -> np.ndarray:
        """Get agent state as numpy array."""
        return np.array([
            self.energy,
            self.health,
            self.mindfulness_monitor.get_mindfulness_level() if self.mindfulness_monitor else 0.5,
            self.x,
            self.y,
            self.water,
            self.age / 10000.0,  # Normalized age
            self.prediction_error,
            1.0 if self.is_cooperating else 0.0,
            1.0 if self.in_conflict else 0.0
        ], dtype=np.float32)

    def load_state_vector(self, state: np.ndarray):
        """Load agent state from numpy array."""
        if len(state) >= 10:
            self.energy = float(state[0])
            self.health = float(state[1])
            self.x = int(state[3])
            self.y = int(state[4])
            self.water = float(state[5])
            self.age = int(state[6] * 10000)
            self.prediction_error = float(state[7])
            self.is_cooperating = bool(state[8] > 0.5)
            self.in_conflict = bool(state[9] > 0.5)

    def load_weights_from_dict(self, weights: Dict[str, Any]):
        """Load network weights from dictionary (e.g., from hypernetwork)."""
        if not TORCH_AVAILABLE or self.policy_net is None:
            return

        # Load weights into policy network
        for name, param in self.policy_net.named_parameters():
            if name in weights:
                param.data.copy_(weights[name])

        if self.world_model is not None:
            for name, param in self.world_model.named_parameters():
                if name in weights:
                    param.data.copy_(weights[name])
