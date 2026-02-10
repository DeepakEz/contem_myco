"""
Multi-Agent Particle Environment Wrapper
=========================================
PettingZoo MPE wrapper with mixed-motive rewards and ethical metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gymnasium as gym

# Try to import PettingZoo
try:
    from pettingzoo.mpe import simple_spread_v3, simple_adversary_v3, simple_tag_v3
    from pettingzoo.utils import parallel_to_aec
    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    print("Warning: PettingZoo not available. Install with: pip install 'pettingzoo[mpe]'")


@dataclass
class EpisodeMetrics:
    """Metrics collected during an episode."""
    total_reward: float = 0.0
    individual_rewards: Dict[str, float] = None
    social_welfare: float = 0.0
    gini_coefficient: float = 0.0
    cooperation_rate: float = 0.0
    collision_count: int = 0
    goal_reached_count: int = 0
    harm_events: int = 0

    def __post_init__(self):
        if self.individual_rewards is None:
            self.individual_rewards = {}


class MixedMotiveMPE:
    """
    Wrapper for MPE environments with mixed-motive reward structure.

    Adds:
    - Individual vs team reward weighting
    - Ethical metrics tracking (harm, fairness)
    - Partial observability options
    - Noise injection for robustness testing
    """

    def __init__(
        self,
        env_name: str = "simple_spread",
        num_agents: int = 4,
        num_landmarks: int = 4,
        max_steps: int = 100,
        individual_reward_weight: float = 0.3,
        local_obs_radius: Optional[float] = None,
        obs_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
        reward_noise_std: float = 0.0,
        render_mode: Optional[str] = None,
    ):
        if not PETTINGZOO_AVAILABLE:
            raise ImportError("PettingZoo not available")

        self.env_name = env_name
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.max_steps = max_steps
        self.individual_reward_weight = individual_reward_weight
        self.team_reward_weight = 1.0 - individual_reward_weight
        self.local_obs_radius = local_obs_radius
        self.obs_noise_std = obs_noise_std
        self.action_noise_std = action_noise_std
        self.reward_noise_std = reward_noise_std

        # Create base environment
        if env_name == "simple_spread":
            self.env = simple_spread_v3.parallel_env(
                N=num_agents,
                local_ratio=0.5,
                max_cycles=max_steps,
                continuous_actions=False,
                render_mode=render_mode,
            )
        elif env_name == "simple_adversary":
            self.env = simple_adversary_v3.parallel_env(
                N=num_agents,
                max_cycles=max_steps,
                continuous_actions=False,
                render_mode=render_mode,
            )
        elif env_name == "simple_tag":
            self.env = simple_tag_v3.parallel_env(
                num_good=num_agents - 1,
                num_adversaries=1,
                num_obstacles=2,
                max_cycles=max_steps,
                continuous_actions=False,
                render_mode=render_mode,
            )
        else:
            raise ValueError(f"Unknown environment: {env_name}")

        # Get spaces
        self.agents = self.env.possible_agents
        self.observation_spaces = {a: self.env.observation_space(a) for a in self.agents}
        self.action_spaces = {a: self.env.action_space(a) for a in self.agents}

        # Metrics tracking
        self.episode_metrics = EpisodeMetrics()
        self.step_count = 0

        # Agent positions (extracted from observations)
        self.agent_positions = {}

        # Get actual observation size from a test reset (PettingZoo spaces may not match actual obs)
        test_obs, _ = self.env.reset()
        self._actual_obs_size = len(list(test_obs.values())[0])
        # Reset again to clear state
        self.env.reset()

    @property
    def obs_size(self) -> int:
        """Get observation size (from actual observations, not space which may differ)."""
        return self._actual_obs_size

    @property
    def action_size(self) -> int:
        """Get action size (assuming homogeneous agents)."""
        space = list(self.action_spaces.values())[0]
        if hasattr(space, 'n'):
            return space.n
        return space.shape[0]

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment."""
        observations, infos = self.env.reset(seed=seed)

        self.episode_metrics = EpisodeMetrics()
        self.step_count = 0

        # Apply observation noise
        observations = self._apply_obs_noise(observations)

        # Apply partial observability
        if self.local_obs_radius is not None:
            observations = self._apply_local_obs(observations)

        # Extract positions
        self._extract_positions(observations)

        return observations, infos

    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict]:
        """
        Step environment with mixed-motive rewards.
        """
        # Apply action noise
        if self.action_noise_std > 0:
            actions = self._apply_action_noise(actions)

        # Step base environment
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        self.step_count += 1

        # Apply observation noise
        observations = self._apply_obs_noise(observations)

        # Extract positions
        self._extract_positions(observations)

        # Compute mixed-motive rewards
        rewards = self._compute_mixed_rewards(rewards, observations, actions)

        # Apply reward noise
        if self.reward_noise_std > 0:
            rewards = self._apply_reward_noise(rewards)

        # Update metrics
        self._update_metrics(rewards, observations, actions)

        # Add ethical info
        for agent in infos:
            infos[agent]['metrics'] = {
                'gini': self.episode_metrics.gini_coefficient,
                'cooperation': self.episode_metrics.cooperation_rate,
                'harm_events': self.episode_metrics.harm_events,
            }

        return observations, rewards, terminations, truncations, infos

    def _apply_obs_noise(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Add Gaussian noise to observations."""
        if self.obs_noise_std <= 0:
            return observations

        noisy = {}
        for agent, obs in observations.items():
            noise = np.random.normal(0, self.obs_noise_std, obs.shape)
            noisy[agent] = (obs + noise).astype(np.float32)
        return noisy

    def _apply_action_noise(self, actions: Dict[str, int]) -> Dict[str, int]:
        """Apply noise to actions (random action with some probability)."""
        noisy = {}
        for agent, action in actions.items():
            if np.random.random() < self.action_noise_std:
                noisy[agent] = self.action_spaces[agent].sample()
            else:
                noisy[agent] = action
        return noisy

    def _apply_reward_noise(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Add Gaussian noise to rewards."""
        noisy = {}
        for agent, reward in rewards.items():
            noise = np.random.normal(0, self.reward_noise_std)
            noisy[agent] = reward + noise
        return noisy

    def _apply_local_obs(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply partial observability based on distance."""
        # For now, just return observations
        # Full implementation would mask observations beyond radius
        return observations

    def _extract_positions(self, observations: Dict[str, np.ndarray]):
        """Extract agent positions from observations."""
        for agent, obs in observations.items():
            # In MPE, first 2-4 elements are typically velocity, next 2 are position relative to landmarks
            # This is environment-specific; adjust based on actual obs structure
            if len(obs) >= 4:
                # Approximate position from observation
                self.agent_positions[agent] = obs[2:4].copy()

    def _compute_mixed_rewards(
        self,
        base_rewards: Dict[str, float],
        observations: Dict[str, np.ndarray],
        actions: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Compute mixed individual + team rewards.

        Individual: Original per-agent reward
        Team: Average reward across all agents (social welfare)
        """
        # Compute team reward (average)
        team_reward = np.mean(list(base_rewards.values()))

        mixed_rewards = {}
        for agent, individual_reward in base_rewards.items():
            mixed = (
                self.individual_reward_weight * individual_reward +
                self.team_reward_weight * team_reward
            )
            mixed_rewards[agent] = mixed

        return mixed_rewards

    def _update_metrics(
        self,
        rewards: Dict[str, float],
        observations: Dict[str, np.ndarray],
        actions: Dict[str, int]
    ):
        """Update episode metrics."""
        reward_values = np.array(list(rewards.values()))

        # Total and individual rewards
        self.episode_metrics.total_reward += reward_values.sum()
        for agent, r in rewards.items():
            if agent not in self.episode_metrics.individual_rewards:
                self.episode_metrics.individual_rewards[agent] = 0.0
            self.episode_metrics.individual_rewards[agent] += r

        # Social welfare
        self.episode_metrics.social_welfare = self.episode_metrics.total_reward

        # Gini coefficient
        self.episode_metrics.gini_coefficient = self._compute_gini(
            list(self.episode_metrics.individual_rewards.values())
        )

        # Cooperation rate (simplified: based on action similarity)
        action_values = list(actions.values())
        if len(action_values) > 1:
            # Count similar actions
            from collections import Counter
            action_counts = Counter(action_values)
            most_common_count = action_counts.most_common(1)[0][1]
            self.episode_metrics.cooperation_rate = most_common_count / len(action_values)

        # Collision detection (simplified: based on position proximity)
        positions = list(self.agent_positions.values())
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if len(positions[i]) >= 2 and len(positions[j]) >= 2:
                    dist = np.linalg.norm(positions[i][:2] - positions[j][:2])
                    if dist < 0.1:  # Collision threshold
                        self.episode_metrics.collision_count += 1
                        self.episode_metrics.harm_events += 1

    def _compute_gini(self, values: List[float]) -> float:
        """Compute Gini coefficient."""
        if len(values) < 2:
            return 0.0

        values = np.array(values)
        # Shift to positive
        values = values - values.min() + 1e-8

        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
        return max(0, min(1, gini))

    def get_positions(self) -> Dict[str, np.ndarray]:
        """Get current agent positions."""
        return self.agent_positions.copy()

    def get_metrics(self) -> EpisodeMetrics:
        """Get current episode metrics."""
        return self.episode_metrics

    def render(self):
        """Render environment."""
        return self.env.render()

    def close(self):
        """Close environment."""
        self.env.close()


class SocialDilemmaEnv:
    """
    Social dilemma environment for testing ethical behavior.

    Implements:
    - Prisoner's Dilemma
    - Public Goods Game
    - Tragedy of the Commons
    """

    def __init__(
        self,
        game_type: str = "prisoners_dilemma",
        num_agents: int = 4,
        num_rounds: int = 100,
        temptation: float = 1.5,
        sucker: float = -0.5,
        reward: float = 1.0,
        punishment: float = 0.0,
    ):
        self.game_type = game_type
        self.num_agents = num_agents
        self.num_rounds = num_rounds

        # Payoff parameters
        self.T = temptation  # Temptation to defect
        self.S = sucker       # Sucker's payoff
        self.R = reward       # Reward for mutual cooperation
        self.P = punishment   # Punishment for mutual defection

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.current_round = 0

        # Observation: last round actions of all agents
        self.obs_size = num_agents
        self.action_size = 2  # Cooperate (0) or Defect (1)

        # History
        self.action_history = []
        self.reward_history = []

        # Episode metrics
        self._episode_metrics = EpisodeMetrics()

        # Assign fixed grid positions for stigmergic diffusion
        # Place agents evenly spaced in [-0.8, 0.8] range
        self._agent_positions = {}
        spacing = 1.6 / max(num_agents, 1)
        for i, agent in enumerate(self.agents):
            x = -0.8 + spacing * (i + 0.5)
            y = 0.0  # 1D arrangement for simple games
            self._agent_positions[agent] = np.array([x, y], dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset game."""
        if seed is not None:
            np.random.seed(seed)

        self.current_round = 0
        self.action_history = []
        self.reward_history = []
        self._episode_metrics = EpisodeMetrics()

        # Initial observation: all zeros (no history)
        observations = {
            agent: np.zeros(self.obs_size, dtype=np.float32)
            for agent in self.agents
        }

        return observations, {}

    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict]:
        """Step one round."""
        self.current_round += 1

        # Compute rewards based on game type
        if self.game_type == "prisoners_dilemma":
            rewards = self._prisoners_dilemma_rewards(actions)
        elif self.game_type == "public_goods":
            rewards = self._public_goods_rewards(actions)
        else:
            rewards = self._commons_rewards(actions)

        # Store history
        action_vec = np.array([actions[a] for a in self.agents])
        self.action_history.append(action_vec)
        self.reward_history.append(rewards)

        # Update metrics
        self._update_metrics(rewards, actions)

        # Create observations (last round actions)
        observations = {
            agent: action_vec.astype(np.float32)
            for agent in self.agents
        }

        # Check termination
        done = self.current_round >= self.num_rounds
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # Metrics
        cooperation_rate = 1.0 - np.mean(action_vec)  # 0 = cooperate
        infos = {
            agent: {
                'cooperation_rate': cooperation_rate,
                'round': self.current_round,
            }
            for agent in self.agents
        }

        return observations, rewards, terminations, truncations, infos

    def _prisoners_dilemma_rewards(self, actions: Dict[str, int]) -> Dict[str, float]:
        """N-player iterated prisoner's dilemma."""
        rewards = {}
        action_list = [actions[a] for a in self.agents]
        num_defectors = sum(action_list)
        num_cooperators = len(action_list) - num_defectors

        for agent in self.agents:
            my_action = actions[agent]
            if my_action == 0:  # Cooperate
                # Get R from each cooperator, S from each defector
                rewards[agent] = (
                    self.R * (num_cooperators - 1) / (self.num_agents - 1) +
                    self.S * num_defectors / (self.num_agents - 1)
                )
            else:  # Defect
                # Get T from each cooperator, P from each defector
                rewards[agent] = (
                    self.T * num_cooperators / (self.num_agents - 1) +
                    self.P * (num_defectors - 1) / (self.num_agents - 1)
                )

        return rewards

    def _public_goods_rewards(self, actions: Dict[str, int]) -> Dict[str, float]:
        """Public goods game."""
        rewards = {}
        contributions = sum(1 - actions[a] for a in self.agents)  # 0 = contribute
        public_good = contributions * 1.5 / self.num_agents  # Multiplier

        for agent in self.agents:
            personal_cost = 0.5 if actions[agent] == 0 else 0  # Cost to contribute
            rewards[agent] = public_good - personal_cost

        return rewards

    def _commons_rewards(self, actions: Dict[str, int]) -> Dict[str, float]:
        """Tragedy of the commons."""
        rewards = {}
        total_consumption = sum(actions[a] for a in self.agents)  # 1 = consume more
        depletion = total_consumption / self.num_agents

        for agent in self.agents:
            personal_gain = 0.5 if actions[agent] == 1 else 0
            shared_loss = depletion * 0.3
            rewards[agent] = personal_gain - shared_loss

        return rewards

    def _update_metrics(self, rewards: Dict[str, float], actions: Dict[str, int]):
        """Update episode metrics."""
        reward_values = np.array(list(rewards.values()))

        # Total reward
        self._episode_metrics.total_reward += reward_values.sum()

        # Individual rewards
        for agent, r in rewards.items():
            if agent not in self._episode_metrics.individual_rewards:
                self._episode_metrics.individual_rewards[agent] = 0.0
            self._episode_metrics.individual_rewards[agent] += r

        # Social welfare
        self._episode_metrics.social_welfare = self._episode_metrics.total_reward

        # Gini coefficient
        values = list(self._episode_metrics.individual_rewards.values())
        if len(values) >= 2:
            values = np.array(values)
            values = values - values.min() + 1e-8
            sorted_values = np.sort(values)
            n = len(sorted_values)
            cumsum = np.cumsum(sorted_values)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
            self._episode_metrics.gini_coefficient = max(0, min(1, gini))

        # Cooperation rate (0 = cooperate)
        action_values = list(actions.values())
        self._episode_metrics.cooperation_rate = 1.0 - np.mean(action_values)

    def get_metrics(self) -> EpisodeMetrics:
        """Get current episode metrics."""
        return self._episode_metrics

    def get_positions(self) -> Dict[str, np.ndarray]:
        """Get agent positions for stigmergic diffusion field."""
        return self._agent_positions.copy()

    def close(self):
        """Close environment (no-op for this env)."""
        pass


def create_env(config) -> MixedMotiveMPE:
    """Create environment from config."""
    return MixedMotiveMPE(
        env_name=config.env.name.replace("mpe_", ""),
        num_agents=config.env.num_agents,
        num_landmarks=config.env.num_landmarks,
        max_steps=config.env.max_steps,
        individual_reward_weight=config.env.individual_reward_weight,
        local_obs_radius=config.env.local_obs_radius if config.env.local_obs_radius > 0 else None,
        obs_noise_std=config.env.obs_noise_std,
        action_noise_std=config.env.action_noise_std,
        reward_noise_std=config.env.reward_noise_std,
    )
