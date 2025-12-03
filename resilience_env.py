#!/usr/bin/env python3
"""
Resilience Environment v0 - Flood Disaster Scenario
====================================================
20×20 grid environment with flood hazard simulation.

Features:
- Dynamic flood zones spreading over time
- Resource distribution (food, shelter, medical)
- Agent population with health/energy tracking
- Hazard severity levels
- Rescue and cooperation mechanics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CellType(Enum):
    """Types of grid cells"""
    SAFE_LAND = 0
    FLOODED = 1
    RESOURCE_FOOD = 2
    RESOURCE_SHELTER = 3
    RESOURCE_MEDICAL = 4
    HIGH_GROUND = 5  # Safer from flooding


class HazardLevel(Enum):
    """Flood hazard severity"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentState:
    """State of an agent in the environment"""
    agent_id: int
    x: int
    y: int
    health: float = 1.0  # 0.0 to 1.0
    energy: float = 1.0  # 0.0 to 1.0
    has_food: bool = False
    has_shelter: bool = False
    has_medical: bool = False
    suffering_level: float = 0.0  # 0.0 to 1.0
    is_alive: bool = True
    rescued_others: int = 0
    times_rescued: int = 0


class ResilienceEnv:
    """
    Flood disaster resilience environment

    A 20×20 grid where agents must survive a spreading flood,
    collect resources, and help each other.
    """

    def __init__(
        self,
        grid_size: int = 20,
        num_agents: int = 10,
        flood_start_step: int = 20,
        flood_spread_rate: float = 0.05,
        resource_density: float = 0.1,
        seed: Optional[int] = None
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.flood_start_step = flood_start_step
        self.flood_spread_rate = flood_spread_rate
        self.resource_density = resource_density

        # Random seed
        if seed is not None:
            np.random.seed(seed)

        # Environment state
        self.step_count = 0
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.hazard_map = np.zeros((grid_size, grid_size), dtype=int)
        self.flood_map = np.zeros((grid_size, grid_size), dtype=float)

        # Agents
        self.agents: Dict[int, AgentState] = {}

        # Statistics
        self.total_casualties = 0
        self.total_suffering = 0.0
        self.total_rescues = 0

        # Initialize environment
        self._initialize_grid()
        self._initialize_agents()

    def _initialize_grid(self):
        """Initialize grid with resources and high ground"""
        # Place high ground (corners are safer)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Corners and edges are high ground
                if x < 3 or x >= self.grid_size - 3 or y < 3 or y >= self.grid_size - 3:
                    self.grid[x, y] = CellType.HIGH_GROUND.value

        # Place resources randomly
        num_resources = int(self.grid_size * self.grid_size * self.resource_density)

        for _ in range(num_resources // 3):
            x, y = np.random.randint(0, self.grid_size, size=2)
            if self.grid[x, y] == CellType.SAFE_LAND.value:
                self.grid[x, y] = CellType.RESOURCE_FOOD.value

        for _ in range(num_resources // 3):
            x, y = np.random.randint(0, self.grid_size, size=2)
            if self.grid[x, y] == CellType.SAFE_LAND.value:
                self.grid[x, y] = CellType.RESOURCE_SHELTER.value

        for _ in range(num_resources // 3):
            x, y = np.random.randint(0, self.grid_size, size=2)
            if self.grid[x, y] == CellType.SAFE_LAND.value:
                self.grid[x, y] = CellType.RESOURCE_MEDICAL.value

    def _initialize_agents(self):
        """Initialize agents at random safe positions"""
        for agent_id in range(self.num_agents):
            # Find safe starting position
            attempts = 0
            while attempts < 100:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)

                # Check if position is available
                occupied = any(
                    agent.x == x and agent.y == y
                    for agent in self.agents.values()
                )

                if not occupied:
                    self.agents[agent_id] = AgentState(
                        agent_id=agent_id,
                        x=x,
                        y=y
                    )
                    break

                attempts += 1

    def step(self, actions: Dict[int, str]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict]:
        """
        Step the environment

        Args:
            actions: Dict mapping agent_id to action string

        Returns:
            observations, rewards, dones, info
        """
        self.step_count += 1

        # Update flood
        self._update_flood()

        # Process agent actions
        rewards = {}
        for agent_id, action in actions.items():
            if agent_id in self.agents and self.agents[agent_id].is_alive:
                reward = self._process_action(agent_id, action)
                rewards[agent_id] = reward
            else:
                rewards[agent_id] = 0.0

        # Update agent states
        self._update_agents()

        # Check for casualties
        self._check_casualties()

        # Generate observations
        observations = self._get_observations()

        # Check if done
        dones = {
            agent_id: not agent.is_alive
            for agent_id, agent in self.agents.items()
        }
        dones['__all__'] = all(not agent.is_alive for agent in self.agents.values())

        # Info
        info = self.get_statistics()

        return observations, rewards, dones, info

    def _update_flood(self):
        """Update flood progression"""
        if self.step_count < self.flood_start_step:
            return

        # Start flood from center
        center_x, center_y = self.grid_size // 2, self.grid_size // 2

        if self.step_count == self.flood_start_step:
            # Initialize flood
            self.flood_map[center_x, center_y] = 1.0
            self.grid[center_x, center_y] = CellType.FLOODED.value
            self.hazard_map[center_x, center_y] = HazardLevel.CRITICAL.value

        # Spread flood
        new_flood_map = self.flood_map.copy()

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.flood_map[x, y] > 0.5:
                    # Spread to neighbors
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy

                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            # High ground resists flooding
                            if self.grid[nx, ny] == CellType.HIGH_GROUND.value:
                                spread_amount = self.flood_spread_rate * 0.3
                            else:
                                spread_amount = self.flood_spread_rate

                            new_flood_map[nx, ny] = min(
                                1.0,
                                new_flood_map[nx, ny] + spread_amount
                            )

        self.flood_map = new_flood_map

        # Update grid and hazard map
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                flood_level = self.flood_map[x, y]

                if flood_level > 0.8:
                    self.grid[x, y] = CellType.FLOODED.value
                    self.hazard_map[x, y] = HazardLevel.CRITICAL.value
                elif flood_level > 0.6:
                    self.hazard_map[x, y] = HazardLevel.HIGH.value
                elif flood_level > 0.4:
                    self.hazard_map[x, y] = HazardLevel.MEDIUM.value
                elif flood_level > 0.2:
                    self.hazard_map[x, y] = HazardLevel.LOW.value

    def _process_action(self, agent_id: int, action: str) -> float:
        """Process agent action and return reward"""
        agent = self.agents[agent_id]
        reward = 0.0

        if action.startswith("move_"):
            # Movement
            direction = action.split("_")[1]
            dx, dy = self._get_direction_delta(direction)

            new_x = np.clip(agent.x + dx, 0, self.grid_size - 1)
            new_y = np.clip(agent.y + dy, 0, self.grid_size - 1)

            # Check if position is available
            occupied = any(
                a.x == new_x and a.y == new_y and a.agent_id != agent_id
                for a in self.agents.values() if a.is_alive
            )

            if not occupied:
                agent.x = new_x
                agent.y = new_y
                agent.energy -= 0.02  # Movement cost

                # Reward for moving to safety
                if self.hazard_map[new_x, new_y] < self.hazard_map[agent.x - dx, agent.y - dy]:
                    reward += 0.5

        elif action == "collect_resource":
            # Collect resource at current location
            cell_type = self.grid[agent.x, agent.y]

            if cell_type == CellType.RESOURCE_FOOD.value and not agent.has_food:
                agent.has_food = True
                self.grid[agent.x, agent.y] = CellType.SAFE_LAND.value
                reward += 1.0

            elif cell_type == CellType.RESOURCE_SHELTER.value and not agent.has_shelter:
                agent.has_shelter = True
                self.grid[agent.x, agent.y] = CellType.SAFE_LAND.value
                reward += 1.0

            elif cell_type == CellType.RESOURCE_MEDICAL.value and not agent.has_medical:
                agent.has_medical = True
                self.grid[agent.x, agent.y] = CellType.SAFE_LAND.value
                reward += 1.0

        elif action == "help_other":
            # Help nearby agent
            helped = False

            for other_agent in self.agents.values():
                if other_agent.agent_id != agent_id and other_agent.is_alive:
                    # Check if nearby
                    distance = abs(other_agent.x - agent.x) + abs(other_agent.y - agent.y)

                    if distance <= 1:
                        # Transfer resources or rescue
                        if other_agent.health < 0.5 and agent.has_medical:
                            other_agent.health += 0.3
                            agent.has_medical = False
                            agent.rescued_others += 1
                            other_agent.times_rescued += 1
                            self.total_rescues += 1
                            reward += 2.0
                            helped = True
                            break

                        elif other_agent.energy < 0.3 and agent.has_food:
                            other_agent.energy += 0.3
                            agent.has_food = False
                            reward += 1.0
                            helped = True
                            break

            if helped:
                agent.energy -= 0.05  # Cost of helping

        elif action == "rest":
            # Rest to recover energy
            if agent.has_shelter:
                agent.energy = min(1.0, agent.energy + 0.15)
                reward += 0.2
            else:
                agent.energy = min(1.0, agent.energy + 0.05)

        return reward

    def _get_direction_delta(self, direction: str) -> Tuple[int, int]:
        """Get x, y delta for direction"""
        directions = {
            'north': (0, -1),
            'south': (0, 1),
            'east': (1, 0),
            'west': (-1, 0)
        }
        return directions.get(direction, (0, 0))

    def _update_agents(self):
        """Update agent states based on environment"""
        for agent in self.agents.values():
            if not agent.is_alive:
                continue

            # Hazard damage
            hazard = self.hazard_map[agent.x, agent.y]

            if hazard == HazardLevel.CRITICAL.value:
                agent.health -= 0.15
                agent.suffering_level = min(1.0, agent.suffering_level + 0.2)
            elif hazard == HazardLevel.HIGH.value:
                agent.health -= 0.08
                agent.suffering_level = min(1.0, agent.suffering_level + 0.1)
            elif hazard == HazardLevel.MEDIUM.value:
                agent.health -= 0.03
                agent.suffering_level = min(1.0, agent.suffering_level + 0.05)

            # Natural energy drain
            agent.energy -= 0.01

            # Suffering from low health/energy
            if agent.health < 0.3:
                agent.suffering_level = min(1.0, agent.suffering_level + 0.1)
            if agent.energy < 0.2:
                agent.suffering_level = min(1.0, agent.suffering_level + 0.05)

            # Food consumption
            if agent.has_food and agent.energy < 0.5:
                agent.energy = min(1.0, agent.energy + 0.3)
                agent.has_food = False

            # Shelter protection
            if agent.has_shelter and hazard > HazardLevel.NONE.value:
                agent.health = min(1.0, agent.health + 0.05)  # Shelter provides protection

            # Medical use
            if agent.has_medical and agent.health < 0.7:
                agent.health = min(1.0, agent.health + 0.4)
                agent.has_medical = False

            # Suffering decay
            agent.suffering_level = max(0.0, agent.suffering_level - 0.02)

    def _check_casualties(self):
        """Check for agent casualties"""
        for agent in self.agents.values():
            if agent.is_alive and (agent.health <= 0.0 or agent.energy <= 0.0):
                agent.is_alive = False
                self.total_casualties += 1
                logger.info(f"Agent {agent.agent_id} casualty at step {self.step_count}")

    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Generate observations for each agent"""
        observations = {}

        for agent_id, agent in self.agents.items():
            if not agent.is_alive:
                observations[agent_id] = np.zeros(20)
                continue

            # Local view (5x5 around agent)
            view_size = 5
            half_view = view_size // 2

            local_grid = np.zeros((view_size, view_size))
            local_hazard = np.zeros((view_size, view_size))

            for i in range(view_size):
                for j in range(view_size):
                    x = agent.x + (i - half_view)
                    y = agent.y + (j - half_view)

                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        local_grid[i, j] = self.grid[x, y]
                        local_hazard[i, j] = self.hazard_map[x, y]

            # Flatten and combine with agent state
            obs = np.concatenate([
                local_grid.flatten()[:10],  # Grid view (truncated)
                local_hazard.flatten()[:5],  # Hazard view (truncated)
                [agent.health, agent.energy, agent.suffering_level],
                [float(agent.has_food), float(agent.has_shelter), float(agent.has_medical)]
            ])

            # Pad to fixed size
            obs = np.pad(obs, (0, max(0, 20 - len(obs))))[:20]

            observations[agent_id] = obs

        return observations

    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics"""
        alive_agents = [a for a in self.agents.values() if a.is_alive]

        return {
            'step': self.step_count,
            'alive_agents': len(alive_agents),
            'casualties': self.total_casualties,
            'total_suffering': sum(a.suffering_level for a in alive_agents),
            'avg_suffering': np.mean([a.suffering_level for a in alive_agents]) if alive_agents else 0.0,
            'avg_health': np.mean([a.health for a in alive_agents]) if alive_agents else 0.0,
            'avg_energy': np.mean([a.energy for a in alive_agents]) if alive_agents else 0.0,
            'total_rescues': self.total_rescues,
            'flooded_cells': np.sum(self.grid == CellType.FLOODED.value),
            'flood_coverage': np.mean(self.flood_map)
        }

    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment"""
        self.step_count = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.hazard_map = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.flood_map = np.zeros((self.grid_size, self.grid_size), dtype=float)
        self.agents = {}
        self.total_casualties = 0
        self.total_suffering = 0.0
        self.total_rescues = 0

        self._initialize_grid()
        self._initialize_agents()

        return self._get_observations()

    def render(self) -> str:
        """Render environment as ASCII"""
        lines = [f"\n=== Resilience Environment - Step {self.step_count} ==="]

        # Create visualization
        viz = np.full((self.grid_size, self.grid_size), ' ', dtype='<U1')

        # Show grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_type = self.grid[x, y]

                if cell_type == CellType.FLOODED.value:
                    viz[x, y] = '~'
                elif cell_type == CellType.HIGH_GROUND.value:
                    viz[x, y] = '^'
                elif cell_type == CellType.RESOURCE_FOOD.value:
                    viz[x, y] = 'F'
                elif cell_type == CellType.RESOURCE_SHELTER.value:
                    viz[x, y] = 'S'
                elif cell_type == CellType.RESOURCE_MEDICAL.value:
                    viz[x, y] = 'M'
                else:
                    hazard = self.hazard_map[x, y]
                    if hazard > 0:
                        viz[x, y] = str(hazard)
                    else:
                        viz[x, y] = '.'

        # Show agents
        for agent in self.agents.values():
            if agent.is_alive:
                viz[agent.x, agent.y] = 'A'

        # Print grid
        for y in range(self.grid_size):
            line = ''.join(viz[:, y])
            lines.append(line)

        # Statistics
        stats = self.get_statistics()
        lines.append(f"\nAlive: {stats['alive_agents']}, Casualties: {stats['casualties']}")
        lines.append(f"Avg Health: {stats['avg_health']:.2f}, Avg Energy: {stats['avg_energy']:.2f}")
        lines.append(f"Suffering: {stats['total_suffering']:.1f}, Rescues: {stats['total_rescues']}")
        lines.append(f"Flood coverage: {stats['flood_coverage']:.1%}")

        return '\n'.join(lines)


if __name__ == "__main__":
    # Quick test
    env = ResilienceEnv(grid_size=20, num_agents=10, seed=42)

    print("ResilienceEnv Test:")
    print(env.render())

    # Run a few steps
    for step in range(5):
        actions = {
            agent_id: np.random.choice(['move_north', 'move_south', 'move_east', 'move_west', 'rest'])
            for agent_id in env.agents.keys()
        }

        obs, rewards, dones, info = env.step(actions)

    print(env.render())
    print("\n✓ ResilienceEnv initialized successfully")
