"""
MycoNet 3.0 Environment Module
==============================

Simulates the 2D grid world with resources, signal propagation, and environmental dynamics.
Implements full environment physics as specified in the blueprint.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from scipy.ndimage import convolve
from scipy.signal import convolve2d
import logging

from .config import EnvironmentConfig, WisdomSignalType

logger = logging.getLogger(__name__)


class TerrainType(Enum):
    """Types of terrain in the environment."""
    OPEN = 0
    OBSTACLE = 1
    WATER_SOURCE = 2
    FOOD_SOURCE = 3
    HAZARD = 4


class StochasticEvent(Enum):
    """Types of random environmental events."""
    NONE = auto()
    RESOURCE_BLOOM = auto()
    RESOURCE_DEPLETION = auto()
    DISASTER = auto()
    WEATHER_CHANGE = auto()
    PREDATOR_APPEARANCE = auto()


@dataclass
class ResourcePatch:
    """Represents a resource patch in the environment."""
    x: int
    y: int
    resource_type: str  # 'food' or 'water'
    amount: float
    max_amount: float
    regeneration_rate: float
    radius: int = 2


@dataclass
class EnvironmentState:
    """Complete state of the environment at a given time."""
    time_step: int
    resource_grid: np.ndarray  # 2D array of resource concentrations
    signal_layers: Dict[WisdomSignalType, np.ndarray]  # Signal grids for each type
    terrain_grid: np.ndarray  # 2D array of terrain types
    agent_positions: Dict[int, Tuple[int, int]]  # agent_id -> (x, y)
    active_events: List[StochasticEvent]
    metrics: Dict[str, float]


class WisdomSignalGrid:
    """
    Multi-layer signal propagation system.

    Implements diffusion equations for each signal type:
    ∂s_i/∂t = D_i∇²s_i - λ_i s_i + Σ I_agent δ(x-x_agent)

    Where:
    - D_i is the diffusion rate for signal type i
    - λ_i is the decay rate for signal type i
    - I_agent is the emission intensity from an agent
    """

    # Signal-specific diffusion and decay rates
    SIGNAL_PROPERTIES = {
        WisdomSignalType.SUFFERING_ALERT: {'diffusion': 0.15, 'decay': 0.03},
        WisdomSignalType.WISDOM_BEACON: {'diffusion': 0.08, 'decay': 0.02},
        WisdomSignalType.COMPASSION_GRADIENT: {'diffusion': 0.12, 'decay': 0.04},
        WisdomSignalType.MEDITATION_SYNC: {'diffusion': 0.20, 'decay': 0.08},
        WisdomSignalType.ETHICAL_INSIGHT: {'diffusion': 0.10, 'decay': 0.03},
        WisdomSignalType.HELP_REQUEST: {'diffusion': 0.18, 'decay': 0.05},
        WisdomSignalType.DANGER_WARNING: {'diffusion': 0.25, 'decay': 0.06},
        WisdomSignalType.COOPERATION_CALL: {'diffusion': 0.14, 'decay': 0.04},
    }

    # Signal interaction rules (amplification/dampening)
    SIGNAL_INTERACTIONS = {
        (WisdomSignalType.COMPASSION_GRADIENT, WisdomSignalType.MEDITATION_SYNC): 1.3,  # Amplify
        (WisdomSignalType.SUFFERING_ALERT, WisdomSignalType.COMPASSION_GRADIENT): 1.2,
        (WisdomSignalType.WISDOM_BEACON, WisdomSignalType.MEDITATION_SYNC): 1.4,
        (WisdomSignalType.DANGER_WARNING, WisdomSignalType.COOPERATION_CALL): 0.8,  # Dampen
    }

    def __init__(self, width: int, height: int, base_diffusion: float = 0.1, base_decay: float = 0.05):
        self.width = width
        self.height = height
        self.base_diffusion = base_diffusion
        self.base_decay = base_decay

        # Initialize signal layers
        self.layers: Dict[WisdomSignalType, np.ndarray] = {
            signal_type: np.zeros((height, width), dtype=np.float32)
            for signal_type in WisdomSignalType
        }

        # Content grids for carrying semantic information
        self.content_grids: Dict[WisdomSignalType, List[List[Optional[Dict]]]] = {
            signal_type: [[None for _ in range(width)] for _ in range(height)]
            for signal_type in WisdomSignalType
        }

        # Create diffusion kernel (Laplacian approximation)
        self.diffusion_kernel = np.array([
            [0.05, 0.2, 0.05],
            [0.2, -1.0, 0.2],
            [0.05, 0.2, 0.05]
        ], dtype=np.float32)

    def add_signal(self, signal_type: WisdomSignalType, x: int, y: int,
                   intensity: float, content: Optional[Dict] = None):
        """Add a signal emission at the specified location."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.layers[signal_type][y, x] = min(1.0, self.layers[signal_type][y, x] + intensity)
            if content is not None:
                self.content_grids[signal_type][y][x] = content

    def propagate(self, dt: float = 1.0):
        """
        Execute one diffusion step for all signal layers.

        Implements the PDE: ∂s/∂t = D∇²s - λs
        """
        for signal_type in WisdomSignalType:
            # Get signal-specific parameters
            props = self.SIGNAL_PROPERTIES.get(
                signal_type,
                {'diffusion': self.base_diffusion, 'decay': self.base_decay}
            )
            D = props['diffusion']
            decay = props['decay']

            layer = self.layers[signal_type]

            # Compute Laplacian using convolution
            laplacian = convolve2d(layer, self.diffusion_kernel, mode='same', boundary='wrap')

            # Update: s += dt * (D * ∇²s - λ * s)
            layer += dt * (D * laplacian - decay * layer)

            # Clamp values
            np.clip(layer, 0.0, 1.0, out=layer)

            # Remove very weak signals
            layer[layer < 0.01] = 0.0

        # Apply signal interactions
        self._apply_interactions()

    def _apply_interactions(self):
        """Apply signal interaction rules (amplification/dampening)."""
        for (sig1, sig2), factor in self.SIGNAL_INTERACTIONS.items():
            layer1 = self.layers[sig1]
            layer2 = self.layers[sig2]

            # Find overlapping regions
            overlap_mask = (layer1 > 0.1) & (layer2 > 0.1)

            if factor > 1.0:
                # Amplification
                amplification = (factor - 1.0) * layer1[overlap_mask]
                layer2[overlap_mask] = np.minimum(1.0, layer2[overlap_mask] + amplification * 0.1)
            else:
                # Dampening
                dampening = (1.0 - factor) * layer1[overlap_mask]
                layer2[overlap_mask] = np.maximum(0.0, layer2[overlap_mask] - dampening * 0.1)

    def get_local(self, x: int, y: int, radius: int = 2) -> Dict[WisdomSignalType, float]:
        """Return signal intensities in local neighborhood."""
        signals = {}

        for signal_type, layer in self.layers.items():
            # Sum intensities in neighborhood
            x_min = max(0, x - radius)
            x_max = min(self.width, x + radius + 1)
            y_min = max(0, y - radius)
            y_max = min(self.height, y + radius + 1)

            local_sum = np.sum(layer[y_min:y_max, x_min:x_max])
            area = (x_max - x_min) * (y_max - y_min)
            signals[signal_type] = local_sum / max(area, 1)

        return signals

    def get_gradient(self, signal_type: WisdomSignalType, x: int, y: int) -> Tuple[float, float]:
        """Compute signal gradient at location (for gradient following)."""
        layer = self.layers[signal_type]

        # Use central differences
        if 0 < x < self.width - 1:
            grad_x = (layer[y, x + 1] - layer[y, x - 1]) / 2.0
        else:
            grad_x = 0.0

        if 0 < y < self.height - 1:
            grad_y = (layer[y + 1, x] - layer[y - 1, x]) / 2.0
        else:
            grad_y = 0.0

        return (grad_x, grad_y)

    def get_network_stats(self) -> Dict[str, float]:
        """Compute statistics about the wisdom signal network."""
        all_signals = []
        for layer in self.layers.values():
            all_signals.append(layer.flatten())

        if not all_signals:
            return {'signal_diversity': 0.0, 'network_coherence': 0.0}

        combined = np.concatenate(all_signals)

        # Signal diversity: variance of signal intensities
        diversity = float(np.std(combined))

        # Network coherence: mean signal intensity (simplified)
        coherence = float(np.mean(combined))

        # Active cells: proportion of cells with any signal
        active_ratio = float(np.mean(combined > 0.1))

        # Compute per-layer statistics
        layer_stats = {}
        for signal_type, layer in self.layers.items():
            layer_stats[signal_type.name] = {
                'max_intensity': float(np.max(layer)),
                'mean_intensity': float(np.mean(layer)),
                'active_cells': float(np.sum(layer > 0.1))
            }

        return {
            'signal_diversity': diversity,
            'network_coherence': coherence,
            'active_ratio': active_ratio,
            'total_signal_mass': float(np.sum(combined)),
            'layer_stats': layer_stats
        }


class Environment:
    """
    Simulates the world with 2D grid, resources, and signal propagation.

    Implements full environment physics:
    - 2D spatial grid (configurable size)
    - Resource dynamics (generation, depletion)
    - Signal diffusion following PDEs
    - Obstacle/terrain features
    - Environmental stochasticity for testing RX
    """

    def __init__(self, config):
        # Accept either MycoNetConfig or EnvironmentConfig
        if hasattr(config, 'environment'):
            self.config = config.environment
        else:
            self.config = config
        self.width, self.height = self.config.grid_size
        self.time_step = 0

        # Initialize grids
        self.resource_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.terrain_grid = np.zeros((self.height, self.width), dtype=np.int32)

        # Signal system
        self.signal_grid = WisdomSignalGrid(
            self.width, self.height,
            base_diffusion=self.config.signal_diffusion_rate,
            base_decay=self.config.signal_decay_rate
        )

        # Resource patches
        self.resource_patches: List[ResourcePatch] = []

        # Agent tracking
        self.agents: Dict[int, Any] = {}  # agent_id -> agent reference
        self.agent_positions: Dict[int, Tuple[int, int]] = {}

        # Event tracking
        self.active_events: List[StochasticEvent] = []
        self.event_history: List[Dict[str, Any]] = []

        # Initialize environment
        self._initialize_terrain()
        self._initialize_resources()

        logger.info(f"Environment initialized: {self.width}x{self.height} grid")

    def _initialize_terrain(self):
        """Initialize terrain with obstacles and features."""
        # Place random obstacles
        num_obstacles = int(self.width * self.height * self.config.obstacle_density)
        for _ in range(num_obstacles):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            self.terrain_grid[y, x] = TerrainType.OBSTACLE.value

    def _initialize_resources(self):
        """Initialize resource patches throughout the environment."""
        for i in range(self.config.num_resources):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)

            # Avoid obstacles
            while self.terrain_grid[y, x] == TerrainType.OBSTACLE.value:
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)

            resource_type = 'food' if i % 2 == 0 else 'water'
            patch = ResourcePatch(
                x=x, y=y,
                resource_type=resource_type,
                amount=np.random.uniform(0.5, 1.0),
                max_amount=1.0,
                regeneration_rate=self.config.resource_regeneration_rate,
                radius=np.random.randint(1, 4)
            )
            self.resource_patches.append(patch)

            # Mark terrain
            terrain_type = TerrainType.FOOD_SOURCE if resource_type == 'food' else TerrainType.WATER_SOURCE
            self.terrain_grid[y, x] = terrain_type.value

            # Add resources to grid
            self._apply_resource_patch(patch)

    def _apply_resource_patch(self, patch: ResourcePatch):
        """Apply a resource patch to the resource grid."""
        for dy in range(-patch.radius, patch.radius + 1):
            for dx in range(-patch.radius, patch.radius + 1):
                nx, ny = patch.x + dx, patch.y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    if distance <= patch.radius:
                        # Resource intensity decreases with distance from center
                        intensity = patch.amount * (1 - distance / (patch.radius + 1))
                        self.resource_grid[ny, nx] = max(self.resource_grid[ny, nx], intensity)

    def step(self, actions_dict: Dict[int, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Apply agent actions, update physics, and diffuse signals.

        Args:
            actions_dict: Dictionary mapping agent_id to their chosen action

        Returns:
            Dictionary mapping agent_id to their action outcomes
        """
        self.time_step += 1
        outcomes = {}

        # Process agent actions
        for agent_id, action in actions_dict.items():
            outcome = self._process_agent_action(agent_id, action)
            outcomes[agent_id] = outcome

        # Update environment physics
        self._update_resources()
        self._update_signals()
        self._process_stochastic_events()

        return outcomes

    def _process_agent_action(self, agent_id: int, action: Any) -> Dict[str, Any]:
        """Process an individual agent's action."""
        outcome = {
            'success': False,
            'reward': 0.0,
            'info': {}
        }

        if agent_id not in self.agent_positions:
            outcome['info']['error'] = 'Agent not found'
            return outcome

        x, y = self.agent_positions[agent_id]

        # Action processing depends on action type
        # This is a simplified version - full implementation would handle all action types
        if hasattr(action, 'name'):
            action_name = action.name
        else:
            action_name = str(action)

        if 'MOVE' in action_name:
            outcome = self._handle_movement(agent_id, action_name, x, y)
        elif 'EAT' in action_name or 'COLLECT' in action_name:
            outcome = self._handle_resource_consumption(agent_id, x, y)
        else:
            outcome['success'] = True

        return outcome

    def _handle_movement(self, agent_id: int, action_name: str, x: int, y: int) -> Dict[str, Any]:
        """Handle agent movement action."""
        dx, dy = 0, 0
        if 'NORTH' in action_name:
            dy = -1
        elif 'SOUTH' in action_name:
            dy = 1
        elif 'EAST' in action_name:
            dx = 1
        elif 'WEST' in action_name:
            dx = -1

        new_x, new_y = x + dx, y + dy

        # Boundary check
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            return {'success': False, 'reward': -0.1, 'info': {'blocked': 'boundary'}}

        # Obstacle check
        if self.terrain_grid[new_y, new_x] == TerrainType.OBSTACLE.value:
            return {'success': False, 'reward': -0.1, 'info': {'blocked': 'obstacle'}}

        # Move successful
        self.agent_positions[agent_id] = (new_x, new_y)
        return {'success': True, 'reward': 0.0, 'info': {'new_position': (new_x, new_y)}}

    def _handle_resource_consumption(self, agent_id: int, x: int, y: int) -> Dict[str, Any]:
        """Handle agent consuming resources at their location."""
        if self.resource_grid[y, x] > 0:
            consumed = min(0.2, self.resource_grid[y, x])
            self.resource_grid[y, x] -= consumed
            return {'success': True, 'reward': consumed, 'info': {'consumed': consumed}}
        return {'success': False, 'reward': 0.0, 'info': {'no_resources': True}}

    def _update_resources(self):
        """Update resource regeneration."""
        for patch in self.resource_patches:
            if patch.amount < patch.max_amount:
                patch.amount = min(patch.max_amount, patch.amount + patch.regeneration_rate)
                self._apply_resource_patch(patch)

    def _update_signals(self):
        """Propagate wisdom signals."""
        self.signal_grid.propagate(dt=1.0)

    def _process_stochastic_events(self):
        """Process random environmental events."""
        self.active_events = []

        if np.random.random() < self.config.stochastic_event_probability:
            event = self._trigger_random_event()
            if event != StochasticEvent.NONE:
                self.active_events.append(event)
                self.event_history.append({
                    'time_step': self.time_step,
                    'event': event.name,
                    'details': self._get_event_details(event)
                })

    def _trigger_random_event(self) -> StochasticEvent:
        """Trigger a random environmental event."""
        events = [
            StochasticEvent.RESOURCE_BLOOM,
            StochasticEvent.RESOURCE_DEPLETION,
            StochasticEvent.DISASTER,
            StochasticEvent.WEATHER_CHANGE
        ]
        event = np.random.choice(events)

        if event == StochasticEvent.RESOURCE_BLOOM:
            # Add extra resources
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            self.resource_grid[max(0, y - 3):min(self.height, y + 4),
                               max(0, x - 3):min(self.width, x + 4)] += 0.3
            np.clip(self.resource_grid, 0, 1, out=self.resource_grid)

        elif event == StochasticEvent.RESOURCE_DEPLETION:
            # Remove resources in an area
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            self.resource_grid[max(0, y - 3):min(self.height, y + 4),
                               max(0, x - 3):min(self.width, x + 4)] *= 0.5

        elif event == StochasticEvent.DISASTER:
            # Emit danger signals
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            self.signal_grid.add_signal(WisdomSignalType.DANGER_WARNING, x, y, 1.0)

        logger.debug(f"Stochastic event triggered: {event.name} at step {self.time_step}")
        return event

    def _get_event_details(self, event: StochasticEvent) -> Dict[str, Any]:
        """Get details about an environmental event."""
        return {
            'event_type': event.name,
            'time_step': self.time_step
        }

    def get_observation(self, agent_id: int, radius: int = 5) -> Dict[str, Any]:
        """
        Return local observation for an agent.

        Args:
            agent_id: ID of the agent requesting observation
            radius: Observation radius

        Returns:
            Dictionary containing local environment information
        """
        if agent_id not in self.agent_positions:
            return {'error': 'Agent not found'}

        x, y = self.agent_positions[agent_id]

        # Get local resource view
        x_min, x_max = max(0, x - radius), min(self.width, x + radius + 1)
        y_min, y_max = max(0, y - radius), min(self.height, y + radius + 1)

        local_resources = self.resource_grid[y_min:y_max, x_min:x_max].copy()
        local_terrain = self.terrain_grid[y_min:y_max, x_min:x_max].copy()

        # Get local signals
        local_signals = self.signal_grid.get_local(x, y, radius)

        # Get nearby agents
        nearby_agents = []
        for other_id, (ox, oy) in self.agent_positions.items():
            if other_id != agent_id:
                distance = np.sqrt((ox - x) ** 2 + (oy - y) ** 2)
                if distance <= radius:
                    nearby_agents.append({
                        'id': other_id,
                        'distance': distance,
                        'relative_position': (ox - x, oy - y)
                    })

        return {
            'position': (x, y),
            'local_resources': local_resources,
            'local_terrain': local_terrain,
            'local_signals': local_signals,
            'nearby_agents': nearby_agents,
            'time_step': self.time_step,
            'active_events': [e.name for e in self.active_events]
        }

    def add_resource(self, x: int, y: int, amount: float):
        """Add resources at a specific location."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.resource_grid[y, x] = min(1.0, self.resource_grid[y, x] + amount)

    def trigger_event(self, event_type: StochasticEvent, x: Optional[int] = None, y: Optional[int] = None):
        """Manually trigger an environmental event."""
        if x is None:
            x = np.random.randint(0, self.width)
        if y is None:
            y = np.random.randint(0, self.height)

        self.active_events.append(event_type)
        logger.info(f"Manual event triggered: {event_type.name} at ({x}, {y})")

    def register_agent(self, agent_id: int, x: int, y: int, agent_ref: Optional[Any] = None):
        """Register an agent in the environment."""
        self.agent_positions[agent_id] = (x, y)
        if agent_ref is not None:
            self.agents[agent_id] = agent_ref

    def unregister_agent(self, agent_id: int):
        """Remove an agent from the environment."""
        if agent_id in self.agent_positions:
            del self.agent_positions[agent_id]
        if agent_id in self.agents:
            del self.agents[agent_id]

    def is_passable(self, x: int, y: int) -> bool:
        """Check if a location is passable."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return self.terrain_grid[y, x] != TerrainType.OBSTACLE.value

    def get_cell_info(self, x: int, y: int) -> Dict[str, Any]:
        """Get detailed information about a specific cell."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return {'error': 'Out of bounds'}

        return {
            'terrain': TerrainType(self.terrain_grid[y, x]).name,
            'resources': float(self.resource_grid[y, x]),
            'signals': self.signal_grid.get_local(x, y, radius=0),
            'passable': self.is_passable(x, y)
        }

    def get_state(self) -> EnvironmentState:
        """Get complete environment state."""
        return EnvironmentState(
            time_step=self.time_step,
            resource_grid=self.resource_grid.copy(),
            signal_layers={k: v.copy() for k, v in self.signal_grid.layers.items()},
            terrain_grid=self.terrain_grid.copy(),
            agent_positions=self.agent_positions.copy(),
            active_events=self.active_events.copy(),
            metrics=self.get_metrics()
        )

    def get_metrics(self) -> Dict[str, float]:
        """Get environment metrics."""
        signal_stats = self.signal_grid.get_network_stats()

        return {
            'total_resources': float(np.sum(self.resource_grid)),
            'resource_coverage': float(np.mean(self.resource_grid > 0.1)),
            'signal_diversity': signal_stats.get('signal_diversity', 0.0),
            'network_coherence': signal_stats.get('network_coherence', 0.0),
            'num_agents': len(self.agent_positions),
            'time_step': self.time_step
        }

    def reset(self):
        """Reset environment to initial state."""
        self.time_step = 0
        self.resource_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.terrain_grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.signal_grid = WisdomSignalGrid(
            self.width, self.height,
            base_diffusion=self.config.signal_diffusion_rate,
            base_decay=self.config.signal_decay_rate
        )
        self.resource_patches = []
        self.agents = {}
        self.agent_positions = {}
        self.active_events = []
        self.event_history = []

        self._initialize_terrain()
        self._initialize_resources()

        logger.info("Environment reset")

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid and passable."""
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.is_passable(x, y)

    def get_random_valid_position(self) -> Tuple[int, int]:
        """Get a random valid position in the environment."""
        for _ in range(100):  # Max attempts
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.is_passable(x, y):
                return (x, y)
        # Fallback to center if no valid position found
        return (self.width // 2, self.height // 2)

    @property
    def grid_size(self) -> Tuple[int, int]:
        """Get the grid size as (width, height)."""
        return (self.width, self.height)

    def get_local_view(self, position: Tuple[int, int], radius: int = 2) -> np.ndarray:
        """Get local terrain view around a position."""
        x, y = position
        size = 2 * radius + 1
        view = np.zeros((size, size, 3), dtype=np.float32)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                vx, vy = dx + radius, dy + radius

                if 0 <= nx < self.width and 0 <= ny < self.height:
                    view[vx, vy, 0] = self.terrain_grid[ny, nx] / 4.0  # Normalized terrain
                    view[vx, vy, 1] = self.resource_grid[ny, nx]  # Resource amount
                    view[vx, vy, 2] = 1.0 if self.is_passable(nx, ny) else 0.0

        return view

    def get_local_signals(self, position: Tuple[int, int], radius: int = 2) -> np.ndarray:
        """Get local signal values around a position."""
        x, y = position
        size = 2 * radius + 1
        num_signals = len(WisdomSignalType)
        signals = np.zeros((size, size, num_signals), dtype=np.float32)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                vx, vy = dx + radius, dy + radius

                if 0 <= nx < self.width and 0 <= ny < self.height:
                    local_signals = self.signal_grid.get_local(nx, ny, radius=0)
                    for i, signal_type in enumerate(WisdomSignalType):
                        signals[vx, vy, i] = local_signals.get(signal_type, 0.0)

        return signals

    def emit_signal(self, position: Tuple[int, int], signal_type: int, intensity: float):
        """Emit a wisdom signal at a position."""
        x, y = position
        if 0 <= signal_type < len(WisdomSignalType):
            signal = list(WisdomSignalType)[signal_type]
            self.signal_grid.add_signal(signal, x, y, intensity)

    def propagate_signals(self):
        """Propagate all signals one time step."""
        self.signal_grid.propagate()

    def get_resource_at(self, position: Tuple[int, int]) -> float:
        """Get resource amount at a position."""
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.resource_grid[y, x])
        return 0.0

    def harvest_resource(self, position: Tuple[int, int]) -> float:
        """Harvest resource at a position."""
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            amount = float(self.resource_grid[y, x])
            self.resource_grid[y, x] = 0.0
            return amount
        return 0.0

    def regenerate_resources(self):
        """Regenerate resources based on configuration."""
        for patch in self.resource_patches:
            self._apply_resource_patch(patch)

    def count_resources(self) -> int:
        """Count total resources in environment."""
        return int(np.sum(self.resource_grid > 0))

    def mark_entity(self, position: Tuple[int, int], entity_type: str):
        """Mark a position with an entity type (for scenarios)."""
        # Store entity markers for scenario use
        if not hasattr(self, 'entity_markers'):
            self.entity_markers = {}
        self.entity_markers[position] = entity_type

    def apply_damage(self, position: Tuple[int, int], severity: float):
        """Apply damage to a cell (for disaster scenarios)."""
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            self.resource_grid[y, x] *= (1.0 - severity)

    def restore_cell(self, position: Tuple[int, int]):
        """Restore a damaged cell."""
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            self.resource_grid[y, x] = min(1.0, self.resource_grid[y, x] + 0.5)

    def place_pattern(self, x: int, y: int, pattern: np.ndarray, pattern_id: int):
        """Place a pattern in the environment (for concept emergence)."""
        if not hasattr(self, 'patterns'):
            self.patterns = {}
        self.patterns[pattern_id] = {'position': (x, y), 'pattern': pattern}

        # Apply pattern to resource grid
        h, w = pattern.shape
        for dx in range(w):
            for dy in range(h):
                px, py = x + dx, y + dy
                if 0 <= px < self.width and 0 <= py < self.height:
                    if pattern[dy, dx] > 0:
                        self.resource_grid[py, px] = pattern[dy, dx]

    def get_pattern_position(self, pattern_id: int) -> Optional[Tuple[int, int]]:
        """Get position of a placed pattern."""
        if hasattr(self, 'patterns') and pattern_id in self.patterns:
            return self.patterns[pattern_id]['position']
        return None
