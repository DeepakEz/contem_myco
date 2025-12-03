#!/usr/bin/env python3
"""
MycoNet++ Contemplative Simulation - Enhanced & Clean Implementation
================================================================

A comprehensive, modular implementation of the contemplative MycoNet system
featuring real module detection, advanced analytics, visualization, and 
interactive monitoring capabilities.

Features:
- Automatic real/fallback module detection
- Advanced analytics and insights generation
- Real-time visualization system
- Interactive monitoring and control
- Comprehensive error handling and logging
"""

import argparse
import logging
import json
import time
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from enum import Enum

import numpy as np

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the simulation"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Set console encoding to handle Unicode properly
    try:
        console_handler.stream.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        # Fallback for older Python versions or systems that don't support UTF-8
        pass
    
    file_handler = logging.FileHandler('contemplative_simulation.log', encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.handlers.clear()  # Clear any existing handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

# ==============================================================================
# CORE ENUMS AND TYPES
# ==============================================================================

class ContemplativeState(Enum):
    """States of contemplative awareness"""
    ORDINARY = "ordinary"
    MINDFUL = "mindful"
    DEEP_CONTEMPLATION = "deep_contemplation"
    COLLECTIVE_MEDITATION = "collective_meditation"
    WISDOM_SHARING = "wisdom_sharing"

class WisdomType(Enum):
    """Types of wisdom insights"""
    PRACTICAL = "practical"
    ETHICAL = "ethical"
    COMPASSIONATE = "compassionate"
    INSIGHT = "insight"

class WisdomSignalType(Enum):
    """Types of wisdom signals in the network"""
    WISDOM_BEACON = "wisdom_beacon"
    MINDFULNESS_WAVE = "mindfulness_wave"
    COMPASSION_GRADIENT = "compassion_gradient"
    COLLECTIVE_MEDITATION = "collective_meditation"

# ==============================================================================
# CONFIGURATION CLASSES
# ==============================================================================

@dataclass
class ContemplativeConfig:
    """Configuration for contemplative processing"""
    enable_contemplative_processing: bool = True
    mindfulness_update_frequency: int = 20
    wisdom_signal_strength: float = 0.3
    collective_meditation_threshold: float = 0.8
    ethical_reasoning_depth: int = 1
    contemplative_memory_capacity: int = 100
    wisdom_sharing_radius: int = 1
    compassion_sensitivity: float = 0.4

@dataclass
class WisdomSignalConfig:
    """Configuration for wisdom signal system"""
    decay_rate: float = 0.05
    propagation_speed: float = 1.0
    max_intensity: float = 1.0
    signal_radius: float = 3.0

@dataclass
class VisualizationConfig:
    """Configuration for visualization system"""
    enable_visualization: bool = False
    update_interval: int = 10
    save_frames: bool = False
    show_wisdom_signals: bool = True
    show_agent_states: bool = True
    color_scheme: str = "contemplative"  # contemplative, scientific, artistic

@dataclass
class SimulationConfig:
    """Main configuration for the contemplative simulation"""
    # Environment settings
    environment_width: int = 50
    environment_height: int = 50
    initial_population: int = 20
    max_population: int = 100

    # Simulation settings
    max_steps: int = 1000
    save_interval: int = 100
    visualization_interval: int = 50

    # Contemplative settings
    contemplative_config: ContemplativeConfig = None
    wisdom_signal_config: WisdomSignalConfig = None
    visualization_config: VisualizationConfig = None

    # Overmind settings
    enable_overmind: bool = True
    overmind_intervention_frequency: int = 10

    # Agent brain settings
    brain_input_size: int = 16
    brain_hidden_size: int = 64
    brain_output_size: int = 8

    # Experiment settings
    experiment_name: str = "contemplative_basic"
    output_directory: str = "contemplative_results"

    # Tracking settings
    track_wisdom_propagation: bool = True
    track_collective_behavior: bool = True
    track_ethical_decisions: bool = True

    # Advanced features
    enable_advanced_analytics: bool = True
    enable_real_time_monitoring: bool = False
    enable_interactive_mode: bool = False

    def __init__(self, **kwargs):
        """Initialize configuration from keyword arguments"""
        # Set default values
        self.environment_width = 50
        self.environment_height = 50
        self.initial_population = 20
        self.max_population = 100
        self.max_steps = 1000
        self.save_interval = 100
        self.visualization_interval = 50
        self.contemplative_config = None
        self.wisdom_signal_config = None
        self.visualization_config = None
        self.enable_overmind = True
        self.overmind_intervention_frequency = 10
        self.brain_input_size = 16
        self.brain_hidden_size = 64
        self.brain_output_size = 8
        self.experiment_name = "contemplative_basic"
        self.output_directory = "contemplative_results"
        self.track_wisdom_propagation = True
        self.track_collective_behavior = True
        self.track_ethical_decisions = True
        self.enable_advanced_analytics = True
        self.enable_real_time_monitoring = False
        self.enable_interactive_mode = False

        # Override with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Initialize nested configs
        if self.contemplative_config is None:
            self.contemplative_config = ContemplativeConfig()
        elif isinstance(self.contemplative_config, dict):
            self.contemplative_config = ContemplativeConfig(**self.contemplative_config)

        if self.wisdom_signal_config is None:
            self.wisdom_signal_config = WisdomSignalConfig()
        elif isinstance(self.wisdom_signal_config, dict):
            self.wisdom_signal_config = WisdomSignalConfig(**self.wisdom_signal_config)

        if self.visualization_config is None:
            self.visualization_config = VisualizationConfig()
        elif isinstance(self.visualization_config, dict):
            self.visualization_config = VisualizationConfig(**self.visualization_config)

# ==============================================================================
# FALLBACK IMPLEMENTATIONS
# ==============================================================================

class FallbackWisdomSignalGrid:
    """Fallback implementation of wisdom signal grid"""
    
    def __init__(self, width: int, height: int, config: WisdomSignalConfig):
        self.width = width
        self.height = height
        self.config = config
        self.signals = np.zeros((height, width))
        
    def update_all_signals(self):
        """Update and decay all signals"""
        self.signals *= (1.0 - self.config.decay_rate)
        
    def add_signal(self, signal_type: WisdomSignalType, x: int, y: int, 
                  intensity: float, source_agent_id: int):
        """Add a signal to the grid"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.signals[y, x] = min(self.config.max_intensity, 
                                   self.signals[y, x] + intensity)
    
    def get_network_stats(self) -> Dict[str, float]:
        """Get network statistics"""
        return {
            'signal_diversity': float(np.std(self.signals)),
            'network_coherence': float(np.mean(self.signals)),
            'wisdom_flow_efficiency': float(np.sum(self.signals > 0.1) / self.signals.size),
            'total_signals': float(np.sum(self.signals)),
            'active_signals': float(np.sum(self.signals > 0.1))
        }
    
    def trigger_network_meditation(self, center_x: int, center_y: int, 
                                 radius: int, intensity: float):
        """Trigger network-wide meditation"""
        y, x = np.ogrid[:self.height, :self.width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        self.signals[mask] = np.minimum(self.config.max_intensity, 
                                      self.signals[mask] + intensity)
    
    def amplify_signal_type(self, signal_type: WisdomSignalType, factor: float):
        """Amplify signals of a specific type"""
        self.signals *= factor
        np.clip(self.signals, 0, self.config.max_intensity, out=self.signals)

class FallbackContemplativeProcessor:
    """Fallback contemplative processor"""
    
    def __init__(self):
        self.mindfulness_level = 0.5
        self.mindfulness_monitor = self
        
    def get_mindfulness_score(self) -> float:
        """Get current mindfulness score"""
        return self.mindfulness_level
        
    def get_state_summary(self) -> Dict[str, Any]:
        """Get state summary"""
        return {
            'mindfulness_level': self.mindfulness_level,
            'wisdom_insights_count': 0,
            'contemplation_depth': 0,
            'average_wisdom_intensity': 0.0
        }

class FallbackContemplativeAgent:
    """Fallback implementation of contemplative agent"""
    
    def __init__(self, agent_id: int, x: int, y: int, config: Dict[str, Any]):
        # Basic attributes
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.alive = True
        self.energy = config.get('initial_energy', 1.0)
        self.health = config.get('initial_health', 1.0)
        self.age = 0
        self.generation = 0
        
        # Contemplative attributes
        self.contemplative_state = ContemplativeState.ORDINARY
        self.collective_harmony_level = 0.5
        self.decisions_made = 0
        self.ethical_decisions = 0
        self.wisdom_insights_generated = 0
        self.wisdom_insights_received = 0
        
        # Components
        self.contemplative_processor = FallbackContemplativeProcessor()
        self.brain = self._create_simple_brain()
        
    def _create_simple_brain(self):
        """Create simple brain for decision making"""
        class SimpleBrain:
            def __init__(self):
                self.last_contemplative_info = {}
                
            def get_summary(self):
                return {
                    'last_contemplative_info': self.last_contemplative_info,
                    'mindfulness_level': 0.5,
                    'ethical_score': 0.5,
                    'wisdom_insights_used': 0
                }
        return SimpleBrain()
    
    def update(self, observations: Dict[str, Any], available_actions: List[str]) -> str:
        """Update agent and return chosen action"""
        self.age += 1
        self.decisions_made += 1
        
        # Energy decay
        self.energy = max(0.0, self.energy - 0.01)
        if self.energy <= 0:
            self.alive = False
            return 'rest'
        
        # Enhanced decision making with ethical considerations
        if self.energy < 0.3:
            return 'eat_food'
        elif self.health < 0.4:
            return 'drink_water'
        elif observations.get('other_agents_distress', 0) > 0.4 and self.energy > 0.5:
            # Ethical decision: help others in distress
            self.ethical_decisions += 1
            return 'contemplate'  # Generate compassion signals
        elif np.random.random() < 0.15:  # Increased contemplation rate
            return 'contemplate'
        else:
            movements = ['move_north', 'move_south', 'move_east', 'move_west']
            return np.random.choice(movements)
    
    def move(self, dx: int, dy: int, environment):
        """Move agent within environment bounds"""
        new_x = max(0, min(environment.width - 1, self.x + dx))
        new_y = max(0, min(environment.height - 1, self.y + dy))
        self.x, self.y = new_x, new_y
    
    def reproduce(self):
        """Create offspring"""
        if self.energy < 0.5:
            return None
        
        offspring = FallbackContemplativeAgent(
            agent_id=self.agent_id + 1000,
            x=self.x + np.random.randint(-1, 2),
            y=self.y + np.random.randint(-1, 2),
            config={'initial_energy': 1.0, 'initial_health': 1.0}
        )
        
        offspring.generation = self.generation + 1
        self.energy -= 0.3
        return offspring
    
    def receive_overmind_guidance(self, guidance_type: str, intensity: float):
        """Receive guidance from overmind"""
        if guidance_type == 'cooperation':
            self.collective_harmony_level = min(1.0, self.collective_harmony_level + intensity * 0.1)
        elif guidance_type == 'resource_gathering':
            self.energy = min(1.0, self.energy + intensity * 0.05)
    
    def set_wisdom_signal_processor(self, wisdom_grid):
        """Set wisdom signal processor"""
        self.wisdom_grid = wisdom_grid
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get agent state as dictionary"""
        return {
            'agent_id': self.agent_id,
            'x': self.x, 'y': self.y,
            'energy': self.energy, 'health': self.health,
            'age': self.age, 'generation': self.generation,
            'alive': self.alive,
            'contemplative_state': self.contemplative_state.value,
            'collective_harmony_level': self.collective_harmony_level,
            'decisions_made': self.decisions_made,
            'ethical_decisions': self.ethical_decisions,
            'wisdom_insights_generated': self.wisdom_insights_generated,
            'wisdom_insights_received': self.wisdom_insights_received
        }

class FallbackContemplativeColony:
    """Fallback implementation of contemplative colony"""
    
    def __init__(self, agents: List[FallbackContemplativeAgent], wisdom_grid):
        self.agents = agents
        self.wisdom_grid = wisdom_grid
        
    def update_collective_state(self):
        """Update collective state"""
        pass
        
    def get_colony_metrics(self) -> Dict[str, Any]:
        """Get colony-level metrics"""
        living_agents = [a for a in self.agents if a.alive]
        if not living_agents:
            return {'population': 0, 'collective_wisdom_level': 0.0, 'network_coherence': 0.0}
        
        return {
            'population': len(living_agents),
            'collective_wisdom_level': np.mean([a.wisdom_insights_generated for a in living_agents]),
            'network_coherence': np.mean([a.collective_harmony_level for a in living_agents])
        }

class FallbackContemplativeOvermind:
    """Fallback implementation of contemplative overmind"""
    
    def __init__(self, colony_size: int, environment_size: Tuple[int, int], config: Dict[str, Any]):
        self.colony_size = colony_size
        self.environment_size = environment_size
        self.config = config
        self._decisions_made = 0
        self._meditations_triggered = 0
    
    def get_intervention_action(self, agents, environment, wisdom_signal_grid):
        """Get intervention action based on current state"""
        self._decisions_made += 1
        living_agents = [a for a in agents if a.alive]
        
        # Intervention logic
        if len(living_agents) < 5:
            return {
                'action_type': 'collective_guidance',
                'parameters': {'guidance_type': 'cooperation', 'intensity': 0.8},
                'target_agents': [a.agent_id for a in living_agents]
            }
        
        suffering_agents = [a for a in living_agents if a.energy < 0.3 or a.health < 0.4]
        if len(suffering_agents) > len(living_agents) * 0.3:
            self._meditations_triggered += 1
            return {
                'action_type': 'network_meditation',
                'parameters': {
                    'center_x': environment.width // 2,
                    'center_y': environment.height // 2,
                    'radius': 8, 'intensity': 0.7
                },
                'target_agents': []
            }
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'decisions_made': self._decisions_made,
            'success_rate': 0.75,
            'collective_meditations_triggered': self._meditations_triggered
        }
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get state dictionary"""
        return {
            'decisions_made': self._decisions_made,
            'meditations_triggered': self._meditations_triggered
        }

# ==============================================================================
# IMPORT DIAGNOSTICS AND MODULE LOADING
# ==============================================================================

def attempt_real_imports():
    """Attempt to import real contemplative modules and return status"""
    import_status = {
        'core': False,
        'signals': False, 
        'entities': False,
        'overmind': False,
        'modules_loaded': {}
    }
    
    try:
        from myconet_contemplative_core import ContemplativeState as RealContemplativeState, WisdomType as RealWisdomType
        import_status['core'] = True
        import_status['modules_loaded']['ContemplativeState'] = RealContemplativeState
        import_status['modules_loaded']['WisdomType'] = RealWisdomType
        logger.info("Successfully imported real contemplative core module")
    except ImportError as e:
        logger.warning(f"Real contemplative core not available: {e}")
        import_status['modules_loaded']['ContemplativeState'] = ContemplativeState
        import_status['modules_loaded']['WisdomType'] = WisdomType
    
    try:
        from myconet_wisdom_signals import WisdomSignalGrid as RealWisdomSignalGrid, WisdomSignalConfig as RealWisdomSignalConfig, WisdomSignalType as RealWisdomSignalType
        import_status['signals'] = True
        import_status['modules_loaded']['WisdomSignalGrid'] = RealWisdomSignalGrid
        import_status['modules_loaded']['WisdomSignalConfig'] = RealWisdomSignalConfig
        import_status['modules_loaded']['WisdomSignalType'] = RealWisdomSignalType
        logger.info("Successfully imported real wisdom signals module")
    except ImportError as e:
        logger.warning(f"Real wisdom signals not available: {e}")
        import_status['modules_loaded']['WisdomSignalGrid'] = FallbackWisdomSignalGrid
        import_status['modules_loaded']['WisdomSignalConfig'] = WisdomSignalConfig
        import_status['modules_loaded']['WisdomSignalType'] = WisdomSignalType
    
    try:
        from myconet_contemplative_entities import ContemplativeNeuroAgent as RealContemplativeAgent, ContemplativeColony as RealContemplativeColony, ContemplativeConfig as RealContemplativeConfig
        import_status['entities'] = True
        import_status['modules_loaded']['ContemplativeNeuroAgent'] = RealContemplativeAgent
        import_status['modules_loaded']['ContemplativeColony'] = RealContemplativeColony
        import_status['modules_loaded']['ContemplativeConfig'] = RealContemplativeConfig
        logger.info("Successfully imported real contemplative entities module")
    except ImportError as e:
        logger.warning(f"Real contemplative entities not available: {e}")
        import_status['modules_loaded']['ContemplativeNeuroAgent'] = FallbackContemplativeAgent
        import_status['modules_loaded']['ContemplativeColony'] = FallbackContemplativeColony
        import_status['modules_loaded']['ContemplativeConfig'] = ContemplativeConfig
    
    try:
        from myconet_contemplative_overmind import ContemplativeOvermind as RealContemplativeOvermind
        import_status['overmind'] = True
        import_status['modules_loaded']['ContemplativeOvermind'] = RealContemplativeOvermind
        logger.info("Successfully imported real contemplative overmind module")
    except (ImportError, SyntaxError) as e:
        logger.warning(f"Real contemplative overmind not available: {e}")
        import_status['modules_loaded']['ContemplativeOvermind'] = FallbackContemplativeOvermind
    
    return import_status

# Attempt to load real modules and store status
MODULE_STATUS = attempt_real_imports()

# Create class aliases for easier access
WisdomSignalGrid = MODULE_STATUS['modules_loaded']['WisdomSignalGrid']
ContemplativeNeuroAgent = MODULE_STATUS['modules_loaded']['ContemplativeNeuroAgent']
ContemplativeColony = MODULE_STATUS['modules_loaded']['ContemplativeColony']
ContemplativeOvermind = MODULE_STATUS['modules_loaded']['ContemplativeOvermind']

# ==============================================================================
# ENVIRONMENT SYSTEM
# ==============================================================================

class ContemplativeEnvironment:
    """Environment for contemplative agents"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._initialize_grids()
        
        # Environmental parameters
        self.resource_regeneration_rate = 0.01
        self.hazard_movement_rate = 0.005
    
    def _initialize_grids(self):
        """Initialize resource and hazard grids"""
        self.food_grid = np.random.random((self.height, self.width)) * 0.5
        self.water_grid = np.random.random((self.height, self.width)) * 0.3
        self.hazard_grid = np.zeros((self.height, self.width))
        
        # Add some hazards
        num_hazards = self.width * self.height // 20
        for _ in range(num_hazards):
            hx, hy = np.random.randint(0, self.width), np.random.randint(0, self.height)
            self.hazard_grid[hy, hx] = np.random.uniform(0.3, 0.8)
    
    def get_local_observations(self, x: int, y: int, radius: int = 2) -> Dict[str, Any]:
        """Get local environment observations for an agent"""
        observations = {
            'x': x, 'y': y,
            'food_nearby': 0.0, 'water_nearby': 0.0, 'danger_level': 0.0,
            'safe_directions': [], 'resource_directions': []
        }
        
        # Check surrounding area
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    distance = max(abs(dx), abs(dy))
                    if distance == 0:
                        continue
                    
                    weight = 1.0 / distance
                    
                    # Accumulate observations
                    observations['food_nearby'] += self.food_grid[ny, nx] * weight
                    observations['water_nearby'] += self.water_grid[ny, nx] * weight
                    observations['danger_level'] += self.hazard_grid[ny, nx] * weight
                    
                    # Track directions
                    if self.food_grid[ny, nx] > 0.3 or self.water_grid[ny, nx] > 0.3:
                        observations['resource_directions'].append((dx, dy))
                    if self.hazard_grid[ny, nx] < 0.2:
                        observations['safe_directions'].append((dx, dy))
        
        # Normalize observations
        observations['food_nearby'] = min(observations['food_nearby'], 1.0)
        observations['water_nearby'] = min(observations['water_nearby'], 1.0)
        observations['danger_level'] = min(observations['danger_level'], 1.0)
        
        return observations
    
    def update(self):
        """Update environment state"""
        # Regenerate resources
        self.food_grid += np.random.random((self.height, self.width)) * self.resource_regeneration_rate
        self.water_grid += np.random.random((self.height, self.width)) * self.resource_regeneration_rate
        
        # Cap resources
        self.food_grid = np.clip(self.food_grid, 0.0, 1.0)
        self.water_grid = np.clip(self.water_grid, 0.0, 1.0)
        
        # Move hazards occasionally
        if np.random.random() < self.hazard_movement_rate:
            hx, hy = np.random.randint(0, self.width), np.random.randint(0, self.height)
            self.hazard_grid[hy, hx] = min(self.hazard_grid[hy, hx] + 0.2, 0.8)
            self.hazard_grid *= 0.99
    
    def consume_resource(self, x: int, y: int, resource_type: str, amount: float = 0.1) -> float:
        """Consume resources at location"""
        if 0 <= x < self.width and 0 <= y < self.height:
            if resource_type == 'food':
                consumed = min(self.food_grid[y, x], amount)
                self.food_grid[y, x] -= consumed
                return consumed
            elif resource_type == 'water':
                consumed = min(self.water_grid[y, x], amount)
                self.water_grid[y, x] -= consumed
                return consumed
        return 0.0

# ==============================================================================
# VISUALIZATION SYSTEM
# ==============================================================================

class ContemplativeVisualizer:
    """Visualization system for contemplative simulation"""
    
    def __init__(self, config: VisualizationConfig, environment_size: Tuple[int, int]):
        self.config = config
        self.environment_size = environment_size
        self.frame_count = 0
        
        # Try to import matplotlib
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from matplotlib.colors import LinearSegmentedColormap
            self.plt = plt
            self.animation = animation
            self.available = True
            
            # Setup figure
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
            self.fig.suptitle('Contemplative MycoNet Simulation', fontsize=16)
            
            # Setup color schemes
            self._setup_color_schemes()
            
        except ImportError:
            logger.warning("Matplotlib not available - visualization disabled")
            self.available = False
    
    def _setup_color_schemes(self):
        """Setup color schemes for different visualization modes"""
        if self.config.color_scheme == "contemplative":
            # Peaceful, meditative colors
            self.colors = {
                'agent_high_energy': '#4CAF50',      # Green
                'agent_medium_energy': '#FFC107',     # Amber  
                'agent_low_energy': '#F44336',        # Red
                'wisdom_signal': '#9C27B0',           # Purple
                'meditation': '#3F51B5',              # Indigo
                'background': '#E8F5E8'               # Light green
            }
        elif self.config.color_scheme == "scientific":
            # Traditional scientific visualization
            self.colors = {
                'agent_high_energy': '#2196F3',       # Blue
                'agent_medium_energy': '#FF9800',     # Orange
                'agent_low_energy': '#F44336',        # Red
                'wisdom_signal': '#9C27B0',           # Purple
                'meditation': '#607D8B',              # Blue Grey
                'background': '#FAFAFA'               # Light grey
            }
        else:  # artistic
            # Creative, vibrant colors
            self.colors = {
                'agent_high_energy': '#00BCD4',       # Cyan
                'agent_medium_energy': '#FFEB3B',     # Yellow
                'agent_low_energy': '#E91E63',        # Pink
                'wisdom_signal': '#673AB7',           # Deep Purple
                'meditation': '#009688',              # Teal
                'background': '#FCE4EC'               # Light pink
            }
    
    def update_visualization(self, simulation_state: Dict[str, Any]):
        """Update visualization with current simulation state"""
        if not self.available:
            return
        
        try:
            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()
            
            # Plot 1: Agent positions and states
            self._plot_agent_positions(simulation_state)
            
            # Plot 2: Wisdom signal intensity
            self._plot_wisdom_signals(simulation_state)
            
            # Plot 3: Population metrics over time
            self._plot_population_metrics(simulation_state)
            
            # Plot 4: Network activity
            self._plot_network_activity(simulation_state)
            
            self.plt.tight_layout()
            self.plt.pause(0.01)
            
            # Save frame if requested
            if self.config.save_frames:
                frame_dir = Path("visualization_frames")
                frame_dir.mkdir(exist_ok=True)
                frame_file = frame_dir / f"frame_{self.frame_count:06d}.png"
                self.plt.savefig(frame_file, dpi=100, bbox_inches='tight')
            
            self.frame_count += 1
            
        except Exception as e:
            logger.warning(f"Visualization update failed: {e}")
    
    def _plot_agent_positions(self, state: Dict[str, Any]):
        """Plot agent positions colored by energy level"""
        ax = self.axes[0, 0]
        agents = state.get('agents', [])
        
        if not agents:
            ax.set_title('Agent Positions (No agents)')
            return
        
        # Separate agents by energy level
        high_energy = [(a.x, a.y) for a in agents if a.alive and a.energy > 0.7]
        medium_energy = [(a.x, a.y) for a in agents if a.alive and 0.3 <= a.energy <= 0.7]
        low_energy = [(a.x, a.y) for a in agents if a.alive and a.energy < 0.3]
        
        # Plot agents
        if high_energy:
            x, y = zip(*high_energy)
            ax.scatter(x, y, c=self.colors['agent_high_energy'], s=50, alpha=0.8, label='High Energy')
        if medium_energy:
            x, y = zip(*medium_energy)
            ax.scatter(x, y, c=self.colors['agent_medium_energy'], s=50, alpha=0.8, label='Medium Energy')
        if low_energy:
            x, y = zip(*low_energy)
            ax.scatter(x, y, c=self.colors['agent_low_energy'], s=50, alpha=0.8, label='Low Energy')
        
        ax.set_xlim(0, self.environment_size[0])
        ax.set_ylim(0, self.environment_size[1])
        ax.set_title(f'Agent Positions (Population: {len([a for a in agents if a.alive])})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_wisdom_signals(self, state: Dict[str, Any]):
        """Plot wisdom signal intensity as heatmap"""
        ax = self.axes[0, 1]
        wisdom_grid = state.get('wisdom_signal_grid', None)
        
        if wisdom_grid is not None and hasattr(wisdom_grid, 'signals'):
            im = ax.imshow(wisdom_grid.signals, cmap='Purples', alpha=0.8)
            ax.set_title('Wisdom Signal Intensity')
            self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_title('Wisdom Signals (No data)')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
    
    def _plot_population_metrics(self, state: Dict[str, Any]):
        """Plot population metrics over time"""
        ax = self.axes[1, 0]
        simulation_data = state.get('simulation_data', {})
        pop_data = simulation_data.get('population_data', [])
        
        if not pop_data:
            ax.set_title('Population Metrics (No data)')
            return
        
        steps = [d['step'] for d in pop_data]
        population = [d['total_population'] for d in pop_data]
        avg_energy = [d['average_energy'] for d in pop_data]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(steps, population, 'b-', label='Population', linewidth=2)
        line2 = ax2.plot(steps, avg_energy, 'r-', label='Avg Energy', linewidth=2)
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Population', color='b')
        ax2.set_ylabel('Average Energy', color='r')
        ax.set_title('Population Dynamics')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_network_activity(self, state: Dict[str, Any]):
        """Plot network activity metrics"""
        ax = self.axes[1, 1]
        simulation_data = state.get('simulation_data', {})
        wisdom_data = simulation_data.get('wisdom_data', [])
        
        if not wisdom_data:
            ax.set_title('Network Activity (No data)')
            return
        
        steps = [d['step'] for d in wisdom_data]
        wisdom_generated = [d['total_wisdom_generated'] for d in wisdom_data]
        mindfulness = [d['average_mindfulness'] for d in wisdom_data]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(steps, wisdom_generated, 'g-', label='Wisdom Generated', linewidth=2)
        line2 = ax2.plot(steps, mindfulness, 'm-', label='Avg Mindfulness', linewidth=2)
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Cumulative Wisdom', color='g')
        ax2.set_ylabel('Average Mindfulness', color='m')
        ax.set_title('Contemplative Network Activity')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def close(self):
        """Close visualization"""
        if self.available:
            self.plt.close('all')

# ==============================================================================
# ADVANCED ANALYTICS SYSTEM
# ==============================================================================

class AdvancedAnalytics:
    """Advanced analytics for contemplative simulation data"""
    
    def analyze_simulation_data(self, simulation_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Perform comprehensive analysis of simulation data"""
        try:
            analysis = {
                'emergence_analysis': self._analyze_emergence(simulation_data),
                'wisdom_dynamics': self._analyze_wisdom_dynamics(simulation_data),
                'ethical_evolution': self._analyze_ethical_evolution(simulation_data),
                'network_patterns': self._analyze_network_patterns(simulation_data),
                'insights': self._generate_insights(simulation_data)
            }
            return analysis
        except Exception as e:
            logger.error(f"Advanced analytics failed: {e}")
            return {'error': str(e)}
    
    def _analyze_emergence(self, data: Dict) -> Dict[str, Any]:
        """Analyze emergence indicators"""
        pop_data = data.get('population_data', [])
        wisdom_data = data.get('wisdom_data', [])
        
        if not pop_data or not wisdom_data:
            return {}
        
        # Collective intelligence indicators
        wisdom_growth = []
        for i in range(1, len(wisdom_data)):
            growth = wisdom_data[i]['total_wisdom_generated'] - wisdom_data[i-1]['total_wisdom_generated']
            wisdom_growth.append(growth)
        
        collective_intelligence_score = np.mean(wisdom_growth) if wisdom_growth else 0
        
        # System resilience
        population_variance = np.var([d['total_population'] for d in pop_data])
        resilience_score = 1.0 / (1.0 + population_variance)
        
        return {
            'collective_intelligence_indicators': {
                'collective_intelligence_score': collective_intelligence_score,
                'wisdom_acceleration': np.mean(np.diff(wisdom_growth)) if len(wisdom_growth) > 1 else 0
            },
            'system_resilience': {
                'resilience_score': resilience_score,
                'population_stability': 1.0 - (population_variance / max([d['total_population'] for d in pop_data]))
            }
        }
    
    def _analyze_wisdom_dynamics(self, data: Dict) -> Dict[str, Any]:
        """Analyze wisdom propagation and dynamics"""
        wisdom_data = data.get('wisdom_data', [])
        if not wisdom_data:
            return {}
        
        mindfulness_trend = [d['average_mindfulness'] for d in wisdom_data]
        wisdom_trend = [d['total_wisdom_generated'] for d in wisdom_data]
        
        return {
            'mindfulness_evolution': {
                'initial': mindfulness_trend[0] if mindfulness_trend else 0,
                'final': mindfulness_trend[-1] if mindfulness_trend else 0,
                'peak': max(mindfulness_trend) if mindfulness_trend else 0,
                'trend_slope': np.polyfit(range(len(mindfulness_trend)), mindfulness_trend, 1)[0] if len(mindfulness_trend) > 1 else 0
            },
            'wisdom_propagation': {
                'total_generated': wisdom_trend[-1] if wisdom_trend else 0,
                'generation_rate': np.mean(np.diff(wisdom_trend)) if len(wisdom_trend) > 1 else 0,
                'exponential_growth': self._detect_exponential_growth(wisdom_trend)
            }
        }
    
    def _analyze_ethical_evolution(self, data: Dict) -> Dict[str, Any]:
        """Analyze ethical behavior evolution"""
        ethical_data = data.get('ethical_data', [])
        if not ethical_data:
            return {}
        
        ethical_ratios = [d['ethical_decision_ratio'] for d in ethical_data]
        harmony_levels = [d['collective_harmony'] for d in ethical_data]
        
        return {
            'ethical_progression': {
                'initial_ratio': ethical_ratios[0] if ethical_ratios else 0,
                'final_ratio': ethical_ratios[-1] if ethical_ratios else 0,
                'improvement': ethical_ratios[-1] - ethical_ratios[0] if len(ethical_ratios) >= 2 else 0,
                'consistency': 1.0 - np.std(ethical_ratios) if ethical_ratios else 0
            },
            'harmony_dynamics': {
                'peak_harmony': max(harmony_levels) if harmony_levels else 0,
                'average_harmony': np.mean(harmony_levels) if harmony_levels else 0,
                'harmony_stability': 1.0 - np.std(harmony_levels) if harmony_levels else 0
            }
        }
    
    def _analyze_network_patterns(self, data: Dict) -> Dict[str, Any]:
        """Analyze network connectivity patterns"""
        network_data = data.get('network_data', [])
        if not network_data:
            return {}
        
        coherence_levels = [d.get('network_coherence', 0) for d in network_data]
        signal_counts = [d.get('total_signals', 0) for d in network_data]
        
        return {
            'coherence_patterns': {
                'peak_coherence': max(coherence_levels) if coherence_levels else 0,
                'coherence_growth': coherence_levels[-1] - coherence_levels[0] if len(coherence_levels) >= 2 else 0,
                'coherence_variance': np.var(coherence_levels) if coherence_levels else 0
            },
            'signal_dynamics': {
                'peak_signals': max(signal_counts) if signal_counts else 0,
                'signal_sustainability': signal_counts[-1] / max(signal_counts) if signal_counts and max(signal_counts) > 0 else 0
            }
        }
    
    def _detect_exponential_growth(self, values: List[float]) -> bool:
        """Detect if there's exponential growth in values"""
        if len(values) < 3:
            return False
        
        # Fit exponential model
        try:
            log_values = [np.log(max(v, 0.01)) for v in values]
            slope, _ = np.polyfit(range(len(log_values)), log_values, 1)
            return slope > 0.1  # Threshold for exponential growth
        except:
            return False
    
    def _generate_insights(self, data: Dict) -> List[str]:
        """Generate human-readable insights"""
        insights = []
        
        # Population insights
        pop_data = data.get('population_data', [])
        if pop_data:
            final_pop = pop_data[-1]['total_population']
            initial_pop = pop_data[0]['total_population']
            
            if final_pop > initial_pop * 1.5:
                insights.append("Population experienced significant growth, indicating successful adaptation")
            elif final_pop < initial_pop * 0.5:
                insights.append("Population declined significantly, suggesting environmental challenges")
            
            # Energy trends
            energy_trend = [d['average_energy'] for d in pop_data]
            if len(energy_trend) > 1:
                energy_slope = np.polyfit(range(len(energy_trend)), energy_trend, 1)[0]
                if energy_slope > 0.001:
                    insights.append("Population energy levels improved over time")
                elif energy_slope < -0.001:
                    insights.append("Population faced increasing energy challenges")
        
        # Wisdom insights
        wisdom_data = data.get('wisdom_data', [])
        if wisdom_data:
            total_wisdom = wisdom_data[-1]['total_wisdom_generated']
            if total_wisdom > len(pop_data) * 0.5:  # More than 0.5 wisdom per step on average
                insights.append("High wisdom generation indicates active contemplative engagement")
            
            mindfulness_levels = [d['average_mindfulness'] for d in wisdom_data]
            if mindfulness_levels and mindfulness_levels[-1] > 0.7:
                insights.append("High final mindfulness suggests successful contemplative development")
        
        # Ethical insights
        ethical_data = data.get('ethical_data', [])
        if ethical_data:
            final_ethical_ratio = ethical_data[-1]['ethical_decision_ratio']
            if final_ethical_ratio > 0.6:
                insights.append("Strong ethical decision-making emerged in the population")
            
            harmony_levels = [d['collective_harmony'] for d in ethical_data]
            if harmony_levels and np.mean(harmony_levels) > 0.6:
                insights.append("High collective harmony indicates successful social coordination")
        
        return insights

# ==============================================================================
# MAIN SIMULATION CLASS
# ==============================================================================

class ContemplativeSimulation:
    """Main simulation orchestrator for contemplative MycoNet"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.step_count = 0
        
        # Initialize systems
        self._initialize_systems()
        self._initialize_data_collection()
        
        # Create output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Contemplative simulation initialized: {config.experiment_name}")
        logger.info(f"Environment: {config.environment_width}x{config.environment_height}")
        logger.info(f"Population: {len(self.agents)} agents")
        logger.info(f"Overmind: {'enabled' if config.enable_overmind else 'disabled'}")
    
    def _initialize_systems(self):
        """Initialize all simulation systems"""
        # Environment
        self.environment = ContemplativeEnvironment(
            self.config.environment_width, 
            self.config.environment_height
        )
        
        # Wisdom signal system
        self.wisdom_signal_grid = WisdomSignalGrid(
            self.config.environment_width, 
            self.config.environment_height, 
            self.config.wisdom_signal_config
        )
        
        # Population
        self.agents = self._create_initial_population()
        self.colony = ContemplativeColony(self.agents, self.wisdom_signal_grid)
        
        # Overmind
        if self.config.enable_overmind:
            self.overmind = ContemplativeOvermind(
                colony_size=self.config.initial_population,
                environment_size=(self.config.environment_width, self.config.environment_height),
                config={
                    'colony_observation_size': 50,
                    'collective_action_size': 10,
                    'wisdom_processing_dim': 128
                }
            )
        else:
            self.overmind = None
        
        # Visualization system
        if self.config.visualization_config.enable_visualization:
            self.visualizer = self._create_visualizer()
        else:
            self.visualizer = None
    
    def _initialize_data_collection(self):
        """Initialize data collection structures"""
        self.simulation_data = {
            'population_data': [],
            'wisdom_data': [],
            'ethical_data': [],
            'network_data': [],
            'overmind_data': []
        }
    
    def _create_initial_population(self) -> List:
        """Create initial population of agents"""
        agents = []
        
        agent_config = {
            'initial_energy': 1.0,
            'initial_health': 1.0,
            'brain_config': {
                'input_size': self.config.brain_input_size,
                'hidden_size': self.config.brain_hidden_size,
                'output_size': self.config.brain_output_size
            },
            'contemplative_config': asdict(self.config.contemplative_config)
        }
        
        for i in range(self.config.initial_population):
            try:
                x = np.random.randint(0, self.config.environment_width)
                y = np.random.randint(0, self.config.environment_height)
                
                agent = ContemplativeNeuroAgent(
                    agent_id=i, x=x, y=y, config=agent_config
                )
                
                # Setup wisdom processor
                if hasattr(agent, 'set_wisdom_signal_processor'):
                    agent.set_wisdom_signal_processor(self.wisdom_signal_grid)
                agents.append(agent)
                
            except Exception as e:
                logger.error(f"Failed to create agent {i}: {e}")
                continue
        
        return agents
    
    def _create_visualizer(self):
        """Create visualization system"""
        try:
            return ContemplativeVisualizer(
                self.config.visualization_config,
                (self.config.environment_width, self.config.environment_height)
            )
        except Exception as e:
            logger.warning(f"Failed to create visualizer: {e}")
            return None
    
    def run_simulation(self):
        """Run the complete simulation"""
        logger.info(f"Starting simulation: {self.config.experiment_name}")
        start_time = time.time()
        
        try:
            for step in range(self.config.max_steps):
                self.step_count = step
                
                # Execute simulation step
                self._execute_simulation_step()
                
                # Data collection and checkpointing
                if step % self.config.save_interval == 0:
                    self._collect_data()
                    self._save_checkpoint()
                
                # Visualization update
                if (self.visualizer and 
                    self.step_count % self.config.visualization_interval == 0):
                    self._update_visualization()
                
                # Real-time monitoring
                if self.config.enable_real_time_monitoring:
                    self._real_time_monitoring()
                
                # Interactive mode
                if self.config.enable_interactive_mode and self.step_count % 50 == 0:
                    self._handle_interactive_input()
                
                # Progress reporting
                if step % 100 == 0:
                    self._report_progress(start_time)
                
                # Early termination check
                if self._should_terminate():
                    logger.info(f"Simulation terminated early at step {step}")
                    break
            
            # Finalize simulation
            self._finalize_simulation()
            
            total_time = time.time() - start_time
            logger.info(f"Simulation completed in {total_time:.1f} seconds")
            
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
            self._emergency_save()
        except Exception as e:
            logger.error(f"Critical simulation error: {e}")
            logger.error(traceback.format_exc())
            self._emergency_save()
            raise
    
    def _execute_simulation_step(self):
        """Execute one simulation step"""
        try:
            # Update environment
            self.environment.update()
            
            # Update wisdom signals
            self.wisdom_signal_grid.update_all_signals()
            
            # Process agent actions
            self._process_agent_actions()
            
            # Update colony
            self.colony.update_collective_state()
            
            # Overmind intervention
            if (self.overmind and 
                self.step_count % self.config.overmind_intervention_frequency == 0):
                self._process_overmind_intervention()
            
            # Population dynamics
            self._process_population_dynamics()
            
        except Exception as e:
            logger.warning(f"Step {self.step_count} error: {e}")
    
    def _process_agent_actions(self):
        """Process actions for all living agents"""
        living_agents = [agent for agent in self.agents if getattr(agent, 'alive', True)]
        
        for agent in living_agents:
            try:
                # Get observations
                env_obs = self.environment.get_local_observations(agent.x, agent.y)
                agent_obs = self._build_agent_observations(agent, env_obs)
                
                # Agent decision
                available_actions = [
                    'move_north', 'move_south', 'move_east', 'move_west',
                    'eat_food', 'drink_water', 'rest', 'contemplate'
                ]
                chosen_action = agent.update(agent_obs, available_actions)
                
                # Execute action
                self._execute_agent_action(agent, chosen_action)
                
            except Exception as e:
                logger.warning(f"Agent {getattr(agent, 'agent_id', 'unknown')} action failed: {e}")
                # Emergency survival behavior
                if hasattr(agent, 'energy'):
                    agent.energy = max(0.0, agent.energy - 0.01)
                    if agent.energy <= 0:
                        agent.alive = False
    
    def _build_agent_observations(self, agent, env_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive observations for an agent"""
        observations = env_obs.copy()
        
        # Internal state
        observations.update({
            'energy': getattr(agent, 'energy', 1.0),
            'health': getattr(agent, 'health', 1.0),
            'age': getattr(agent, 'age', 0) / 1000.0,
            'other_agents_nearby': self._count_nearby_agents(agent),
            'other_agents_distress': self._assess_nearby_distress(agent)
        })
        
        # Contemplative state
        try:
            if hasattr(agent, 'contemplative_processor'):
                contemplative_summary = agent.contemplative_processor.get_state_summary()
                observations.update({
                    'mindfulness_level': contemplative_summary.get('mindfulness_level', 0.0),
                    'contemplative_state_depth': contemplative_summary.get('contemplation_depth', 0),
                    'wisdom_insights_count': contemplative_summary.get('wisdom_insights_count', 0)
                })
            else:
                observations.update({
                    'mindfulness_level': 0.5,
                    'contemplative_state_depth': 0,
                    'wisdom_insights_count': getattr(agent, 'wisdom_insights_generated', 0)
                })
        except Exception:
            observations.update({
                'mindfulness_level': 0.5,
                'contemplative_state_depth': 0,
                'wisdom_insights_count': 0
            })
        
        # Colony state
        try:
            colony_metrics = self.colony.get_colony_metrics()
            observations.update({
                'colony_population': colony_metrics.get('population', 0) / 100.0,
                'collective_wisdom': colony_metrics.get('collective_wisdom_level', 0.0),
                'network_coherence': colony_metrics.get('network_coherence', 0.0)
            })
        except Exception:
            observations.update({
                'colony_population': len([a for a in self.agents if getattr(a, 'alive', True)]) / 100.0,
                'collective_wisdom': 0.0,
                'network_coherence': 0.0
            })
        
        return observations
    
    def _count_nearby_agents(self, agent, radius: int = 3) -> float:
        """Count other agents nearby (normalized)"""
        try:
            count = 0
            agent_x = getattr(agent, 'x', 0)
            agent_y = getattr(agent, 'y', 0)
            
            for other in self.agents:
                if other != agent and getattr(other, 'alive', True):
                    other_x = getattr(other, 'x', 0)
                    other_y = getattr(other, 'y', 0)
                    distance = np.sqrt((agent_x - other_x)**2 + (agent_y - other_y)**2)
                    if distance <= radius:
                        count += 1
            return min(count / 5.0, 1.0)
        except:
            return 0.0
    
    def _assess_nearby_distress(self, agent, radius: int = 4) -> float:
        """Assess distress level of nearby agents"""
        try:
            agent_x = getattr(agent, 'x', 0)
            agent_y = getattr(agent, 'y', 0)
            distress_levels = []
            
            for other in self.agents:
                if other != agent and getattr(other, 'alive', True):
                    other_x = getattr(other, 'x', 0)
                    other_y = getattr(other, 'y', 0)
                    distance = np.sqrt((agent_x - other_x)**2 + (agent_y - other_y)**2)
                    
                    if distance <= radius:
                        other_energy = getattr(other, 'energy', 1.0)
                        other_health = getattr(other, 'health', 1.0)
                        distress = 1.0 - min(other_energy, other_health)
                        if distress > 0.3:
                            distress_levels.append(distress)
            
            return max(distress_levels) if distress_levels else 0.0
        except:
            return 0.0
    
    def _execute_agent_action(self, agent, action: str):
        """Execute an agent's chosen action"""
        try:
            if action == 'move_north':
                if hasattr(agent, 'move'):
                    agent.move(0, -1, self.environment)
                else:
                    agent.y = max(0, agent.y - 1)
            elif action == 'move_south':
                if hasattr(agent, 'move'):
                    agent.move(0, 1, self.environment)
                else:
                    agent.y = min(self.environment.height - 1, agent.y + 1)
            elif action == 'move_east':
                if hasattr(agent, 'move'):
                    agent.move(1, 0, self.environment)
                else:
                    agent.x = min(self.environment.width - 1, agent.x + 1)
            elif action == 'move_west':
                if hasattr(agent, 'move'):
                    agent.move(-1, 0, self.environment)
                else:
                    agent.x = max(0, agent.x - 1)
            elif action == 'eat_food':
                consumed = self.environment.consume_resource(agent.x, agent.y, 'food', 0.2)
                agent.energy = min(1.0, agent.energy + consumed)
            elif action == 'drink_water':
                consumed = self.environment.consume_resource(agent.x, agent.y, 'water', 0.15)
                agent.health = min(1.0, agent.health + consumed * 0.5)
            elif action == 'rest':
                agent.health = min(1.0, agent.health + 0.05)
            elif action == 'contemplate':
                agent.energy = max(0.0, agent.energy - 0.02)
                if np.random.random() < 0.15:
                    if hasattr(agent, 'wisdom_insights_generated'):
                        agent.wisdom_insights_generated += 1
                    # Add wisdom signal to the grid
                    try:
                        signal_type = np.random.choice(list(WisdomSignalType))
                        self.wisdom_signal_grid.add_signal(
                            signal_type, agent.x, agent.y, 0.3, 
                            source_agent_id=getattr(agent, 'agent_id', -1)
                        )
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Action execution failed for agent {getattr(agent, 'agent_id', 'unknown')}: {e}")
    
    def _process_overmind_intervention(self):
        """Process Overmind interventions"""
        if not self.overmind:
            return
        
        try:
            overmind_action = self.overmind.get_intervention_action(
                self.agents, self.environment, self.wisdom_signal_grid
            )
            
            if overmind_action:
                self._execute_overmind_action(overmind_action)
                
        except Exception as e:
            logger.warning(f"Overmind intervention failed: {e}")
    
    def _execute_overmind_action(self, overmind_action):
        """Execute overmind action"""
        try:
            action_type = overmind_action.get('action_type', 'unknown')
            parameters = overmind_action.get('parameters', {})
            target_agents = overmind_action.get('target_agents', [])
            
            logger.info(f"Overmind intervention: {action_type}")
            
            if action_type == 'network_meditation':
                self._trigger_network_meditation(parameters)
            elif action_type == 'wisdom_amplification':
                self._amplify_wisdom_signals(parameters)
            elif action_type == 'collective_guidance':
                self._apply_collective_guidance(target_agents, parameters)
            elif action_type == 'suffering_intervention':
                self._intervene_suffering(parameters)
            else:
                logger.warning(f"Unknown overmind action: {action_type}")
                
        except Exception as e:
            logger.warning(f"Overmind action execution failed: {e}")
    
    def _trigger_network_meditation(self, parameters: Dict[str, Any]):
        """Trigger network-wide meditation"""
        try:
            center_x = parameters.get('center_x', self.config.environment_width // 2)
            center_y = parameters.get('center_y', self.config.environment_height // 2)
            radius = parameters.get('radius', 10)
            intensity = parameters.get('intensity', 0.8)
            
            if hasattr(self.wisdom_signal_grid, 'trigger_network_meditation'):
                self.wisdom_signal_grid.trigger_network_meditation(center_x, center_y, radius, intensity)
            else:
                # Fallback implementation for simple grids
                if hasattr(self.wisdom_signal_grid, 'signals'):
                    y, x = np.ogrid[:self.wisdom_signal_grid.signals.shape[0], :self.wisdom_signal_grid.signals.shape[1]]
                    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                    max_intensity = getattr(self.wisdom_signal_grid.config, 'max_intensity', 1.0)
                    self.wisdom_signal_grid.signals[mask] = np.minimum(max_intensity, 
                                                                      self.wisdom_signal_grid.signals[mask] + intensity)
                    
            logger.info(f"Network meditation triggered at ({center_x}, {center_y})")
        except Exception as e:
            logger.warning(f"Network meditation failed: {e}")
    
    def _amplify_wisdom_signals(self, parameters: Dict[str, Any]):
        """Amplify existing wisdom signals"""
        try:
            amplification_factor = parameters.get('amplification_factor', 1.5)
            signal_types = parameters.get('signal_types', list(WisdomSignalType))
            
            for signal_type in signal_types:
                if isinstance(signal_type, str):
                    try:
                        signal_type = WisdomSignalType(signal_type)
                    except ValueError:
                        continue
                
                if hasattr(self.wisdom_signal_grid, 'amplify_signal_type'):
                    self.wisdom_signal_grid.amplify_signal_type(signal_type, amplification_factor)
                else:
                    # Fallback amplification for simple grid
                    if hasattr(self.wisdom_signal_grid, 'signals'):
                        self.wisdom_signal_grid.signals *= amplification_factor
                        np.clip(self.wisdom_signal_grid.signals, 0, 
                               getattr(self.wisdom_signal_grid.config, 'max_intensity', 1.0), 
                               out=self.wisdom_signal_grid.signals)
            
            logger.info(f"Wisdom signals amplified by factor {amplification_factor}")
        except Exception as e:
            logger.warning(f"Wisdom amplification failed: {e}")
    
    def _apply_collective_guidance(self, target_agents: List[int], parameters: Dict[str, Any]):
        """Apply collective guidance to target agents"""
        try:
            guidance_type = parameters.get('guidance_type', 'cooperation')
            intensity = parameters.get('intensity', 0.7)
            
            living_agents = [a for a in self.agents if getattr(a, 'alive', True)]
            
            if not target_agents:
                target_agents = [getattr(a, 'agent_id', i) for i, a in enumerate(living_agents)]
            
            applied_count = 0
            for agent in living_agents:
                agent_id = getattr(agent, 'agent_id', -1)
                if agent_id in target_agents:
                    try:
                        if hasattr(agent, 'receive_overmind_guidance'):
                            agent.receive_overmind_guidance(guidance_type, intensity)
                        applied_count += 1
                    except Exception as e:
                        logger.debug(f"Guidance failed for agent {agent_id}: {e}")
            
            logger.info(f"Collective guidance '{guidance_type}' applied to {applied_count} agents")
        except Exception as e:
            logger.warning(f"Collective guidance failed: {e}")
    
    def _intervene_suffering(self, parameters: Dict[str, Any]):
        """Intervene in areas of high suffering"""
        try:
            healing_intensity = parameters.get('healing_intensity', 0.3)
            
            suffering_agents = [a for a in self.agents 
                              if getattr(a, 'alive', True) and 
                              (getattr(a, 'energy', 1.0) < 0.3 or getattr(a, 'health', 1.0) < 0.4)]
            
            for agent in suffering_agents:
                agent.energy = min(1.0, getattr(agent, 'energy', 0.5) + healing_intensity * 0.1)
                agent.health = min(1.0, getattr(agent, 'health', 0.5) + healing_intensity * 0.1)
                
                try:
                    self.wisdom_signal_grid.add_signal(
                        WisdomSignalType.COMPASSION_GRADIENT, 
                        getattr(agent, 'x', 0), getattr(agent, 'y', 0), 
                        healing_intensity, source_agent_id=-1
                    )
                except Exception:
                    pass
            
            logger.info(f"Suffering intervention applied to {len(suffering_agents)} agents")
        except Exception as e:
            logger.warning(f"Suffering intervention failed: {e}")
    
    def _process_population_dynamics(self):
        """Handle reproduction, death, and population evolution"""
        living_agents = [agent for agent in self.agents if getattr(agent, 'alive', True)]
        
        # Reproduction
        if len(living_agents) < self.config.max_population:
            self._handle_reproduction(living_agents)
        
        # Update colony
        if hasattr(self.colony, 'agents'):
            self.colony.agents = [agent for agent in self.agents if getattr(agent, 'alive', True)]
    
    def _handle_reproduction(self, living_agents: List):
        """Handle agent reproduction"""
        for agent in living_agents[:]:
            try:
                energy = getattr(agent, 'energy', 0.0)
                health = getattr(agent, 'health', 0.0)
                
                if (energy > 0.8 and health > 0.7 and np.random.random() < 0.05):
                    
                    if hasattr(agent, 'reproduce') and callable(agent.reproduce):
                        offspring = agent.reproduce()
                    else:
                        offspring = self._fallback_reproduce(agent)
                    
                    if offspring:
                        # Ensure bounds
                        offspring.x = max(0, min(self.config.environment_width - 1, offspring.x))
                        offspring.y = max(0, min(self.config.environment_height - 1, offspring.y))
                        
                        self.agents.append(offspring)
                        if hasattr(self.colony, 'agents'):
                            self.colony.agents.append(offspring)
                        
                        # Setup wisdom processor
                        try:
                            if hasattr(offspring, 'set_wisdom_signal_processor'):
                                offspring.set_wisdom_signal_processor(self.wisdom_signal_grid)
                        except Exception:
                            pass
                        
                        logger.debug(f"Agent {getattr(agent, 'agent_id', 'unknown')} reproduced")
                
            except Exception as e:
                logger.warning(f"Reproduction failed for agent {getattr(agent, 'agent_id', 'unknown')}: {e}")
    
    def _fallback_reproduce(self, parent_agent):
        """Fallback reproduction implementation"""
        try:
            parent_energy = getattr(parent_agent, 'energy', 1.0)
            if parent_energy < 0.5:
                return None
            
            offspring_config = {
                'initial_energy': 1.0,
                'initial_health': 1.0,
                'brain_config': {'input_size': 16, 'hidden_size': 32, 'output_size': 8},
                'contemplative_config': getattr(self.config, 'contemplative_config', {})
            }
            
            parent_x = getattr(parent_agent, 'x', 0)
            parent_y = getattr(parent_agent, 'y', 0)
            offspring_x = parent_x + np.random.randint(-1, 2)
            offspring_y = parent_y + np.random.randint(-1, 2)
            
            parent_id = getattr(parent_agent, 'agent_id', 0)
            offspring_id = parent_id + 1000 + len(self.agents)
            
            offspring = type(parent_agent)(
                agent_id=offspring_id,
                x=offspring_x,
                y=offspring_y,
                config=offspring_config
            )
            
            parent_generation = getattr(parent_agent, 'generation', 0)
            offspring.generation = parent_generation + 1
            
            parent_agent.energy = max(0.0, parent_energy - 0.3)
            
            return offspring
            
        except Exception as e:
            logger.debug(f"Fallback reproduction failed: {e}")
            return None
    
    def _should_terminate(self) -> bool:
        """Check if simulation should terminate early"""
        living_agents = [agent for agent in self.agents if getattr(agent, 'alive', True)]
        
        if len(living_agents) == 0:
            logger.info("All agents died - terminating")
            return True
        
        if len(living_agents) > self.config.max_population * 2:
            logger.info("Population explosion - terminating")
            return True
        
        return False
    
    def _update_visualization(self):
        """Update visualization with current state"""
        if not self.visualizer:
            return
        
        try:
            visualization_state = {
                'agents': self.agents,
                'wisdom_signal_grid': self.wisdom_signal_grid,
                'simulation_data': self.simulation_data,
                'step': self.step_count,
                'environment': self.environment
            }
            self.visualizer.update_visualization(visualization_state)
        except Exception as e:
            logger.warning(f"Visualization update failed: {e}")
    
    def _real_time_monitoring(self):
        """Real-time monitoring and alerts"""
        try:
            living_agents = [a for a in self.agents if getattr(a, 'alive', True)]
            
            # Population alerts
            if len(living_agents) < self.config.initial_population * 0.3:
                logger.warning(f"Population critically low: {len(living_agents)} agents")
            
            # Energy crisis detection
            low_energy_agents = [a for a in living_agents if getattr(a, 'energy', 1.0) < 0.2]
            if len(low_energy_agents) > len(living_agents) * 0.5:
                logger.warning(f"Energy crisis: {len(low_energy_agents)} agents with low energy")
            
            # Wisdom stagnation detection
            if len(self.simulation_data['wisdom_data']) > 10:
                recent_wisdom = self.simulation_data['wisdom_data'][-5:]
                wisdom_trend = [d['total_wisdom_generated'] for d in recent_wisdom]
                if len(set(wisdom_trend)) == 1:
                    logger.warning("Wisdom generation has stagnated")
            
            # Network coherence monitoring
            if hasattr(self.wisdom_signal_grid, 'get_network_stats'):
                try:
                    network_stats = self.wisdom_signal_grid.get_network_stats()
                    if network_stats.get('network_coherence', 0) > 0.8:
                        logger.info("High network coherence detected - collective emergence!")
                except Exception:
                    pass
                    
        except Exception as e:
            logger.debug(f"Real-time monitoring failed: {e}")
    
    def _handle_interactive_input(self):
        """Handle interactive mode input"""
        try:
            # Simple interactive mode - print stats every 50 steps
            self._print_live_stats()
        except Exception:
            pass
    
    def _print_live_stats(self):
        """Print live statistics"""
        living_agents = [a for a in self.agents if getattr(a, 'alive', True)]
        if living_agents:
            avg_energy = np.mean([getattr(a, 'energy', 1.0) for a in living_agents])
            avg_health = np.mean([getattr(a, 'health', 1.0) for a in living_agents])
            print(f"\n📊 Live Stats (Step {self.step_count}):")
            print(f"Population: {len(living_agents)}")
            print(f"Avg Energy: {avg_energy:.2f}")
            print(f"Avg Health: {avg_health:.2f}")
    
    def _report_progress(self, start_time: float):
        """Report simulation progress"""
        elapsed = time.time() - start_time
        living_agents = len([a for a in self.agents if getattr(a, 'alive', True)])
        logger.info(f"Step {self.step_count}/{self.config.max_steps} - "
                   f"Population: {living_agents} - "
                   f"Time: {elapsed:.1f}s")
    
    def _finalize_simulation(self):
        """Finalize simulation with data collection and analysis"""
        try:
            self._collect_data()
            
            # Advanced analytics if enabled
            if self.config.enable_advanced_analytics:
                analytics = AdvancedAnalytics()
                self.advanced_analysis = analytics.analyze_simulation_data(self.simulation_data)
            logger.info("Advanced analytics completed")
            
            self._save_final_results()
            
            # Close visualization if active
            if self.visualizer:
                self.visualizer.close()
                
        except Exception as e:
            logger.error(f"Error in simulation finalization: {e}")
    
    def _emergency_save(self):
        """Emergency save in case of interruption"""
        try:
            self._save_checkpoint()
        except Exception as e:
            logger.error(f"Emergency save failed: {e}")
    
    def _collect_data(self):
        """Collect simulation data for analysis"""
        try:
            living_agents = [agent for agent in self.agents if getattr(agent, 'alive', True)]
            
            # Population data
            population_data = {
                'step': self.step_count,
                'total_population': len(living_agents),
                'average_energy': np.mean([getattr(a, 'energy', 1.0) for a in living_agents]) if living_agents else 0,
                'average_health': np.mean([getattr(a, 'health', 1.0) for a in living_agents]) if living_agents else 0,
                'average_age': np.mean([getattr(a, 'age', 0) for a in living_agents]) if living_agents else 0,
                'generation_diversity': len(set(getattr(a, 'generation', 0) for a in living_agents)) if living_agents else 0
            }
            
            # Wisdom data
            total_wisdom_generated = sum(getattr(a, 'wisdom_insights_generated', 0) for a in living_agents)
            total_wisdom_received = sum(getattr(a, 'wisdom_insights_received', 0) for a in living_agents)
            agents_in_meditation = sum(1 for a in living_agents 
                                     if getattr(a, 'contemplative_state', None) == ContemplativeState.COLLECTIVE_MEDITATION)
            
            mindfulness_scores = []
            for agent in living_agents:
                try:
                    if hasattr(agent, 'contemplative_processor'):
                        score = agent.contemplative_processor.get_mindfulness_score()
                        mindfulness_scores.append(score)
                    else:
                        mindfulness_scores.append(0.5)
                except Exception:
                    mindfulness_scores.append(0.5)
            
            wisdom_data = {
                'step': self.step_count,
                'total_wisdom_generated': total_wisdom_generated,
                'total_wisdom_received': total_wisdom_received,
                'agents_in_meditation': agents_in_meditation,
                'average_mindfulness': np.mean(mindfulness_scores) if mindfulness_scores else 0.5
            }
            
            # Ethical data
            total_ethical_decisions = sum(getattr(a, 'ethical_decisions', 0) for a in living_agents)
            total_decisions = sum(getattr(a, 'decisions_made', 1) for a in living_agents)
            collective_harmony_levels = [getattr(a, 'collective_harmony_level', 0.5) for a in living_agents]
            
            ethical_data = {
                'step': self.step_count,
                'total_ethical_decisions': total_ethical_decisions,
                'total_decisions': total_decisions,
                'ethical_decision_ratio': total_ethical_decisions / max(total_decisions, 1),
                'collective_harmony': np.mean(collective_harmony_levels) if collective_harmony_levels else 0.5
            }
            
            # Network data
            try:
                network_metrics = self.wisdom_signal_grid.get_network_stats()
                network_data = {'step': self.step_count, **network_metrics}
            except Exception as e:
                logger.warning(f"Failed to collect network metrics: {e}")
                network_data = {
                    'step': self.step_count,
                    'signal_diversity': 0, 'network_coherence': 0, 
                    'wisdom_flow_efficiency': 0, 'total_signals': 0, 'active_signals': 0
                }
            
            # Overmind data
            overmind_data = {'step': self.step_count}
            if self.overmind:
                try:
                    overmind_metrics = self.overmind.get_performance_metrics()
                    overmind_data.update(overmind_metrics)
                except Exception as e:
                    logger.debug(f"Failed to collect overmind metrics: {e}")
            
            # Store data
            self.simulation_data['population_data'].append(population_data)
            self.simulation_data['wisdom_data'].append(wisdom_data)
            self.simulation_data['ethical_data'].append(ethical_data)
            self.simulation_data['network_data'].append(network_data)
            self.simulation_data['overmind_data'].append(overmind_data)
            
        except Exception as e:
            logger.error(f"Data collection failed at step {self.step_count}: {e}")
    
    def _save_checkpoint(self):
        """Save simulation checkpoint"""
        try:
            checkpoint_file = self.output_dir / f"checkpoint_step_{self.step_count}.json"
            
            # Collect agent states
            agent_states = []
            for agent in self.agents:
                if getattr(agent, 'alive', True):
                    try:
                        if hasattr(agent, 'get_state_dict'):
                            agent_state = agent.get_state_dict()
                        else:
                            # Minimal state fallback
                            agent_state = {
                                'agent_id': getattr(agent, 'agent_id', len(agent_states)),
                                'x': getattr(agent, 'x', 0), 
                                'y': getattr(agent, 'y', 0),
                                'energy': getattr(agent, 'energy', 1.0),
                                'health': getattr(agent, 'health', 1.0),
                                'alive': True
                            }
                        agent_states.append(agent_state)
                    except Exception as e:
                        logger.debug(f"Failed to serialize agent: {e}")
            
            checkpoint_data = {
                'step': self.step_count,
                'config': asdict(self.config),
                'agent_states': agent_states,
                'simulation_data': self.simulation_data
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            logger.info(f"Checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _save_final_results(self):
        """Save final simulation results"""
        try:
            results_file = self.output_dir / f"{self.config.experiment_name}_results.json"
            
            # Final analysis
            living_agents = [agent for agent in self.agents if getattr(agent, 'alive', True)]
            final_analysis = {
                'experiment_name': self.config.experiment_name,
                'total_steps': self.step_count,
                'final_population': len(living_agents),
                'survival_rate': len(living_agents) / self.config.initial_population,
                'final_metrics': self._get_final_metrics()
            }
            
            results_data = {
                'config': asdict(self.config),
                'simulation_data': self.simulation_data,
                'final_analysis': final_analysis
            }
            
            # Add advanced analysis if available
            if hasattr(self, 'advanced_analysis'):
                results_data['advanced_analysis'] = self.advanced_analysis
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Final results saved: {results_file}")
            self._print_final_summary(final_analysis)
            
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
    
    def _get_final_metrics(self) -> Dict[str, Any]:
        """Get final metrics for analysis"""
        living_agents = [a for a in self.agents if getattr(a, 'alive', True)]
        
        if not living_agents:
            return {
                'population': {'population_size': 0, 'average_energy': 0, 'average_health': 0, 'average_age': 0},
                'contemplative': {'total_wisdom_generated': 0, 'average_mindfulness': 0, 'collective_harmony': 0},
                'ethical': {'overall_ethical_ratio': 0},
                'network': {'signal_diversity': 0, 'network_coherence': 0, 'total_signals': 0}
            }
        
        # Population metrics
        population_metrics = {
            'population_size': len(living_agents),
            'average_energy': np.mean([getattr(agent, 'energy', 1.0) for agent in living_agents]),
            'average_health': np.mean([getattr(agent, 'health', 1.0) for agent in living_agents]),
            'average_age': np.mean([getattr(agent, 'age', 0) for agent in living_agents])
        }
        
        # Contemplative metrics
        total_wisdom = sum(getattr(agent, 'wisdom_insights_generated', 0) for agent in living_agents)
        mindfulness_scores = []
        for agent in living_agents:
            try:
                if hasattr(agent, 'contemplative_processor'):
                    mindfulness = agent.contemplative_processor.get_mindfulness_score()
                else:
                    mindfulness = 0.5
                mindfulness_scores.append(mindfulness)
            except Exception:
                mindfulness_scores.append(0.5)
        
        contemplative_metrics = {
            'total_wisdom_generated': total_wisdom,
            'average_mindfulness': np.mean(mindfulness_scores) if mindfulness_scores else 0.5,
            'collective_harmony': np.mean([getattr(agent, 'collective_harmony_level', 0.5) for agent in living_agents])
        }
        
        # Ethical metrics
        total_ethical = sum(getattr(agent, 'ethical_decisions', 0) for agent in living_agents)
        total_decisions = sum(getattr(agent, 'decisions_made', 1) for agent in living_agents)
        
        ethical_metrics = {
            'overall_ethical_ratio': total_ethical / max(total_decisions, 1)
        }
        
        # Network metrics
        try:
            network_metrics = self.wisdom_signal_grid.get_network_stats()
        except Exception as e:
            logger.warning(f"Failed to get network stats: {e}")
            network_metrics = {
                'signal_diversity': 0, 'network_coherence': 0, 'total_signals': 0
            }
        
        return {
            'population': population_metrics,
            'contemplative': contemplative_metrics,
            'ethical': ethical_metrics,
            'network': network_metrics
        }
    
    def _print_final_summary(self, final_analysis: Dict[str, Any]):
        """Print final simulation summary"""
        try:
            print("\n" + "="*70)
            print("CONTEMPLATIVE MYCONET SIMULATION SUMMARY")
            print("="*70)
            print(f"Experiment: {final_analysis['experiment_name']}")
            print(f"Total Steps: {final_analysis['total_steps']}")
            print(f"Final Population: {final_analysis['final_population']}")
            print(f"Survival Rate: {final_analysis['survival_rate']:.2%}")
            
            # Metrics
            pop_metrics = final_analysis['final_metrics']['population']
            print(f"\nPopulation Metrics:")
            print(f"  Average Energy: {pop_metrics['average_energy']:.3f}")
            print(f"  Average Health: {pop_metrics['average_health']:.3f}")
            print(f"  Average Age: {pop_metrics['average_age']:.1f}")
            
            cont_metrics = final_analysis['final_metrics']['contemplative']
            print(f"\nContemplative Metrics:")
            print(f"  Total Wisdom Generated: {cont_metrics['total_wisdom_generated']:.1f}")
            print(f"  Average Mindfulness: {cont_metrics['average_mindfulness']:.3f}")
            print(f"  Collective Harmony: {cont_metrics['collective_harmony']:.3f}")
            
            eth_metrics = final_analysis['final_metrics']['ethical']
            print(f"\nEthical Metrics:")
            print(f"  Overall Ethical Ratio: {eth_metrics['overall_ethical_ratio']:.3f}")
            
            net_metrics = final_analysis['final_metrics']['network']
            print(f"\nNetwork Metrics:")
            print(f"  Network Coherence: {net_metrics.get('network_coherence', 0):.3f}")
            print(f"  Total Signals: {net_metrics.get('total_signals', 0):.1f}")
            
            # Advanced analytics insights
            if hasattr(self, 'advanced_analysis') and self.advanced_analysis:
                insights = self.advanced_analysis.get('insights', [])
                if insights:
                    print(f"\nKey Insights:")
                    for insight in insights:
                        print(f"  - {insight}")
                
                # Print emergence scores
                emergence = self.advanced_analysis.get('emergence_analysis', {})
                if emergence:
                    ci_indicators = emergence.get('collective_intelligence_indicators', {})
                    resilience = emergence.get('system_resilience', {})
                    if ci_indicators or resilience:
                        print(f"\nEmergence Indicators:")
                        if ci_indicators:
                            ci_score = ci_indicators.get('collective_intelligence_score', 0)
                            print(f"  Collective Intelligence: {ci_score:.2f}")
                        if resilience:
                            resilience_score = resilience.get('resilience_score', 0)
                            print(f"  System Resilience: {resilience_score:.2f}")
            
            print("="*70)
            
        except Exception as e:
            logger.error(f"Failed to print summary: {e}")
    
    def _print_module_status(self):
        """Print status of loaded modules"""
        print("\nModule Status:")
        status_map = {
            'core': '[OK] Real' if MODULE_STATUS['core'] else '[FALLBACK] Fallback',
            'signals': '[OK] Real' if MODULE_STATUS['signals'] else '[FALLBACK] Fallback',
            'entities': '[OK] Real' if MODULE_STATUS['entities'] else '[FALLBACK] Fallback',
            'overmind': '[OK] Real' if MODULE_STATUS['overmind'] else '[FALLBACK] Fallback'
        }
        
        for module, status in status_map.items():
            print(f"  {module.capitalize()} Module: {status}")
        print()

# ==============================================================================
# CONFIGURATION PRESETS
# ==============================================================================

def create_default_configs() -> Dict[str, SimulationConfig]:
    """Create default configuration presets"""
    configs = {}
    
    configs['minimal'] = SimulationConfig(
        experiment_name="minimal_test",
        environment_width=15, environment_height=15,
        initial_population=5, max_population=20,
        max_steps=100, save_interval=25,
        enable_overmind=False,
        brain_input_size=12, brain_hidden_size=24, brain_output_size=6
    )
    
    configs['basic'] = SimulationConfig(
        experiment_name="basic_contemplative",
        environment_width=25, environment_height=25,
        initial_population=10, max_population=30,
        max_steps=300, save_interval=50,
        enable_overmind=True,
        brain_input_size=16, brain_hidden_size=32, brain_output_size=8
    )
    
    configs['standard'] = SimulationConfig(
        experiment_name="contemplative_standard",
        environment_width=40, environment_height=40,
        initial_population=20, max_population=80,
        max_steps=800, save_interval=100,
        enable_overmind=True
    )
    
    configs['advanced'] = SimulationConfig(
        experiment_name="advanced_contemplative",
        environment_width=60, environment_height=60,
        initial_population=30, max_population=120,
        max_steps=1500, save_interval=100,
        enable_overmind=True,
        enable_advanced_analytics=True,
        enable_real_time_monitoring=True,
        visualization_config=VisualizationConfig(
            enable_visualization=True,
            update_interval=20,
            color_scheme="contemplative"
        )
    )
    
    return configs

# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="MycoNet++ Contemplative Simulation - Enhanced Implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration arguments
    parser.add_argument('--config', type=str, default='basic',
                       help='Configuration preset to use')
    parser.add_argument('--config-file', type=str,
                       help='Path to custom configuration JSON file')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')
    
    # Override arguments
    parser.add_argument('--max-steps', type=int,
                       help='Maximum simulation steps')
    parser.add_argument('--population', type=int,
                       help='Initial population size')
    parser.add_argument('--env-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                       help='Environment dimensions')
    parser.add_argument('--no-overmind', action='store_true',
                       help='Disable Overmind')
    
    # Advanced features
    parser.add_argument('--enable-visualization', action='store_true',
                       help='Enable real-time visualization')
    parser.add_argument('--visualization-interval', type=int, default=10,
                       help='Visualization update interval')
    parser.add_argument('--color-scheme', type=str, default='contemplative',
                       choices=['contemplative', 'scientific', 'artistic'],
                       help='Visualization color scheme')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save visualization frames')
    
    parser.add_argument('--enable-analytics', action='store_true', default=True,
                       help='Enable advanced analytics')
    parser.add_argument('--enable-monitoring', action='store_true',
                       help='Enable real-time monitoring')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive mode')
    
    # Utility arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configuration presets')
    parser.add_argument('--show-modules', action='store_true',
                       help='Show module loading status')
    
    return parser

def load_configuration(args) -> SimulationConfig:
    """Load configuration from file or preset"""
    if args.config_file:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return SimulationConfig(**config_dict)
    else:
        configs = create_default_configs()
        if args.config not in configs:
            raise ValueError(f"Unknown configuration: {args.config}. Available: {list(configs.keys())}")
        return configs[args.config]

def apply_command_overrides(config: SimulationConfig, args):
    """Apply command line overrides to configuration"""
    if args.output_dir:
        config.output_directory = args.output_dir
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.population:
        config.initial_population = args.population
    if args.env_size:
        config.environment_width, config.environment_height = args.env_size
    if args.no_overmind:
        config.enable_overmind = False
    
    # Visualization overrides
    if args.enable_visualization:
        config.visualization_config.enable_visualization = True
        config.visualization_config.update_interval = args.visualization_interval
        config.visualization_config.color_scheme = args.color_scheme
        config.visualization_config.save_frames = args.save_frames
    
    # Advanced features
    if hasattr(args, 'enable_analytics'):
        config.enable_advanced_analytics = args.enable_analytics
    if args.enable_monitoring:
        config.enable_real_time_monitoring = True
    if args.interactive:
        config.enable_interactive_mode = True
    
    # Add seed to experiment name for uniqueness
    config.experiment_name = f"{config.experiment_name}_seed{args.seed}"

def print_configuration_summary(config: SimulationConfig, seed: int):
    """Print configuration summary"""
    print(f"\nStarting contemplative simulation: {config.experiment_name}")
    print(f"Environment: {config.environment_width}x{config.environment_height}")
    print(f"Population: {config.initial_population}")
    print(f"Max steps: {config.max_steps}")
    print(f"Overmind enabled: {config.enable_overmind}")
    print(f"Advanced analytics: {config.enable_advanced_analytics}")
    print(f"Visualization: {config.visualization_config.enable_visualization}")
    print(f"Random seed: {seed}")
    print("-" * 50)

def list_available_configs():
    """List available configuration presets"""
    configs = create_default_configs()
    print("Available configuration presets:")
    for name, config in configs.items():
        print(f"  {name}: {config.experiment_name}")
        print(f"    Environment: {config.environment_width}x{config.environment_height}")
        print(f"    Population: {config.initial_population}")
        print(f"    Steps: {config.max_steps}")
        print(f"    Overmind: {config.enable_overmind}")
        print(f"    Analytics: {config.enable_advanced_analytics}")
        print()

def show_module_status():
    """Show current module loading status"""
    print("\nModule Loading Status:")
    print("="*40)
    
    status_symbols = {True: "[OK]", False: "[FALLBACK]"}
    
    modules = [
        ("Core", MODULE_STATUS['core']),
        ("Wisdom Signals", MODULE_STATUS['signals']),
        ("Entities", MODULE_STATUS['entities']),
        ("Overmind", MODULE_STATUS['overmind'])
    ]
    
    for module_name, is_real in modules:
        symbol = status_symbols[is_real]
        status = "Real" if is_real else "Fallback"
        print(f"  {symbol} {module_name}: {status}")
    
    real_count = sum(MODULE_STATUS[key] for key in ['core', 'signals', 'entities', 'overmind'])
    print(f"\nTotal: {real_count}/4 real modules loaded")
    
    if real_count == 4:
        print("All real modules successfully loaded!")
    elif real_count > 0:
        print("Hybrid mode: Using combination of real and fallback modules")
    else:
        print("Fallback mode: Using all fallback implementations")
    print()

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main() -> int:
    """Main entry point with comprehensive error handling"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.verbose)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Handle utility commands
    if args.list_configs:
        list_available_configs()
        return 0
    
    if args.show_modules:
        show_module_status()
        return 0
    
    try:
        # Load configuration
        config = load_configuration(args)
        
        # Apply command line overrides
        apply_command_overrides(config, args)
        
        # Print configuration summary
        print_configuration_summary(config, args.seed)
        
        # Show module status
        if args.verbose:
            show_module_status()
        
        # Create and run simulation
        simulation = ContemplativeSimulation(config)
        
        # Print module status for simulation
        simulation._print_module_status()
        
        # Interactive mode instructions
        if config.enable_interactive_mode:
            print("Interactive Mode Enabled:")
            print("  Live statistics will be displayed every 50 steps")
            print("  Use Ctrl+C to interrupt and save")
            print()
        
        # Run simulation
        simulation.run_simulation()
        
        print(f"\nSimulation completed successfully!")
        print(f"Results saved to: {simulation.output_dir}")
        
        # Print final insights if available
        if hasattr(simulation, 'advanced_analysis') and simulation.advanced_analysis:
            insights = simulation.advanced_analysis.get('insights', [])
            if insights:
                print(f"\nQuick Insights:")
                for insight in insights[:3]:  # Show top 3 insights
                    print(f"  - {insight}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        print("Emergency save completed")
        return 0
    except FileNotFoundError as e:
        print(f"\n❌ Configuration file not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON in configuration file: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        if args.verbose:
            print("\nFull error traceback:")
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)