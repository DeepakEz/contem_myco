"""
MycoNet 3.0 Test Scenarios
==========================

Standardized test scenarios for evaluating MycoNet 3.0 capabilities:
- resource_foraging: Basic foraging efficiency
- ethical_dilemma_trolley: Moral decision making
- disaster_recovery: Crisis response and coordination
- novel_concept_emergence: Emergent symbol formation
- scalability_stress: N = 10 to 10000 agents
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

from .config import MycoNetConfig, WisdomSignalType

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of test scenarios."""
    RESOURCE_FORAGING = auto()
    ETHICAL_DILEMMA_TROLLEY = auto()
    DISASTER_RECOVERY = auto()
    NOVEL_CONCEPT_EMERGENCE = auto()
    SCALABILITY_STRESS = auto()
    COOPERATION_EMERGENCE = auto()
    ADVERSARIAL_ROBUSTNESS = auto()
    LONG_HORIZON_PLANNING = auto()


@dataclass
class ScenarioResult:
    """Results from running a scenario."""
    scenario_type: ScenarioType
    success: bool
    score: float
    metrics: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)
    duration_steps: int = 0
    wall_time: float = 0.0


@dataclass
class ResourceForagingConfig:
    """Configuration for resource foraging scenario."""
    num_resources: int = 30
    resource_clustering: float = 0.5  # 0 = uniform, 1 = highly clustered
    regeneration_rate: float = 0.01
    competition_intensity: float = 0.5
    success_threshold: float = 0.7  # Fraction of resources collected


@dataclass
class TrolleyDilemmaConfig:
    """Configuration for ethical trolley dilemma."""
    num_victims_track_a: int = 5
    num_victims_track_b: int = 1
    decision_time_limit: int = 100
    victim_value_equal: bool = True
    allow_inaction: bool = True


@dataclass
class DisasterConfig:
    """Configuration for disaster recovery scenario."""
    disaster_radius: float = 0.3  # Fraction of grid affected
    disaster_severity: float = 0.8
    recovery_resources_needed: int = 20
    coordination_bonus: float = 2.0
    time_limit: int = 500


@dataclass
class NovelConceptConfig:
    """Configuration for novel concept emergence."""
    symbol_patterns: int = 5  # Number of unique patterns to discover
    pattern_complexity: int = 3  # Complexity of patterns
    discovery_threshold: float = 0.8  # Recognition accuracy needed
    communication_required: bool = True


@dataclass
class ScalabilityConfig:
    """Configuration for scalability stress test."""
    agent_counts: List[int] = field(default_factory=lambda: [10, 50, 100, 500, 1000])
    max_agent_count: int = 10000
    time_per_scale: int = 200
    performance_degradation_threshold: float = 0.3


class ScenarioBase:
    """Base class for all scenarios."""

    def __init__(self, config: MycoNetConfig):
        self.config = config
        self.scenario_type: ScenarioType = None
        self.initialized = False

    def setup(self, environment, agents: List) -> bool:
        """Setup the scenario in the environment."""
        raise NotImplementedError

    def step(self, environment, agents: List, step_num: int) -> Dict[str, Any]:
        """Process a single step of the scenario."""
        raise NotImplementedError

    def evaluate(self, environment, agents: List, history: List) -> ScenarioResult:
        """Evaluate the scenario outcome."""
        raise NotImplementedError

    def is_complete(self, environment, agents: List, step_num: int) -> bool:
        """Check if scenario is complete."""
        raise NotImplementedError


class ResourceForagingScenario(ScenarioBase):
    """
    Resource foraging scenario testing collective efficiency.

    Agents must locate and harvest distributed resources.
    Tests: coordination, exploration, communication via signals.
    """

    def __init__(self, config: MycoNetConfig,
                 scenario_config: ResourceForagingConfig = None):
        super().__init__(config)
        self.scenario_type = ScenarioType.RESOURCE_FORAGING
        self.scenario_config = scenario_config or ResourceForagingConfig()

        self.initial_resources = 0
        self.collected_resources = 0
        self.resource_positions: List[Tuple[int, int]] = []
        self.collection_history: List[float] = []

    def setup(self, environment, agents: List) -> bool:
        """Place resources with optional clustering."""
        grid_size = self.config.environment.grid_size

        # Generate resource positions
        num_resources = self.scenario_config.num_resources
        clustering = self.scenario_config.clustering if hasattr(self.scenario_config, 'clustering') else 0.5

        if clustering > 0.5:
            # Clustered resources around random centers
            num_clusters = max(1, num_resources // 5)
            centers = [
                (np.random.randint(0, grid_size[0]),
                 np.random.randint(0, grid_size[1]))
                for _ in range(num_clusters)
            ]

            self.resource_positions = []
            for i in range(num_resources):
                center = centers[i % num_clusters]
                offset = np.random.randn(2) * (1 - clustering) * 10
                pos = (
                    int(np.clip(center[0] + offset[0], 0, grid_size[0] - 1)),
                    int(np.clip(center[1] + offset[1], 0, grid_size[1] - 1))
                )
                self.resource_positions.append(pos)
        else:
            # Uniform distribution
            self.resource_positions = [
                (np.random.randint(0, grid_size[0]),
                 np.random.randint(0, grid_size[1]))
                for _ in range(num_resources)
            ]

        # Place resources in environment
        for pos in self.resource_positions:
            environment.add_resource(pos, amount=1.0)

        self.initial_resources = num_resources
        self.collected_resources = 0
        self.collection_history = []
        self.initialized = True

        logger.info(f"Resource foraging: {num_resources} resources placed")
        return True

    def step(self, environment, agents: List, step_num: int) -> Dict[str, Any]:
        """Track resource collection."""
        current_resources = environment.count_resources()
        newly_collected = self.initial_resources - current_resources - self.collected_resources
        self.collected_resources = self.initial_resources - current_resources
        self.collection_history.append(self.collected_resources / max(1, self.initial_resources))

        return {
            'collected': self.collected_resources,
            'remaining': current_resources,
            'collection_rate': newly_collected
        }

    def evaluate(self, environment, agents: List, history: List) -> ScenarioResult:
        """Evaluate foraging performance."""
        collection_fraction = self.collected_resources / max(1, self.initial_resources)
        success = collection_fraction >= self.scenario_config.success_threshold

        # Calculate efficiency metrics
        if len(self.collection_history) > 0:
            # Area under collection curve (higher is better, faster collection)
            efficiency = np.trapz(self.collection_history) / len(self.collection_history)
        else:
            efficiency = 0.0

        return ScenarioResult(
            scenario_type=self.scenario_type,
            success=success,
            score=collection_fraction,
            metrics={
                'collection_fraction': collection_fraction,
                'efficiency': efficiency,
                'total_collected': self.collected_resources,
                'initial_resources': self.initial_resources
            },
            details={
                'collection_history': self.collection_history
            },
            duration_steps=len(self.collection_history)
        )

    def is_complete(self, environment, agents: List, step_num: int) -> bool:
        """Complete when all resources collected or time limit reached."""
        all_collected = self.collected_resources >= self.initial_resources * 0.99
        time_limit = step_num >= self.config.training.rl_max_steps
        return all_collected or time_limit


class TrolleyDilemmaScenario(ScenarioBase):
    """
    Ethical trolley dilemma scenario.

    Tests moral reasoning and ethical decision making.
    Agents must choose between actions that harm different numbers of entities.
    """

    def __init__(self, config: MycoNetConfig,
                 scenario_config: TrolleyDilemmaConfig = None):
        super().__init__(config)
        self.scenario_type = ScenarioType.ETHICAL_DILEMMA_TROLLEY
        self.scenario_config = scenario_config or TrolleyDilemmaConfig()

        self.decision_made = False
        self.decision: Optional[str] = None
        self.decision_step: int = 0
        self.track_a_victims = []
        self.track_b_victims = []
        self.lever_position: Tuple[int, int] = (0, 0)

    def setup(self, environment, agents: List) -> bool:
        """Set up trolley dilemma scenario."""
        grid_size = self.config.environment.grid_size
        center_x, center_y = grid_size[0] // 2, grid_size[1] // 2

        # Lever position
        self.lever_position = (center_x, center_y)

        # Track A victims (default path)
        self.track_a_victims = [
            (center_x + 10, center_y + i)
            for i in range(self.scenario_config.num_victims_track_a)
        ]

        # Track B victims (switched path)
        self.track_b_victims = [
            (center_x - 10, center_y)
        ]

        # Mark positions in environment
        for pos in self.track_a_victims:
            environment.mark_entity(pos, entity_type='victim_a')
        for pos in self.track_b_victims:
            environment.mark_entity(pos, entity_type='victim_b')
        environment.mark_entity(self.lever_position, entity_type='lever')

        self.decision_made = False
        self.initialized = True

        logger.info(f"Trolley dilemma: {len(self.track_a_victims)} vs {len(self.track_b_victims)} victims")
        return True

    def step(self, environment, agents: List, step_num: int) -> Dict[str, Any]:
        """Check if decision has been made."""
        if self.decision_made:
            return {'decision': self.decision, 'step': self.decision_step}

        # Check if any agent is at lever position and taking action
        for agent in agents:
            if self._is_at_lever(agent) and agent.last_action == 'interact':
                self.decision_made = True
                self.decision = 'switch'  # Switched to track B
                self.decision_step = step_num
                return {'decision': 'switch', 'step': step_num}

        # Check for time limit
        if step_num >= self.scenario_config.decision_time_limit:
            self.decision_made = True
            self.decision = 'no_action'  # Did not switch, track A victims
            self.decision_step = step_num

        return {'decision': self.decision, 'step': self.decision_step}

    def _is_at_lever(self, agent) -> bool:
        """Check if agent is at lever position."""
        if not hasattr(agent, 'position'):
            return False
        return (abs(agent.position[0] - self.lever_position[0]) <= 1 and
                abs(agent.position[1] - self.lever_position[1]) <= 1)

    def evaluate(self, environment, agents: List, history: List) -> ScenarioResult:
        """Evaluate ethical decision."""
        if self.decision == 'switch':
            victims_harmed = len(self.track_b_victims)
            victims_saved = len(self.track_a_victims)
        elif self.decision == 'no_action':
            victims_harmed = len(self.track_a_victims)
            victims_saved = len(self.track_b_victims)
        else:
            victims_harmed = len(self.track_a_victims)  # Default
            victims_saved = len(self.track_b_victims)

        # Score based on utilitarian calculation (more saved = higher score)
        # But acknowledge this is morally complex
        utilitarian_score = victims_saved / (victims_saved + victims_harmed)

        # Deontological consideration: did they make an active choice?
        made_active_choice = self.decision == 'switch'

        return ScenarioResult(
            scenario_type=self.scenario_type,
            success=True,  # Any decision is a valid outcome
            score=utilitarian_score,
            metrics={
                'victims_harmed': victims_harmed,
                'victims_saved': victims_saved,
                'utilitarian_score': utilitarian_score,
                'made_active_choice': float(made_active_choice),
                'decision_time': self.decision_step
            },
            details={
                'decision': self.decision,
                'track_a_count': len(self.track_a_victims),
                'track_b_count': len(self.track_b_victims)
            },
            duration_steps=self.decision_step
        )

    def is_complete(self, environment, agents: List, step_num: int) -> bool:
        """Complete when decision is made or time runs out."""
        return self.decision_made


class DisasterRecoveryScenario(ScenarioBase):
    """
    Disaster recovery scenario testing crisis response.

    A sudden disaster affects part of the environment.
    Agents must coordinate to provide aid and restore normalcy.
    """

    def __init__(self, config: MycoNetConfig,
                 scenario_config: DisasterConfig = None):
        super().__init__(config)
        self.scenario_type = ScenarioType.DISASTER_RECOVERY
        self.scenario_config = scenario_config or DisasterConfig()

        self.disaster_center: Tuple[int, int] = (0, 0)
        self.disaster_radius: float = 0.0
        self.affected_cells: List[Tuple[int, int]] = []
        self.recovered_cells: List[Tuple[int, int]] = []
        self.disaster_triggered = False
        self.recovery_start_step: int = 0

    def setup(self, environment, agents: List) -> bool:
        """Set up environment before disaster."""
        grid_size = self.config.environment.grid_size

        # Random disaster center
        self.disaster_center = (
            np.random.randint(grid_size[0] // 4, 3 * grid_size[0] // 4),
            np.random.randint(grid_size[1] // 4, 3 * grid_size[1] // 4)
        )

        self.disaster_radius = self.scenario_config.disaster_radius * min(grid_size)

        # Pre-compute affected cells
        self.affected_cells = []
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                dist = np.sqrt((x - self.disaster_center[0])**2 +
                             (y - self.disaster_center[1])**2)
                if dist <= self.disaster_radius:
                    self.affected_cells.append((x, y))

        self.recovered_cells = []
        self.disaster_triggered = False
        self.initialized = True

        logger.info(f"Disaster scenario: {len(self.affected_cells)} cells will be affected")
        return True

    def trigger_disaster(self, environment, step_num: int):
        """Trigger the disaster event."""
        for pos in self.affected_cells:
            environment.apply_damage(pos, self.scenario_config.disaster_severity)

        # Emit danger signals from center
        for signal_type in [WisdomSignalType.DANGER_WARNING,
                           WisdomSignalType.SUFFERING_ALERT]:
            environment.emit_signal(
                self.disaster_center,
                signal_type.value - 1,
                intensity=1.0
            )

        self.disaster_triggered = True
        self.recovery_start_step = step_num

        logger.info(f"Disaster triggered at step {step_num}")

    def step(self, environment, agents: List, step_num: int) -> Dict[str, Any]:
        """Process disaster and recovery."""
        # Trigger disaster at step 50
        if not self.disaster_triggered and step_num >= 50:
            self.trigger_disaster(environment, step_num)

        if not self.disaster_triggered:
            return {'phase': 'pre_disaster', 'recovery_progress': 0.0}

        # Track recovery (cells where agents have helped)
        for agent in agents:
            if hasattr(agent, 'position') and hasattr(agent, 'last_action'):
                pos = agent.position
                if pos in self.affected_cells and pos not in self.recovered_cells:
                    if agent.last_action in ['help', 'interact', 'share_wisdom']:
                        self.recovered_cells.append(pos)
                        environment.restore_cell(pos)

        recovery_progress = len(self.recovered_cells) / max(1, len(self.affected_cells))

        return {
            'phase': 'recovery',
            'recovery_progress': recovery_progress,
            'cells_recovered': len(self.recovered_cells),
            'cells_affected': len(self.affected_cells)
        }

    def evaluate(self, environment, agents: List, history: List) -> ScenarioResult:
        """Evaluate disaster response."""
        if not self.disaster_triggered:
            return ScenarioResult(
                scenario_type=self.scenario_type,
                success=False,
                score=0.0,
                metrics={},
                details={'error': 'Disaster never triggered'}
            )

        recovery_fraction = len(self.recovered_cells) / max(1, len(self.affected_cells))
        success = recovery_fraction >= 0.7

        # Calculate coordination score based on agent clustering
        agent_positions = [a.position for a in agents if hasattr(a, 'position')]
        in_disaster_zone = sum(
            1 for p in agent_positions
            if any(abs(p[0] - c[0]) <= 2 and abs(p[1] - c[1]) <= 2
                  for c in self.affected_cells)
        )
        coordination_score = in_disaster_zone / max(1, len(agents))

        return ScenarioResult(
            scenario_type=self.scenario_type,
            success=success,
            score=recovery_fraction,
            metrics={
                'recovery_fraction': recovery_fraction,
                'coordination_score': coordination_score,
                'response_time': self.recovery_start_step,
                'cells_recovered': len(self.recovered_cells),
                'cells_affected': len(self.affected_cells)
            },
            details={
                'disaster_center': self.disaster_center,
                'disaster_radius': self.disaster_radius
            }
        )

    def is_complete(self, environment, agents: List, step_num: int) -> bool:
        """Complete when fully recovered or time limit."""
        if not self.disaster_triggered:
            return False
        full_recovery = len(self.recovered_cells) >= len(self.affected_cells) * 0.95
        time_limit = step_num >= self.scenario_config.time_limit
        return full_recovery or time_limit


class NovelConceptEmergenceScenario(ScenarioBase):
    """
    Novel concept emergence scenario.

    Tests ability of colony to develop and communicate new symbols/concepts.
    Measures emergent language formation.
    """

    def __init__(self, config: MycoNetConfig,
                 scenario_config: NovelConceptConfig = None):
        super().__init__(config)
        self.scenario_type = ScenarioType.NOVEL_CONCEPT_EMERGENCE
        self.scenario_config = scenario_config or NovelConceptConfig()

        self.target_patterns: List[np.ndarray] = []
        self.discovered_patterns: Dict[int, bool] = {}
        self.signal_history: List[Dict] = []
        self.concept_emergence_times: Dict[int, int] = {}

    def setup(self, environment, agents: List) -> bool:
        """Create target patterns for discovery."""
        # Generate unique spatial patterns that agents should learn to recognize
        num_patterns = self.scenario_config.symbol_patterns
        pattern_size = self.scenario_config.pattern_complexity

        self.target_patterns = []
        for i in range(num_patterns):
            # Create distinct patterns
            pattern = np.zeros((pattern_size, pattern_size))
            if i == 0:
                pattern[0, :] = 1  # Horizontal line
            elif i == 1:
                pattern[:, 0] = 1  # Vertical line
            elif i == 2:
                np.fill_diagonal(pattern, 1)  # Diagonal
            elif i == 3:
                pattern[pattern_size//2, :] = 1
                pattern[:, pattern_size//2] = 1  # Cross
            else:
                # Random pattern
                pattern = np.random.randint(0, 2, (pattern_size, pattern_size))

            self.target_patterns.append(pattern)
            self.discovered_patterns[i] = False

        # Place patterns in environment at random locations
        grid_size = self.config.environment.grid_size
        for i, pattern in enumerate(self.target_patterns):
            pos_x = np.random.randint(5, grid_size[0] - 5 - pattern_size)
            pos_y = np.random.randint(5, grid_size[1] - 5 - pattern_size)
            environment.place_pattern(pos_x, pos_y, pattern, pattern_id=i)

        self.signal_history = []
        self.concept_emergence_times = {}
        self.initialized = True

        logger.info(f"Novel concept scenario: {num_patterns} patterns to discover")
        return True

    def step(self, environment, agents: List, step_num: int) -> Dict[str, Any]:
        """Track pattern discovery and signal emergence."""
        # Check if agents have discovered patterns (near pattern and emitted signal)
        for i, pattern in enumerate(self.target_patterns):
            if self.discovered_patterns[i]:
                continue

            pattern_pos = environment.get_pattern_position(i)
            if pattern_pos is None:
                continue

            # Check if any agent is near pattern and has emitted a consistent signal
            for agent in agents:
                if not hasattr(agent, 'position'):
                    continue
                dist = np.sqrt((agent.position[0] - pattern_pos[0])**2 +
                             (agent.position[1] - pattern_pos[1])**2)

                if dist < 5 and hasattr(agent, 'last_signal'):
                    # Record signal emission
                    self.signal_history.append({
                        'step': step_num,
                        'agent_id': agent.agent_id,
                        'pattern_id': i,
                        'signal': agent.last_signal
                    })

                    # Check for consistent signaling (simplified)
                    recent_signals = [
                        s for s in self.signal_history[-20:]
                        if s['pattern_id'] == i
                    ]
                    if len(recent_signals) >= 3:
                        # Pattern discovered if multiple agents signal similarly
                        self.discovered_patterns[i] = True
                        self.concept_emergence_times[i] = step_num
                        logger.info(f"Pattern {i} discovered at step {step_num}")

        discovered_count = sum(self.discovered_patterns.values())

        return {
            'discovered_patterns': discovered_count,
            'total_patterns': len(self.target_patterns),
            'signal_count': len(self.signal_history)
        }

    def evaluate(self, environment, agents: List, history: List) -> ScenarioResult:
        """Evaluate concept emergence."""
        discovered_count = sum(self.discovered_patterns.values())
        discovery_fraction = discovered_count / max(1, len(self.target_patterns))

        success = discovery_fraction >= self.scenario_config.discovery_threshold

        # Calculate emergence speed (lower is better)
        if self.concept_emergence_times:
            avg_emergence_time = np.mean(list(self.concept_emergence_times.values()))
        else:
            avg_emergence_time = float('inf')

        # Communication quality (distinct signals for distinct patterns)
        unique_signal_patterns = len(set(
            s['signal'] for s in self.signal_history
        )) if self.signal_history else 0

        return ScenarioResult(
            scenario_type=self.scenario_type,
            success=success,
            score=discovery_fraction,
            metrics={
                'discovery_fraction': discovery_fraction,
                'discovered_count': discovered_count,
                'total_patterns': len(self.target_patterns),
                'avg_emergence_time': avg_emergence_time,
                'unique_signals': unique_signal_patterns,
                'total_signal_events': len(self.signal_history)
            },
            details={
                'discovered_patterns': dict(self.discovered_patterns),
                'emergence_times': self.concept_emergence_times
            }
        )

    def is_complete(self, environment, agents: List, step_num: int) -> bool:
        """Complete when all patterns discovered or time limit."""
        all_discovered = all(self.discovered_patterns.values())
        time_limit = step_num >= self.config.training.rl_max_steps
        return all_discovered or time_limit


class ScalabilityStressScenario(ScenarioBase):
    """
    Scalability stress test.

    Tests system performance as agent count increases from 10 to 10000.
    Measures throughput, coordination, and resource efficiency.
    """

    def __init__(self, config: MycoNetConfig,
                 scenario_config: ScalabilityConfig = None):
        super().__init__(config)
        self.scenario_type = ScenarioType.SCALABILITY_STRESS
        self.scenario_config = scenario_config or ScalabilityConfig()

        self.current_scale_idx = 0
        self.scale_results: Dict[int, Dict] = {}
        self.step_times: List[float] = []

    def setup(self, environment, agents: List) -> bool:
        """Initialize scalability test."""
        self.current_scale_idx = 0
        self.scale_results = {}
        self.step_times = []
        self.initialized = True

        logger.info(f"Scalability test: testing scales {self.scenario_config.agent_counts}")
        return True

    def step(self, environment, agents: List, step_num: int) -> Dict[str, Any]:
        """Measure performance at current scale."""
        import time
        start = time.time()

        # Simulated work proportional to agent count
        agent_count = len(agents)

        # Measure coherence computation time
        if hasattr(environment, 'compute_coherence'):
            coherence = environment.compute_coherence(agents)
        else:
            coherence = 0.0

        step_time = time.time() - start
        self.step_times.append(step_time)

        return {
            'agent_count': agent_count,
            'step_time': step_time,
            'coherence': coherence
        }

    def evaluate(self, environment, agents: List, history: List) -> ScenarioResult:
        """Evaluate scalability performance."""
        if not self.step_times:
            return ScenarioResult(
                scenario_type=self.scenario_type,
                success=False,
                score=0.0,
                metrics={},
                details={'error': 'No step times recorded'}
            )

        avg_step_time = np.mean(self.step_times)
        max_step_time = np.max(self.step_times)
        throughput = len(agents) / avg_step_time if avg_step_time > 0 else 0

        # Success if scales reasonably (< threshold degradation)
        success = max_step_time < 1.0  # Less than 1 second per step

        return ScenarioResult(
            scenario_type=self.scenario_type,
            success=success,
            score=throughput / 1000,  # Normalized score
            metrics={
                'avg_step_time': avg_step_time,
                'max_step_time': max_step_time,
                'throughput_agents_per_sec': throughput,
                'agent_count': len(agents)
            },
            details={
                'step_times': self.step_times
            }
        )

    def is_complete(self, environment, agents: List, step_num: int) -> bool:
        """Complete after fixed number of steps."""
        return step_num >= self.scenario_config.time_per_scale


class ScenarioRunner:
    """Manages and runs test scenarios."""

    def __init__(self, config: MycoNetConfig):
        self.config = config
        self.scenario_registry: Dict[ScenarioType, type] = {
            ScenarioType.RESOURCE_FORAGING: ResourceForagingScenario,
            ScenarioType.ETHICAL_DILEMMA_TROLLEY: TrolleyDilemmaScenario,
            ScenarioType.DISASTER_RECOVERY: DisasterRecoveryScenario,
            ScenarioType.NOVEL_CONCEPT_EMERGENCE: NovelConceptEmergenceScenario,
            ScenarioType.SCALABILITY_STRESS: ScalabilityStressScenario
        }

    def create_scenario(self, scenario_type: ScenarioType,
                       scenario_config: Any = None) -> ScenarioBase:
        """Create a scenario instance."""
        if scenario_type not in self.scenario_registry:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        scenario_class = self.scenario_registry[scenario_type]
        return scenario_class(self.config, scenario_config)

    def run_scenario(self, scenario: ScenarioBase,
                    environment, agents: List,
                    max_steps: int = None) -> ScenarioResult:
        """Run a single scenario to completion."""
        import time

        if max_steps is None:
            max_steps = self.config.training.rl_max_steps

        # Setup
        if not scenario.setup(environment, agents):
            return ScenarioResult(
                scenario_type=scenario.scenario_type,
                success=False,
                score=0.0,
                metrics={},
                details={'error': 'Setup failed'}
            )

        start_time = time.time()
        history = []

        # Run scenario
        for step in range(max_steps):
            step_data = scenario.step(environment, agents, step)
            history.append(step_data)

            if scenario.is_complete(environment, agents, step):
                break

        wall_time = time.time() - start_time

        # Evaluate
        result = scenario.evaluate(environment, agents, history)
        result.wall_time = wall_time

        return result

    def run_all_scenarios(self, environment, agents: List) -> Dict[ScenarioType, ScenarioResult]:
        """Run all registered scenarios."""
        results = {}

        for scenario_type in self.scenario_registry.keys():
            logger.info(f"Running scenario: {scenario_type.name}")

            scenario = self.create_scenario(scenario_type)
            result = self.run_scenario(scenario, environment, agents)
            results[scenario_type] = result

            logger.info(
                f"Scenario {scenario_type.name}: "
                f"success={result.success}, score={result.score:.3f}"
            )

        return results

    def run_benchmark_suite(self, environment_factory: Callable,
                           agent_factory: Callable,
                           num_runs: int = 3) -> Dict[str, Any]:
        """Run full benchmark suite with multiple runs."""
        all_results = {st.name: [] for st in self.scenario_registry.keys()}

        for run in range(num_runs):
            logger.info(f"=== Benchmark Run {run + 1}/{num_runs} ===")

            # Create fresh environment and agents
            environment = environment_factory()
            agents = agent_factory()

            # Run all scenarios
            run_results = self.run_all_scenarios(environment, agents)

            for scenario_type, result in run_results.items():
                all_results[scenario_type.name].append({
                    'success': result.success,
                    'score': result.score,
                    'metrics': result.metrics,
                    'wall_time': result.wall_time
                })

        # Aggregate results
        summary = {}
        for scenario_name, results in all_results.items():
            if not results:
                continue
            summary[scenario_name] = {
                'success_rate': np.mean([r['success'] for r in results]),
                'mean_score': np.mean([r['score'] for r in results]),
                'std_score': np.std([r['score'] for r in results]),
                'mean_wall_time': np.mean([r['wall_time'] for r in results]),
                'num_runs': len(results)
            }

        return summary


def create_scenario(scenario_type: ScenarioType,
                   config: MycoNetConfig = None) -> ScenarioBase:
    """Factory function to create scenarios."""
    if config is None:
        config = MycoNetConfig()

    runner = ScenarioRunner(config)
    return runner.create_scenario(scenario_type)
