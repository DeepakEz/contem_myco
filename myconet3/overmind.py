"""
MycoNet 3.0 Overmind Module
===========================

Central coordinator implementing UPRT field theory.

Major Components:
1. UPRT Field Simulator/Surrogate
2. Metric Analyzer (S, I, C, Φ_f, RX, τ_R)
3. Symbol Detection & Resonance Tracker
4. Decision Planner (intervention strategy)
5. Dharma Compiler (ethical constraint checker)
6. Symbolic Reasoning Engine
7. Hypernetwork for genome→weights
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import logging

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .config import OvermindConfig, DharmaConfig, QREAConfig
from .uprt_field import UPRTField, FieldSurrogateModel, create_field_surrogate
from .field_metrics import ColonyMetrics, MetricsComputer
from .hypernetwork import GenomeHyperNet, EvolutionEngine, create_hypernetwork
from .symbolic_parametric_bridge import SymbolicParametricBridge

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of Overmind interventions."""
    NONE = auto()
    BOOST_SIGNAL = auto()
    RESOURCE_REDISTRIBUTION = auto()
    MEDITATION_CALL = auto()
    ETHICAL_GUIDANCE = auto()
    CRISIS_RESPONSE = auto()
    COHERENCE_RESTORATION = auto()
    SYMBOL_BROADCAST = auto()


@dataclass
class OvermindDecision:
    """Represents a decision made by the Overmind."""
    decision_type: InterventionType
    target_area: Optional[Tuple[int, int]] = None
    intensity: float = 0.5
    duration: int = 10
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.5
    urgency: float = 0.5
    success_probability: float = 0.5
    expected_impact: Dict[str, float] = field(default_factory=dict)


@dataclass
class DetectedSymbol:
    """A symbol detected from topological field patterns."""
    symbol_type: str
    location: Tuple[int, int]
    winding_number: float
    energy: float
    interpretation: str
    age: int
    is_active: bool = True


class DharmaCompiler:
    """
    Ethical constraint checker and enforcement.

    Core Principles:
    1. No unrecoverable suffering (RX >= ε_moral)
    2. Maximize cross-scale coherence without entropy export
    3. Fairness in resource distribution
    4. No agent sacrifice for short-term gain
    """

    def __init__(self, config: DharmaConfig):
        self.config = config
        self.rx_threshold = config.rx_moral_threshold
        self.max_entropy_export = config.max_entropy_export
        self.gini_threshold = config.gini_threshold

        self.violation_log: List[Dict[str, Any]] = []
        self.evaluation_count = 0

    def evaluate_decision(self, proposed_action: OvermindDecision,
                          system_state: ColonyMetrics) -> Tuple[bool, List[str], float]:
        """
        Check if action violates ethical constraints.

        Args:
            proposed_action: Proposed Overmind intervention
            system_state: Current system metrics

        Returns:
            Tuple of (ethical: bool, violations: List[str], score: float)
        """
        self.evaluation_count += 1
        violations = []
        score = 1.0

        # Check RX threshold
        if system_state.recoverability < self.rx_threshold:
            if proposed_action.decision_type not in [
                InterventionType.CRISIS_RESPONSE,
                InterventionType.COHERENCE_RESTORATION
            ]:
                violations.append(f"RX below moral threshold ({system_state.recoverability:.2f} < {self.rx_threshold})")
                score *= 0.5

        # Check entropy export
        if system_state.entropy > 0.8:
            violations.append(f"High entropy may indicate suffering ({system_state.entropy:.2f})")
            score *= 0.7

        # Check fairness (simplified Gini check)
        if system_state.average_energy < 0.3 and proposed_action.decision_type == InterventionType.BOOST_SIGNAL:
            # Boosting signals when agents are starving may not be ethical
            violations.append("Agents need resources more than signals")
            score *= 0.8

        # Check for agent sacrifice
        if (proposed_action.decision_type == InterventionType.RESOURCE_REDISTRIBUTION and
                proposed_action.parameters.get('force_redistribution', False)):
            violations.append("Forced redistribution may harm individual agents")
            score *= 0.6

        # Log violations if any
        ethical = len(violations) == 0
        if not ethical:
            self.violation_log.append({
                'evaluation_id': self.evaluation_count,
                'action': proposed_action.decision_type.name,
                'violations': violations,
                'score': score,
                'system_rx': system_state.recoverability
            })

        return ethical, violations, score

    def suggest_ethical_alternative(self, rejected_action: OvermindDecision,
                                    system_state: ColonyMetrics) -> Optional[OvermindDecision]:
        """
        Propose modified action that satisfies constraints.
        """
        # If RX is low, prioritize crisis response
        if system_state.recoverability < self.rx_threshold:
            return OvermindDecision(
                decision_type=InterventionType.CRISIS_RESPONSE,
                intensity=0.8,
                duration=50,
                reasoning="RX below threshold - activating crisis response",
                confidence=0.9,
                urgency=1.0,
                success_probability=0.7,
                expected_impact={'rx_improvement': 0.2}
            )

        # If original was resource redistribution, suggest gentler approach
        if rejected_action.decision_type == InterventionType.RESOURCE_REDISTRIBUTION:
            return OvermindDecision(
                decision_type=InterventionType.BOOST_SIGNAL,
                intensity=0.5,
                duration=20,
                parameters={'signal_type': 'COOPERATION_CALL'},
                reasoning="Encouraging voluntary sharing instead of forced redistribution",
                confidence=0.7,
                urgency=rejected_action.urgency,
                success_probability=0.5,
                expected_impact={'cooperation_boost': 0.1}
            )

        return None


class Overmind:
    """
    Central coordinator implementing UPRT field theory.

    Monitors the colony, maintains the UPRT field, detects symbols,
    and plans ethical interventions when needed.
    """

    def __init__(self, config: OvermindConfig, dharma_config: Optional[DharmaConfig] = None,
                 qrea_config: Optional[QREAConfig] = None, field_resolution: int = 64):
        self.config = config
        self.enabled = config.enabled
        self.field_resolution = field_resolution

        # UPRT Field
        from .config import UPRTFieldConfig
        field_config = UPRTFieldConfig(field_resolution=field_resolution)
        self.field = UPRTField(field_config)

        # Field surrogate model
        self.field_surrogate = None
        if config.use_surrogate and TORCH_AVAILABLE:
            self.field_surrogate = create_field_surrogate(
                'fno', field_resolution, config.surrogate_hidden_dim
            )

        # Metrics computer
        from .config import MetricsConfig
        self.metrics_computer = MetricsComputer(MetricsConfig())
        self.current_metrics = ColonyMetrics()

        # Dharma Compiler
        self.dharma = DharmaCompiler(dharma_config or DharmaConfig())

        # Symbolic bridge
        self.symbolic_bridge = None
        if config.use_symbolic_bridge:
            self.symbolic_bridge = SymbolicParametricBridge(
                dharma_config, reasoning_mode='hybrid'
            )

        # Hypernetwork and evolution (optional)
        self.hypernetwork = None
        self.evolution_engine = None
        if qrea_config and qrea_config.use_hypernetwork and TORCH_AVAILABLE:
            from .hypernetwork import TargetArchitecture
            self.hypernetwork = create_hypernetwork(
                genome_dim=256,
                architecture=TargetArchitecture.default_agent_architecture()
            )
            self.evolution_engine = EvolutionEngine(qrea_config, genome_dim=256)

        # Symbol tracking
        self.detected_symbols: List[DetectedSymbol] = []
        self.symbol_history: deque = deque(maxlen=500)

        # Decision history
        self.decision_history: List[OvermindDecision] = []
        self.pending_interventions: List[OvermindDecision] = []

        # Intervention thresholds
        self.intervention_threshold = config.intervention_threshold
        self.meditation_threshold = config.meditation_threshold
        self.rx_threshold = config.rx_threshold

        # State tracking
        self.time_step = 0
        self.last_intervention_time = 0
        self.intervention_cooldown = 20

        # Wisdom archive
        self.wisdom_archive: List[Dict[str, Any]] = []

        logger.info(f"Overmind initialized: field_resolution={field_resolution}, "
                    f"enabled={self.enabled}")

    def sense_agents(self, agent_states: List[Dict[str, Any]]):
        """
        Aggregate agent information for field update.

        Args:
            agent_states: List of agent state dictionaries
        """
        # Compute colony metrics
        self.current_metrics = self.metrics_computer.compute_all_metrics(
            agent_states,
            time_step=self.time_step
        )

        # Update field with agent sources
        agent_sources = {}
        for agent in agent_states:
            agent_id = agent.get('agent_id', 0)
            x = int(agent.get('x', 0)) % self.field_resolution
            y = int(agent.get('y', 0)) % self.field_resolution

            # Compute source intensity from agent state
            energy = agent.get('energy', 0.5)
            mindfulness = agent.get('mindfulness_level', 0.5)
            intensity = complex(energy * 0.5, mindfulness * 0.5)

            agent_sources[agent_id] = (x, y, intensity)

        self.agent_sources = agent_sources

    def update_field(self, agent_states: Optional[List[Dict[str, Any]]] = None):
        """
        Evolve UPRT field with agent contributions.

        Args:
            agent_states: Optional updated agent states
        """
        self.time_step += 1

        # Use stored agent sources or compute from new states
        sources = getattr(self, 'agent_sources', {})
        if agent_states:
            self.sense_agents(agent_states)
            sources = self.agent_sources

        # Update field dynamics
        self.field.step_dynamics(sources)

        # Train surrogate periodically
        if self.field_surrogate and self.time_step % self.config.surrogate_update_freq == 0:
            self._train_surrogate()

    def compute_metrics(self) -> ColonyMetrics:
        """Calculate all Field Architecture metrics."""
        # Update coherence from field
        self.current_metrics.coherence = self.field.get_global_coherence()

        return self.current_metrics

    def detect_symbols(self) -> List[DetectedSymbol]:
        """
        Find topological defects and interpret as symbols.

        Returns:
            List of detected symbols
        """
        field_defects = self.field.detect_topological_defects()
        symbols = []

        for defect in field_defects:
            # Interpret defect as symbol
            if abs(defect.winding_number) > 0.5:
                symbol_type = 'vortex' if defect.winding_number > 0 else 'antivortex'
                interpretation = self._interpret_symbol(symbol_type, defect)

                symbol = DetectedSymbol(
                    symbol_type=symbol_type,
                    location=(defect.x, defect.y),
                    winding_number=defect.winding_number,
                    energy=defect.energy,
                    interpretation=interpretation,
                    age=0,
                    is_active=True
                )
                symbols.append(symbol)

        # Update symbol tracking
        self.detected_symbols = symbols
        for s in symbols:
            self.symbol_history.append({
                'time': self.time_step,
                'symbol': s
            })

        return symbols

    def _interpret_symbol(self, symbol_type: str, defect: Any) -> str:
        """Interpret a topological defect symbolically."""
        if symbol_type == 'vortex':
            if defect.energy > 0.5:
                return "High-energy coherence center - potential wisdom source"
            else:
                return "Stable attractor - agents may gather here"
        elif symbol_type == 'antivortex':
            if defect.energy > 0.5:
                return "High-energy dispersal point - potential conflict zone"
            else:
                return "Boundary marker - transition region"
        return "Unknown symbolic pattern"

    def plan_intervention(self, metrics: Optional[ColonyMetrics] = None) -> Optional[OvermindDecision]:
        """
        Decide if and how to intervene based on metrics.

        Args:
            metrics: Current colony metrics (uses stored if not provided)

        Returns:
            Intervention decision or None
        """
        if not self.enabled:
            return None

        metrics = metrics or self.current_metrics

        # Check cooldown
        if self.time_step - self.last_intervention_time < self.intervention_cooldown:
            return None

        decision = None

        # Check for crisis conditions
        if metrics.crisis_level > 0.8:
            decision = OvermindDecision(
                decision_type=InterventionType.CRISIS_RESPONSE,
                intensity=0.9,
                duration=50,
                reasoning=f"Crisis level critical: {metrics.crisis_level:.2f}",
                confidence=0.9,
                urgency=1.0,
                success_probability=0.6,
                expected_impact={'crisis_reduction': 0.3}
            )

        # Check coherence
        elif metrics.coherence < 0.3:
            decision = OvermindDecision(
                decision_type=InterventionType.COHERENCE_RESTORATION,
                intensity=0.7,
                duration=30,
                reasoning=f"Coherence dangerously low: {metrics.coherence:.2f}",
                confidence=0.8,
                urgency=0.8,
                success_probability=0.7,
                expected_impact={'coherence_boost': 0.2}
            )

        # Check recoverability
        elif metrics.recoverability < self.rx_threshold:
            decision = OvermindDecision(
                decision_type=InterventionType.CRISIS_RESPONSE,
                intensity=0.8,
                duration=40,
                reasoning=f"RX below threshold: {metrics.recoverability:.2f}",
                confidence=0.85,
                urgency=0.9,
                success_probability=0.65,
                expected_impact={'rx_improvement': 0.15}
            )

        # Check mindfulness
        elif metrics.collective_mindfulness < self.meditation_threshold:
            decision = OvermindDecision(
                decision_type=InterventionType.MEDITATION_CALL,
                intensity=0.5,
                duration=20,
                reasoning=f"Collective mindfulness low: {metrics.collective_mindfulness:.2f}",
                confidence=0.7,
                urgency=0.5,
                success_probability=0.6,
                expected_impact={'mindfulness_boost': 0.1}
            )

        # Validate decision through Dharma Compiler
        if decision:
            ethical, violations, score = self.dharma.evaluate_decision(decision, metrics)

            if not ethical:
                logger.warning(f"Decision rejected by Dharma: {violations}")
                # Try alternative
                alternative = self.dharma.suggest_ethical_alternative(decision, metrics)
                if alternative:
                    decision = alternative
                else:
                    decision = None

        return decision

    def execute_intervention(self, decision: OvermindDecision,
                             environment: Any, signal_grid: Any):
        """
        Apply Overmind intervention to the system.

        Args:
            decision: Intervention decision
            environment: Environment object
            signal_grid: WisdomSignalGrid object
        """
        if decision is None:
            return

        self.last_intervention_time = self.time_step
        self.decision_history.append(decision)

        if decision.decision_type == InterventionType.BOOST_SIGNAL:
            # Boost specific signal type
            signal_type = decision.parameters.get('signal_type', 'WISDOM_BEACON')
            from .config import WisdomSignalType
            try:
                signal_enum = WisdomSignalType[signal_type]
            except KeyError:
                signal_enum = WisdomSignalType.WISDOM_BEACON

            if decision.target_area:
                x, y = decision.target_area
            else:
                x, y = self.field_resolution // 2, self.field_resolution // 2

            signal_grid.add_signal(signal_enum, x, y, decision.intensity)

        elif decision.decision_type == InterventionType.MEDITATION_CALL:
            # Broadcast meditation synchronization signal
            from .config import WisdomSignalType
            # Broadcast across the field
            for x in range(0, self.field_resolution, 8):
                for y in range(0, self.field_resolution, 8):
                    signal_grid.add_signal(
                        WisdomSignalType.MEDITATION_SYNC,
                        x, y, decision.intensity * 0.5
                    )

        elif decision.decision_type == InterventionType.CRISIS_RESPONSE:
            # Emergency response - boost resources and compassion signals
            from .config import WisdomSignalType
            for x in range(0, self.field_resolution, 4):
                for y in range(0, self.field_resolution, 4):
                    signal_grid.add_signal(
                        WisdomSignalType.COMPASSION_GRADIENT,
                        x, y, decision.intensity * 0.3
                    )
                    environment.add_resource(x, y, 0.1)

        elif decision.decision_type == InterventionType.COHERENCE_RESTORATION:
            # Inject coherent field pattern
            for x in range(0, self.field_resolution, 4):
                for y in range(0, self.field_resolution, 4):
                    self.field.add_agent_source(
                        x, y,
                        complex(0.3, 0.3),
                        spread=3
                    )

        elif decision.decision_type == InterventionType.SYMBOL_BROADCAST:
            # Broadcast symbol interpretation to agents
            from .config import WisdomSignalType
            for symbol in self.detected_symbols:
                signal_grid.add_signal(
                    WisdomSignalType.ETHICAL_INSIGHT,
                    symbol.location[0], symbol.location[1],
                    0.7,
                    content={'interpretation': symbol.interpretation}
                )

        logger.info(f"Intervention executed: {decision.decision_type.name} "
                    f"at step {self.time_step}")

    def symbolic_reflect(self):
        """
        High-level reasoning on patterns using symbolic bridge.
        """
        if self.symbolic_bridge is None:
            return

        # Get current agent states (approximated from field)
        agent_reports = []  # Would be populated from actual agents
        field_state = self.field.get_state()

        # Analyze for anomalies
        insights = self.symbolic_bridge.analyze_anomaly(field_state, agent_reports)

        # Process insights
        if insights:
            system_state = {
                'coherence': self.current_metrics.coherence,
                'recoverability': self.current_metrics.recoverability,
                'time_step': self.time_step
            }

            # Would apply patches to actual systems
            logger.debug(f"Symbolic reflection generated {len(insights)} insights")

    def train_surrogates(self, real_field_data: Optional[np.ndarray] = None):
        """
        Train field surrogate on high-fidelity data.

        Args:
            real_field_data: Ground truth field evolution data
        """
        self._train_surrogate(real_field_data)

    def _train_surrogate(self, real_data: Optional[np.ndarray] = None):
        """Internal surrogate training."""
        if self.field_surrogate is None or not TORCH_AVAILABLE:
            return

        if real_data is None:
            # Use current field state
            field_state = self.field.field
            real_data = np.stack([field_state.real, field_state.imag])

        # Training would happen here
        # For now just log
        logger.debug(f"Surrogate training step at t={self.time_step}")

    def evolve_genomes(self, fitness_scores: List[float]):
        """
        Run evolutionary algorithm on agent genomes.

        Args:
            fitness_scores: Fitness for each genome in population
        """
        if self.evolution_engine is None:
            return

        if TORCH_AVAILABLE:
            population = self.evolution_engine.population
            if population is not None:
                fitness_tensor = torch.tensor(fitness_scores, dtype=torch.float32)
                new_pop = self.evolution_engine.evolve_population(population, fitness_tensor)
                logger.debug(f"Evolution step: gen={self.evolution_engine.generation}")

    def run_high_fidelity_field(self) -> np.ndarray:
        """Run high-fidelity field simulation for surrogate training."""
        # Step field multiple times
        field_copy = self.field.field.copy()
        for _ in range(10):
            self.field.step_dynamics()
        result = self.field.field.copy()
        self.field.field = field_copy  # Restore
        return result

    def get_state(self) -> Dict[str, Any]:
        """Get complete Overmind state."""
        return {
            'enabled': self.enabled,
            'time_step': self.time_step,
            'field_energy': self.field.get_field_energy(),
            'field_coherence': self.field.get_global_coherence(),
            'metrics': self.current_metrics.to_dict(),
            'detected_symbols': len(self.detected_symbols),
            'pending_interventions': len(self.pending_interventions),
            'decision_history_length': len(self.decision_history),
            'dharma_violations': len(self.dharma.violation_log)
        }

    def reset(self):
        """Reset Overmind state."""
        self.time_step = 0
        self.field.reset()
        self.metrics_computer.reset()
        self.current_metrics = ColonyMetrics()
        self.detected_symbols = []
        self.symbol_history.clear()
        self.decision_history = []
        self.pending_interventions = []
        self.last_intervention_time = 0
        self.wisdom_archive = []

        if self.evolution_engine:
            self.evolution_engine.population = None
            self.evolution_engine.generation = 0

        logger.info("Overmind reset")
