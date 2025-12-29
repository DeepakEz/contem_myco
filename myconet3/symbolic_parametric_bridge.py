"""
MycoNet 3.0 Symbolic-Parametric Bridge Module
=============================================

Enables high-level symbolic insights to modify low-level parameters.

Components:
- SymbolicReasoner: Interface to symbolic reasoning or LLM
- ParameterPatcher: Applies modifications to agent/field parameters
- PatchGate: Safety validator for parameter changes

This module implements the top-down symbolic insights that modify
low-level parameters with safety validation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import copy
import json
from collections import deque

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .config import DharmaConfig

logger = logging.getLogger(__name__)


class PatchType(Enum):
    """Types of parameter patches."""
    WEIGHT_MODIFICATION = auto()
    HYPERPARAMETER_CHANGE = auto()
    ARCHITECTURE_MODIFICATION = auto()
    SIGNAL_TYPE_ADDITION = auto()
    CONSTRAINT_ADDITION = auto()
    BEHAVIOR_RULE = auto()


class PatchRisk(Enum):
    """Risk levels for patches."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class SymbolicInsight:
    """Represents a symbolic insight from high-level reasoning."""
    insight_type: str  # 'pattern', 'anomaly', 'optimization', 'ethical'
    content: Dict[str, Any]
    confidence: float
    source: str  # 'llm', 'logic_engine', 'human', 'emergent'
    timestamp: int
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterPatch:
    """Represents a proposed parameter modification."""
    patch_id: str
    patch_type: PatchType
    target_component: str  # 'agent', 'field', 'overmind', 'signal_grid'
    target_path: str  # e.g., 'policy_net.layer1.weight'
    modification: Dict[str, Any]  # The actual change
    source_insight: SymbolicInsight
    risk_level: PatchRisk = PatchRisk.MEDIUM
    validated: bool = False
    applied: bool = False
    rollback_data: Optional[Dict[str, Any]] = None


@dataclass
class PatchValidationResult:
    """Result of patch validation."""
    safe: bool
    risk_assessment: Dict[str, float]
    warnings: List[str]
    blocking_issues: List[str]
    suggested_modifications: Optional[Dict[str, Any]] = None


class SymbolicReasoner:
    """
    Interface for symbolic reasoning over system state.

    Can interface with:
    - Rule-based logic engine
    - LLM for natural language reasoning
    - Pattern recognition algorithms
    """

    def __init__(self, reasoning_mode: str = 'rule_based'):
        self.reasoning_mode = reasoning_mode
        self.insight_history: deque = deque(maxlen=1000)
        self.pattern_library: Dict[str, Dict[str, Any]] = {}

        # Initialize pattern recognition rules
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize pattern recognition rules."""
        self.pattern_library = {
            'resource_scarcity': {
                'condition': lambda s: s.get('average_energy', 1.0) < 0.3,
                'insight_type': 'anomaly',
                'response': 'increase_exploration'
            },
            'coherence_collapse': {
                'condition': lambda s: s.get('coherence', 1.0) < 0.2,
                'insight_type': 'anomaly',
                'response': 'boost_synchronization'
            },
            'ethical_violation': {
                'condition': lambda s: s.get('ethics_score', 1.0) < 0.3,
                'insight_type': 'ethical',
                'response': 'strengthen_dharma_constraints'
            },
            'emergent_cooperation': {
                'condition': lambda s: s.get('cooperation_rate', 0.0) > 0.8,
                'insight_type': 'pattern',
                'response': 'reinforce_cooperation'
            },
            'population_crisis': {
                'condition': lambda s: s.get('population_trend', 0.0) < -0.2,
                'insight_type': 'anomaly',
                'response': 'survival_mode'
            }
        }

    def analyze_state(self, system_state: Dict[str, Any],
                      agent_reports: List[Dict[str, Any]]) -> List[SymbolicInsight]:
        """
        Analyze system state and generate symbolic insights.

        Args:
            system_state: Current metrics and field state
            agent_reports: Reports from individual agents

        Returns:
            List of symbolic insights
        """
        insights = []

        if self.reasoning_mode == 'rule_based':
            insights = self._rule_based_analysis(system_state, agent_reports)
        elif self.reasoning_mode == 'pattern_matching':
            insights = self._pattern_matching_analysis(system_state)
        elif self.reasoning_mode == 'hybrid':
            insights = self._rule_based_analysis(system_state, agent_reports)
            insights.extend(self._emergent_pattern_detection(system_state, agent_reports))

        # Store in history
        for insight in insights:
            self.insight_history.append(insight)

        return insights

    def _rule_based_analysis(self, system_state: Dict[str, Any],
                             agent_reports: List[Dict[str, Any]]) -> List[SymbolicInsight]:
        """Apply rule-based symbolic analysis."""
        insights = []

        for pattern_name, pattern in self.pattern_library.items():
            try:
                if pattern['condition'](system_state):
                    insight = SymbolicInsight(
                        insight_type=pattern['insight_type'],
                        content={
                            'pattern': pattern_name,
                            'response': pattern['response'],
                            'trigger_state': {k: v for k, v in system_state.items()
                                              if isinstance(v, (int, float, str, bool))}
                        },
                        confidence=0.8,
                        source='rule_based',
                        timestamp=system_state.get('time_step', 0)
                    )
                    insights.append(insight)
            except Exception as e:
                logger.warning(f"Error evaluating pattern {pattern_name}: {e}")

        return insights

    def _pattern_matching_analysis(self, system_state: Dict[str, Any]) -> List[SymbolicInsight]:
        """Apply pattern matching analysis."""
        insights = []

        # Trend analysis
        if len(self.insight_history) >= 10:
            recent = list(self.insight_history)[-10:]
            anomaly_count = sum(1 for i in recent if i.insight_type == 'anomaly')

            if anomaly_count >= 5:
                insights.append(SymbolicInsight(
                    insight_type='pattern',
                    content={
                        'pattern': 'recurring_instability',
                        'anomaly_frequency': anomaly_count / 10,
                        'response': 'system_stabilization'
                    },
                    confidence=0.7,
                    source='pattern_matching',
                    timestamp=system_state.get('time_step', 0)
                ))

        return insights

    def _emergent_pattern_detection(self, system_state: Dict[str, Any],
                                    agent_reports: List[Dict[str, Any]]) -> List[SymbolicInsight]:
        """Detect emergent patterns from agent reports."""
        insights = []

        if not agent_reports:
            return insights

        # Detect collective states
        meditation_count = sum(1 for r in agent_reports
                               if r.get('last_action') == 'MEDITATE')

        if meditation_count > len(agent_reports) * 0.5:
            insights.append(SymbolicInsight(
                insight_type='pattern',
                content={
                    'pattern': 'collective_meditation',
                    'participation_rate': meditation_count / len(agent_reports),
                    'response': 'enhance_wisdom_signals'
                },
                confidence=0.9,
                source='emergent',
                timestamp=system_state.get('time_step', 0)
            ))

        # Detect clustering behavior
        positions = [(r.get('x', 0), r.get('y', 0)) for r in agent_reports]
        if positions:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            spatial_spread = np.std(xs) + np.std(ys)

            if spatial_spread < 10:  # Highly clustered
                insights.append(SymbolicInsight(
                    insight_type='pattern',
                    content={
                        'pattern': 'spatial_clustering',
                        'spread': spatial_spread,
                        'response': 'encourage_exploration'
                    },
                    confidence=0.8,
                    source='emergent',
                    timestamp=system_state.get('time_step', 0)
                ))

        return insights

    def generate_hypothesis(self, insight: SymbolicInsight) -> str:
        """Generate natural language hypothesis from insight."""
        templates = {
            'resource_scarcity': "The colony is experiencing resource scarcity. "
                                 "Recommend increasing exploration behavior.",
            'coherence_collapse': "Colony coherence has collapsed below critical threshold. "
                                  "Synchronization signals should be boosted.",
            'ethical_violation': "Ethical violations detected. "
                                 "Dharma constraints need strengthening.",
            'collective_meditation': "Collective meditation state detected. "
                                     "This is an opportunity to enhance wisdom propagation.",
            'spatial_clustering': "Agents are spatially clustered. "
                                  "Exploration incentives may help resource discovery."
        }

        pattern = insight.content.get('pattern', 'unknown')
        return templates.get(pattern, f"Pattern detected: {pattern}")


class ParameterPatcher:
    """
    Applies modifications to agent/field parameters based on symbolic insights.
    """

    def __init__(self):
        self.patch_history: List[ParameterPatch] = []
        self.active_patches: Dict[str, ParameterPatch] = {}

        # Patch templates
        self.patch_templates = self._initialize_patch_templates()

    def _initialize_patch_templates(self) -> Dict[str, Callable]:
        """Initialize patch generation templates."""
        return {
            'increase_exploration': self._create_exploration_boost_patch,
            'boost_synchronization': self._create_sync_boost_patch,
            'strengthen_dharma_constraints': self._create_dharma_patch,
            'reinforce_cooperation': self._create_cooperation_patch,
            'survival_mode': self._create_survival_patch,
            'enhance_wisdom_signals': self._create_wisdom_patch,
            'encourage_exploration': self._create_exploration_boost_patch,
            'system_stabilization': self._create_stabilization_patch,
        }

    def generate_patch(self, insight: SymbolicInsight) -> Optional[ParameterPatch]:
        """
        Generate a parameter patch from a symbolic insight.

        Args:
            insight: Symbolic insight to convert to parameter changes

        Returns:
            ParameterPatch object or None if no patch needed
        """
        response = insight.content.get('response')
        if response not in self.patch_templates:
            logger.warning(f"No patch template for response: {response}")
            return None

        # Generate patch using template
        patch_generator = self.patch_templates[response]
        try:
            patch = patch_generator(insight)
            return patch
        except Exception as e:
            logger.error(f"Error generating patch for {response}: {e}")
            return None

    def _create_exploration_boost_patch(self, insight: SymbolicInsight) -> ParameterPatch:
        """Create patch to boost exploration behavior."""
        return ParameterPatch(
            patch_id=f"explore_{insight.timestamp}",
            patch_type=PatchType.HYPERPARAMETER_CHANGE,
            target_component='agent',
            target_path='exploration_rate',
            modification={
                'operation': 'multiply',
                'value': 1.5,
                'duration': 100  # time steps
            },
            source_insight=insight,
            risk_level=PatchRisk.LOW
        )

    def _create_sync_boost_patch(self, insight: SymbolicInsight) -> ParameterPatch:
        """Create patch to boost synchronization."""
        return ParameterPatch(
            patch_id=f"sync_{insight.timestamp}",
            patch_type=PatchType.HYPERPARAMETER_CHANGE,
            target_component='signal_grid',
            target_path='meditation_sync_strength',
            modification={
                'operation': 'multiply',
                'value': 2.0,
                'duration': 50
            },
            source_insight=insight,
            risk_level=PatchRisk.MEDIUM
        )

    def _create_dharma_patch(self, insight: SymbolicInsight) -> ParameterPatch:
        """Create patch to strengthen ethical constraints."""
        return ParameterPatch(
            patch_id=f"dharma_{insight.timestamp}",
            patch_type=PatchType.CONSTRAINT_ADDITION,
            target_component='overmind',
            target_path='dharma_compiler.rx_threshold',
            modification={
                'operation': 'set',
                'value': 0.85,  # Higher threshold
                'duration': None  # Permanent
            },
            source_insight=insight,
            risk_level=PatchRisk.MEDIUM
        )

    def _create_cooperation_patch(self, insight: SymbolicInsight) -> ParameterPatch:
        """Create patch to reinforce cooperation behavior."""
        return ParameterPatch(
            patch_id=f"coop_{insight.timestamp}",
            patch_type=PatchType.BEHAVIOR_RULE,
            target_component='agent',
            target_path='cooperation_bonus',
            modification={
                'operation': 'add',
                'value': 0.2,
                'duration': 200
            },
            source_insight=insight,
            risk_level=PatchRisk.LOW
        )

    def _create_survival_patch(self, insight: SymbolicInsight) -> ParameterPatch:
        """Create patch for survival mode."""
        return ParameterPatch(
            patch_id=f"survival_{insight.timestamp}",
            patch_type=PatchType.HYPERPARAMETER_CHANGE,
            target_component='agent',
            target_path='resource_priority',
            modification={
                'operation': 'set',
                'value': 2.0,  # Double resource priority
                'duration': 150
            },
            source_insight=insight,
            risk_level=PatchRisk.HIGH
        )

    def _create_wisdom_patch(self, insight: SymbolicInsight) -> ParameterPatch:
        """Create patch to enhance wisdom signals."""
        return ParameterPatch(
            patch_id=f"wisdom_{insight.timestamp}",
            patch_type=PatchType.HYPERPARAMETER_CHANGE,
            target_component='signal_grid',
            target_path='wisdom_beacon_strength',
            modification={
                'operation': 'multiply',
                'value': 1.5,
                'duration': 100
            },
            source_insight=insight,
            risk_level=PatchRisk.LOW
        )

    def _create_stabilization_patch(self, insight: SymbolicInsight) -> ParameterPatch:
        """Create patch for system stabilization."""
        return ParameterPatch(
            patch_id=f"stable_{insight.timestamp}",
            patch_type=PatchType.HYPERPARAMETER_CHANGE,
            target_component='field',
            target_path='damping_coefficient',
            modification={
                'operation': 'multiply',
                'value': 1.3,
                'duration': 100
            },
            source_insight=insight,
            risk_level=PatchRisk.MEDIUM
        )

    def apply_patch(self, patch: ParameterPatch, target_system: Any) -> bool:
        """
        Apply a validated patch to the target system.

        Args:
            patch: Validated patch to apply
            target_system: System component to modify

        Returns:
            True if patch applied successfully
        """
        if not patch.validated:
            logger.warning(f"Attempting to apply unvalidated patch: {patch.patch_id}")
            return False

        try:
            # Store rollback data
            patch.rollback_data = self._get_current_value(target_system, patch.target_path)

            # Apply modification
            modification = patch.modification
            operation = modification['operation']
            value = modification['value']

            current = self._get_current_value(target_system, patch.target_path)

            if operation == 'set':
                new_value = value
            elif operation == 'add':
                new_value = current + value
            elif operation == 'multiply':
                new_value = current * value
            else:
                logger.error(f"Unknown operation: {operation}")
                return False

            self._set_value(target_system, patch.target_path, new_value)

            patch.applied = True
            self.patch_history.append(patch)
            self.active_patches[patch.patch_id] = patch

            logger.info(f"Applied patch {patch.patch_id}: {patch.target_path} = {new_value}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply patch {patch.patch_id}: {e}")
            return False

    def rollback_patch(self, patch_id: str, target_system: Any) -> bool:
        """Rollback a previously applied patch."""
        if patch_id not in self.active_patches:
            logger.warning(f"Patch not found: {patch_id}")
            return False

        patch = self.active_patches[patch_id]

        if patch.rollback_data is None:
            logger.warning(f"No rollback data for patch: {patch_id}")
            return False

        try:
            self._set_value(target_system, patch.target_path, patch.rollback_data)
            del self.active_patches[patch_id]
            logger.info(f"Rolled back patch: {patch_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback patch {patch_id}: {e}")
            return False

    def _get_current_value(self, target: Any, path: str) -> Any:
        """Get current value at path in target."""
        parts = path.split('.')
        current = target

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return 1.0  # Default value

        return current

    def _set_value(self, target: Any, path: str, value: Any):
        """Set value at path in target."""
        parts = path.split('.')

        # Navigate to parent
        current = target
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]

        # Set final value
        final_part = parts[-1]
        if hasattr(current, final_part):
            setattr(current, final_part, value)
        elif isinstance(current, dict):
            current[final_part] = value


class PatchGate:
    """
    Safety validator for parameter changes.

    Ensures patches don't cause:
    - Exploding gradients
    - Ethical violations
    - Performance degradation
    - System instability
    """

    def __init__(self, dharma_config: DharmaConfig):
        self.dharma_config = dharma_config

        # Safety thresholds
        self.max_weight_change = 2.0
        self.max_hyperparameter_change = 5.0
        self.min_rx_threshold = 0.5

        # Validation history
        self.validation_history: List[PatchValidationResult] = []

    def validate(self, patch, current_system_state: Optional[Dict[str, Any]] = None) -> PatchValidationResult:
        """Convenience wrapper to validate simple patch dictionaries in tests."""
        if current_system_state is None:
            current_system_state = {}

        # If a full ParameterPatch is provided, use the rich validator
        if isinstance(patch, ParameterPatch):
            return self.validate_patch(patch, current_system_state)

        # Minimal validation for dictionary-style patches
        magnitude = float(abs(patch.get('magnitude', 0.0)))
        reversible = bool(patch.get('reversible', False))

        risk_assessment = {
            'magnitude': min(1.0, magnitude / self.max_weight_change),
            'reversibility': 0.0 if not reversible else 0.5,
        }

        warnings = []
        blocking_issues = []

        if magnitude > self.max_weight_change:
            blocking_issues.append("Modification magnitude too high")
        elif magnitude > self.max_weight_change * 0.5:
            warnings.append("High modification magnitude")

        if not reversible:
            warnings.append("Patch is not reversible")

        safe = len(blocking_issues) == 0

        result = PatchValidationResult(
            safe=safe,
            risk_assessment=risk_assessment,
            warnings=warnings,
            blocking_issues=blocking_issues,
            suggested_modifications=None
        )

        self.validation_history.append(result)
        return result

    def validate_patch(self, patch: ParameterPatch,
                       current_system_state: Dict[str, Any]) -> PatchValidationResult:
        """
        Validate a parameter patch for safety.

        Args:
            patch: Proposed parameter patch
            current_system_state: Current system metrics

        Returns:
            PatchValidationResult with safety assessment
        """
        warnings = []
        blocking_issues = []
        risk_assessment = {}

        # 1. Check modification magnitude
        magnitude_risk = self._assess_magnitude_risk(patch)
        risk_assessment['magnitude'] = magnitude_risk
        if magnitude_risk > 0.8:
            blocking_issues.append(f"Modification magnitude too high: {magnitude_risk:.2f}")
        elif magnitude_risk > 0.5:
            warnings.append(f"High modification magnitude: {magnitude_risk:.2f}")

        # 2. Check ethical compliance
        ethical_risk = self._assess_ethical_risk(patch, current_system_state)
        risk_assessment['ethical'] = ethical_risk
        if ethical_risk > 0.8:
            blocking_issues.append(f"Ethical violation risk: {ethical_risk:.2f}")
        elif ethical_risk > 0.5:
            warnings.append(f"Moderate ethical risk: {ethical_risk:.2f}")

        # 3. Check stability impact
        stability_risk = self._assess_stability_risk(patch, current_system_state)
        risk_assessment['stability'] = stability_risk
        if stability_risk > 0.8:
            blocking_issues.append(f"Stability risk too high: {stability_risk:.2f}")
        elif stability_risk > 0.5:
            warnings.append(f"Potential stability impact: {stability_risk:.2f}")

        # 4. Check reversibility
        reversibility = self._assess_reversibility(patch)
        risk_assessment['reversibility'] = 1.0 - reversibility
        if reversibility < 0.3:
            warnings.append("Patch may be difficult to reverse")

        # 5. Calculate overall safety
        overall_risk = np.mean(list(risk_assessment.values()))
        safe = len(blocking_issues) == 0 and overall_risk < 0.7

        result = PatchValidationResult(
            safe=safe,
            risk_assessment=risk_assessment,
            warnings=warnings,
            blocking_issues=blocking_issues
        )

        # Suggest modifications if unsafe
        if not safe:
            result.suggested_modifications = self._suggest_safer_alternative(patch)

        self.validation_history.append(result)

        # Update patch validation status
        if safe:
            patch.validated = True

        return result

    def _assess_magnitude_risk(self, patch: ParameterPatch) -> float:
        """Assess risk from modification magnitude."""
        modification = patch.modification
        operation = modification.get('operation', 'set')
        value = modification.get('value', 1.0)

        if operation == 'multiply':
            # Multiplicative changes: risk increases with distance from 1.0
            deviation = abs(value - 1.0)
            return min(1.0, deviation / self.max_weight_change)

        elif operation == 'add':
            # Additive changes: normalize by expected range
            return min(1.0, abs(value) / self.max_hyperparameter_change)

        elif operation == 'set':
            # Set operations: generally lower risk
            return 0.3

        return 0.5

    def _assess_ethical_risk(self, patch: ParameterPatch,
                             system_state: Dict[str, Any]) -> float:
        """Assess ethical risk of patch."""
        # Check if patch targets ethical components
        if 'dharma' in patch.target_path.lower():
            return 0.3  # Dharma modifications are somewhat risky

        # Check current RX status
        current_rx = system_state.get('recoverability', 1.0)
        if current_rx < self.dharma_config.rx_moral_threshold:
            # System already stressed - higher risk for any change
            return 0.6

        # Check patch risk level
        risk_map = {
            PatchRisk.LOW: 0.1,
            PatchRisk.MEDIUM: 0.3,
            PatchRisk.HIGH: 0.6,
            PatchRisk.CRITICAL: 0.9
        }

        return risk_map.get(patch.risk_level, 0.5)

    def _assess_stability_risk(self, patch: ParameterPatch,
                               system_state: Dict[str, Any]) -> float:
        """Assess stability impact of patch."""
        # Check current coherence
        coherence = system_state.get('coherence', 0.5)
        base_risk = 0.3 if coherence > 0.5 else 0.5

        # Certain patch types are more destabilizing
        high_impact_types = [
            PatchType.ARCHITECTURE_MODIFICATION,
            PatchType.WEIGHT_MODIFICATION
        ]

        if patch.patch_type in high_impact_types:
            base_risk += 0.3

        # Duration affects risk
        duration = patch.modification.get('duration')
        if duration is None:  # Permanent
            base_risk += 0.2

        return min(1.0, base_risk)

    def _assess_reversibility(self, patch: ParameterPatch) -> float:
        """Assess how easily the patch can be reversed."""
        # Permanent patches are harder to reverse
        if patch.modification.get('duration') is None:
            return 0.3

        # Some patch types are more reversible
        reversible_types = [
            PatchType.HYPERPARAMETER_CHANGE,
            PatchType.BEHAVIOR_RULE
        ]

        if patch.patch_type in reversible_types:
            return 0.9

        return 0.6

    def _suggest_safer_alternative(self, patch: ParameterPatch) -> Dict[str, Any]:
        """Suggest a safer alternative to the proposed patch."""
        modification = copy.deepcopy(patch.modification)

        # Reduce magnitude
        if modification.get('operation') == 'multiply':
            original_value = modification['value']
            # Move value closer to 1.0
            modification['value'] = 1.0 + (original_value - 1.0) * 0.5

        elif modification.get('operation') == 'add':
            modification['value'] = modification['value'] * 0.5

        # Add time limit if permanent
        if modification.get('duration') is None:
            modification['duration'] = 100

        return {
            'original_modification': patch.modification,
            'suggested_modification': modification,
            'changes': 'Reduced magnitude and/or added duration limit'
        }


class SymbolicParametricBridge:
    """
    Main bridge connecting symbolic reasoning to parameter modifications.

    Enables high-level symbolic insights to modify low-level parameters
    through a validated, safety-checked pipeline.
    """

    def __init__(self, dharma_config: Optional[DharmaConfig] = None,
                 reasoning_mode: str = 'hybrid'):
        self.dharma_config = dharma_config or DharmaConfig()

        # Components
        self.reasoner = SymbolicReasoner(reasoning_mode)
        self.patcher = ParameterPatcher()
        self.gate = PatchGate(self.dharma_config)

        # State
        self.pending_patches: List[ParameterPatch] = []
        self.applied_patches: List[ParameterPatch] = []

    def analyze_anomaly(self, field_state: Any,
                        agent_reports: List[Dict[str, Any]]) -> List[SymbolicInsight]:
        """
        Use symbolic reasoning to interpret unexpected patterns.

        Args:
            field_state: Current field state
            agent_reports: Reports from agents

        Returns:
            List of symbolic insights
        """
        # Extract system state for analysis
        system_state = self._extract_system_state(field_state, agent_reports)

        # Analyze with symbolic reasoner
        insights = self.reasoner.analyze_state(system_state, agent_reports)

        return insights

    def _extract_system_state(self, field_state: Any,
                              agent_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract relevant metrics from system state."""
        state = {
            'time_step': getattr(field_state, 'time_step', 0) if field_state else 0,
        }

        # Extract field metrics if available
        if field_state is not None:
            if hasattr(field_state, 'get_global_coherence'):
                state['coherence'] = field_state.get_global_coherence()
            if hasattr(field_state, 'get_field_energy'):
                state['field_energy'] = field_state.get_field_energy()

        # Extract agent statistics
        if agent_reports:
            energies = [r.get('energy', 0.5) for r in agent_reports]
            state['average_energy'] = np.mean(energies)
            state['cooperation_rate'] = np.mean([
                1 if r.get('is_cooperating', False) else 0
                for r in agent_reports
            ])

        return state

    def generate_parameter_patch(self, symbolic_insight: SymbolicInsight) -> Optional[ParameterPatch]:
        """
        Convert symbolic insight to parameter modifications.

        Args:
            symbolic_insight: High-level insight to convert

        Returns:
            Parameter patch or None if no modification needed
        """
        return self.patcher.generate_patch(symbolic_insight)

    def validate_patch(self, patch: ParameterPatch,
                       current_system_state: Dict[str, Any]) -> PatchValidationResult:
        """
        PatchGate safety check.

        Validates patch for:
        - Exploding gradients
        - Ethical violations
        - Performance degradation

        Returns:
            Validation result with safety assessment
        """
        return self.gate.validate_patch(patch, current_system_state)

    def apply_patch(self, patch: ParameterPatch, target_system: Any) -> bool:
        """
        Deploy validated patch to live system.

        Args:
            patch: Validated patch to apply
            target_system: System component to modify

        Returns:
            True if successful
        """
        if not patch.validated:
            logger.warning(f"Patch {patch.patch_id} not validated")
            return False

        success = self.patcher.apply_patch(patch, target_system)

        if success:
            self.applied_patches.append(patch)

        return success

    def process_insights(self, insights: List[SymbolicInsight],
                         system_state: Dict[str, Any],
                         target_systems: Dict[str, Any]) -> List[ParameterPatch]:
        """
        Process multiple insights through the full pipeline.

        Args:
            insights: List of symbolic insights
            system_state: Current system metrics
            target_systems: Dict mapping component names to objects

        Returns:
            List of successfully applied patches
        """
        applied = []

        for insight in insights:
            # Generate patch
            patch = self.generate_parameter_patch(insight)
            if patch is None:
                continue

            # Validate patch
            result = self.validate_patch(patch, system_state)
            if not result.safe:
                logger.warning(f"Patch {patch.patch_id} failed validation: "
                               f"{result.blocking_issues}")
                continue

            # Find target system
            target = target_systems.get(patch.target_component)
            if target is None:
                logger.warning(f"Target system not found: {patch.target_component}")
                continue

            # Apply patch
            if self.apply_patch(patch, target):
                applied.append(patch)

        return applied

    def get_active_patches(self) -> List[ParameterPatch]:
        """Get list of currently active patches."""
        return list(self.patcher.active_patches.values())

    def rollback_all(self, target_systems: Dict[str, Any]) -> int:
        """Rollback all active patches."""
        count = 0
        for patch_id, patch in list(self.patcher.active_patches.items()):
            target = target_systems.get(patch.target_component)
            if target and self.patcher.rollback_patch(patch_id, target):
                count += 1
        return count
