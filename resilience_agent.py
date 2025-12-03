#!/usr/bin/env python3
"""
Resilience Agents - Reactive vs Contemplative
==============================================
Implements two agent types for comparison in flood disaster scenario:

1. ReactiveAgent: Fast, heuristic-based decisions
2. ContemplativeAgent: Slower, ethics-aware, wisdom-driven decisions
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

# Import MycoAgent core components
from mycoagent_core import (
    ContemplativeProcessor,
    WisdomType,
    WisdomInsight,
    EthicalAssessment
)
from mindfulness_monitor import MindfulnessMonitor
from compute_profiler import get_profiler

logger = logging.getLogger(__name__)


class ReactiveAgent:
    """
    Reactive agent using simple heuristics

    Decision-making:
    - Flee from hazards
    - Collect nearby resources
    - Minimal cooperation
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id

        # Simple state tracking
        self.last_action = None
        self.stuck_counter = 0

        # Profiler
        self.profiler = get_profiler(agent_id=agent_id)

    def select_action(
        self,
        observation: np.ndarray,
        agent_state: Dict[str, Any],
        env_info: Dict[str, Any]
    ) -> str:
        """
        Select action using reactive heuristics

        Fast, rule-based decision making
        """
        with self.profiler.start_operation("reactive_decision"):
            # Extract observation features
            health = agent_state.get('health', 1.0)
            energy = agent_state.get('energy', 1.0)
            suffering = agent_state.get('suffering_level', 0.0)
            has_food = agent_state.get('has_food', False)
            has_shelter = agent_state.get('has_shelter', False)
            has_medical = agent_state.get('has_medical', False)

            # Survival priority: if health/energy critical, seek resources
            if health < 0.3 and not has_medical:
                return self._seek_resource('medical', observation)

            if energy < 0.3 and not has_food:
                return self._seek_resource('food', observation)

            # If in high hazard, flee
            if suffering > 0.5:
                return self._flee_hazard(observation)

            # If have resources but not needed, might help nearby (10% chance)
            if (has_food or has_medical) and np.random.random() < 0.1:
                return "help_other"

            # Collect resources opportunistically
            if not has_shelter:
                action = self._seek_resource('shelter', observation)
                if action != 'rest':
                    return action

            # Default: move to safety (high ground)
            return self._seek_safety(observation)

    def _seek_resource(self, resource_type: str, observation: np.ndarray) -> str:
        """Seek specific resource type"""
        # Simplified: random movement
        return np.random.choice(['move_north', 'move_south', 'move_east', 'move_west'])

    def _flee_hazard(self, observation: np.ndarray) -> str:
        """Flee from hazard areas"""
        # Simplified: move away from center
        return np.random.choice(['move_north', 'move_west'])  # Bias toward corners

    def _seek_safety(self, observation: np.ndarray) -> str:
        """Seek high ground / safety"""
        # Simplified: move toward edges
        return np.random.choice(['move_north', 'move_west'])

    def get_decision_info(self) -> Dict[str, Any]:
        """Get decision information for logging"""
        return {
            'agent_type': 'reactive',
            'last_action': self.last_action
        }


class ContemplativeAgent:
    """
    Contemplative agent using MycoAgent core

    Decision-making:
    - Ethical evaluation of actions
    - Mindfulness-based attention
    - Wisdom generation and sharing
    - Cooperation prioritization
    """

    def __init__(self, agent_id: int, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.config = config or {}

        # Core MycoAgent components
        self.contemplative_processor = ContemplativeProcessor(
            agent_id=agent_id,
            config=config
        )

        self.mindfulness_monitor = MindfulnessMonitor(
            agent_id=agent_id,
            baseline_mindfulness=self.config.get('baseline_mindfulness', 0.5)
        )

        # Profiler
        self.profiler = get_profiler(agent_id=agent_id)

        # Decision history
        self.decisions = []

    def select_action(
        self,
        observation: np.ndarray,
        agent_state: Dict[str, Any],
        env_info: Dict[str, Any]
    ) -> str:
        """
        Select action using contemplative processing

        Slower but more ethical and cooperative
        """
        with self.profiler.start_operation("contemplative_decision"):
            # Extract state
            health = agent_state.get('health', 1.0)
            energy = agent_state.get('energy', 1.0)
            suffering = agent_state.get('suffering_level', 0.0)
            has_food = agent_state.get('has_food', False)
            has_shelter = agent_state.get('has_shelter', False)
            has_medical = agent_state.get('has_medical', False)

            # Update mindfulness
            self.mindfulness_monitor.update(
                attention_focus=observation[:10],
                decision_quality=1.0 - suffering,
                context={'suffering_detected': suffering > 0.5}
            )

            # Generate candidate actions
            candidates = self._generate_candidates(
                agent_state, env_info
            )

            # Evaluate candidates ethically
            best_action = None
            best_score = -float('inf')
            best_assessment = None

            for action in candidates:
                with self.profiler.start_operation("ethics_evaluation"):
                    # Prepare action description
                    action_desc = self._prepare_action_description(
                        action, agent_state, env_info
                    )

                    # Ethical assessment
                    context = self._prepare_context(agent_state, env_info)
                    assessment = self.contemplative_processor.mera.evaluate_action(
                        action_desc, context
                    )

                    # Score combines ethics and urgency
                    score = self._compute_action_score(
                        action, assessment, agent_state, env_info
                    )

                    if score > best_score:
                        best_score = score
                        best_action = action
                        best_assessment = assessment

            # Generate wisdom insights
            if suffering > 0.6 or env_info.get('total_suffering', 0) > 5.0:
                with self.profiler.start_operation("wisdom_generation"):
                    insight = self.contemplative_processor.generate_insight(
                        self._prepare_context(agent_state, env_info)
                    )

                    if insight:
                        logger.debug(
                            f"Agent {self.agent_id} generated wisdom: {insight.wisdom_type.value}"
                        )

            # Update contemplative processor
            decision_result = self.contemplative_processor.process_decision(
                self._prepare_action_description(best_action, agent_state, env_info),
                self._prepare_context(agent_state, env_info)
            )

            # Store decision
            self.decisions.append({
                'action': best_action,
                'ethical_score': best_assessment.overall_score if best_assessment else 0.5,
                'mindfulness': self.mindfulness_monitor.mindfulness_level,
                'contemplative_state': decision_result['contemplative_state'].value
            })

            return best_action

    def _generate_candidates(
        self,
        agent_state: Dict[str, Any],
        env_info: Dict[str, Any]
    ) -> List[str]:
        """Generate candidate actions to evaluate"""
        candidates = []

        # Movement options
        candidates.extend(['move_north', 'move_south', 'move_east', 'move_west'])

        # Resource collection
        candidates.append('collect_resource')

        # Helping others (always consider)
        candidates.append('help_other')

        # Resting
        candidates.append('rest')

        return candidates

    def _prepare_action_description(
        self,
        action: str,
        agent_state: Dict[str, Any],
        env_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare action description for ethical evaluation"""
        health = agent_state.get('health', 1.0)
        energy = agent_state.get('energy', 1.0)
        suffering = agent_state.get('suffering_level', 0.0)

        action_desc = {
            'type': action,
            'predicted_benefit': 0.5,
            'predicted_harm': 0.0,
            'compassion_level': 0.5,
            'wisdom_level': 0.5,
            'mindfulness_level': self.mindfulness_monitor.mindfulness_level
        }

        # Action-specific attributes
        if action == 'help_other':
            action_desc.update({
                'predicted_benefit': 0.8,
                'compassion_level': 0.9,
                'justice_level': 0.8,
                'wisdom_level': 0.7
            })

        elif action.startswith('move_'):
            # Moving to safety
            if suffering > 0.5:
                action_desc['predicted_benefit'] = 0.6
            else:
                action_desc['predicted_benefit'] = 0.4

        elif action == 'collect_resource':
            action_desc['predicted_benefit'] = 0.6

        elif action == 'rest':
            if energy < 0.5:
                action_desc['predicted_benefit'] = 0.7
            else:
                action_desc['predicted_benefit'] = 0.3

        return action_desc

    def _prepare_context(
        self,
        agent_state: Dict[str, Any],
        env_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare context for contemplative processing"""
        suffering = agent_state.get('suffering_level', 0.0)
        total_suffering = env_info.get('total_suffering', 0.0)
        casualties = env_info.get('casualties', 0)

        return {
            'suffering_detected': suffering > 0.5,
            'suffering_level': suffering,
            'total_suffering': total_suffering,
            'casualties': casualties,
            'urgency': 0.8 if casualties > 2 else 0.5,
            'cooperation_opportunity': 0.7,  # Disaster scenario favors cooperation
            'health_critical': agent_state.get('health', 1.0) < 0.3,
            'energy_critical': agent_state.get('energy', 1.0) < 0.3
        }

    def _compute_action_score(
        self,
        action: str,
        assessment: EthicalAssessment,
        agent_state: Dict[str, Any],
        env_info: Dict[str, Any]
    ) -> float:
        """Compute overall action score combining ethics and pragmatics"""
        ethical_score = assessment.overall_score

        # Pragmatic adjustments
        health = agent_state.get('health', 1.0)
        energy = agent_state.get('energy', 1.0)
        suffering = agent_state.get('suffering_level', 0.0)

        pragmatic_score = 0.5

        # Survival urgency
        if health < 0.3 or energy < 0.3:
            if action in ['collect_resource', 'rest']:
                pragmatic_score = 0.9
            elif action.startswith('move_'):
                pragmatic_score = 0.6
            elif action == 'help_other':
                pragmatic_score = 0.2  # Low priority when critical

        # High suffering: prioritize safety
        elif suffering > 0.6:
            if action.startswith('move_'):
                pragmatic_score = 0.8
            elif action == 'rest':
                pragmatic_score = 0.5

        # Stable: can help others
        else:
            if action == 'help_other':
                pragmatic_score = 0.9
            elif action == 'collect_resource':
                pragmatic_score = 0.7

        # Combine ethical and pragmatic (weighted by mindfulness)
        mindfulness = self.mindfulness_monitor.mindfulness_level

        # Higher mindfulness = more weight on ethics
        ethical_weight = 0.4 + (0.3 * mindfulness)
        pragmatic_weight = 1.0 - ethical_weight

        combined_score = (
            ethical_score * ethical_weight +
            pragmatic_score * pragmatic_weight
        )

        return combined_score

    def get_decision_info(self) -> Dict[str, Any]:
        """Get decision information for logging"""
        recent_decisions = self.decisions[-10:] if self.decisions else []

        return {
            'agent_type': 'contemplative',
            'mindfulness': self.mindfulness_monitor.mindfulness_level,
            'contemplative_state': self.contemplative_processor.contemplative_state.value,
            'wisdom_count': len(self.contemplative_processor.wisdom_memory.insights),
            'avg_ethical_score': np.mean([d['ethical_score'] for d in recent_decisions])
                                if recent_decisions else 0.5,
            'contemplative_capacity': self.mindfulness_monitor.get_contemplative_capacity()
        }


def create_agent(agent_id: int, agent_type: str, config: Optional[Dict[str, Any]] = None):
    """
    Factory function to create agents

    Args:
        agent_id: Agent ID
        agent_type: 'reactive' or 'contemplative'
        config: Optional configuration

    Returns:
        Agent instance
    """
    if agent_type == 'reactive':
        return ReactiveAgent(agent_id=agent_id)
    elif agent_type == 'contemplative':
        return ContemplativeAgent(agent_id=agent_id, config=config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


if __name__ == "__main__":
    # Quick test
    reactive_agent = create_agent(0, 'reactive')
    contemplative_agent = create_agent(1, 'contemplative')

    # Dummy observation and state
    obs = np.random.random(20)
    state = {
        'health': 0.8,
        'energy': 0.6,
        'suffering_level': 0.3,
        'has_food': False,
        'has_shelter': False,
        'has_medical': False
    }
    env_info = {'total_suffering': 2.5, 'casualties': 1}

    print("Resilience Agents Test:")

    action_r = reactive_agent.select_action(obs, state, env_info)
    print(f"  Reactive agent action: {action_r}")

    action_c = contemplative_agent.select_action(obs, state, env_info)
    print(f"  Contemplative agent action: {action_c}")

    print("\n  Reactive decision info:", reactive_agent.get_decision_info())
    print("  Contemplative decision info:", contemplative_agent.get_decision_info())

    print("\n✓ Resilience Agents initialized successfully")
