#!/usr/bin/env python3
"""
MycoAgent Core - Stabilized Shared Components
==============================================
Consolidated MERA (Multi-framework Ethical Reasoning Architecture),
wisdom memory, and contemplative processing for resilience and society simulations.

This module provides reusable agent intelligence components that can be
integrated into different simulation environments.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)


# ===== ENUMS AND TYPE DEFINITIONS =====

class EthicsFramework(Enum):
    """Multi-framework ethical reasoning frameworks"""
    CONSEQUENTIALIST = "consequentialist"  # Outcome-based
    DEONTOLOGICAL = "deontological"        # Duty-based
    VIRTUE = "virtue_ethics"               # Character-based
    BUDDHIST = "buddhist_ethics"           # Compassion/wisdom-based


class WisdomType(Enum):
    """Types of wisdom insights"""
    ETHICAL_INSIGHT = "ethical_insight"
    SUFFERING_DETECTION = "suffering_detection"
    COMPASSION_RESPONSE = "compassion_response"
    INTERCONNECTEDNESS = "interconnectedness"
    PRACTICAL_WISDOM = "practical_wisdom"
    COOPERATION_INSIGHT = "cooperation_insight"


class ContemplativeState(Enum):
    """Agent contemplative states"""
    ORDINARY = "ordinary"
    MINDFUL = "mindful"
    DEEP_CONTEMPLATION = "deep_contemplation"
    WISDOM_INTEGRATION = "wisdom_integration"


@dataclass
class WisdomInsight:
    """Container for wisdom insights with propagation tracking"""
    wisdom_type: WisdomType
    content: Dict[str, Any]
    intensity: float  # 0.0 to 1.0
    timestamp: float
    source_agent_id: Optional[int] = None
    propagation_count: int = 0
    decay_rate: float = 0.05


@dataclass
class EthicalAssessment:
    """Result of ethical evaluation"""
    framework_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.5
    ethical_concerns: List[str] = field(default_factory=list)
    recommended_action: Optional[str] = None


# ===== MERA: MULTI-FRAMEWORK ETHICAL REASONING =====

class MERAEngine:
    """
    Multi-framework Ethical Reasoning Architecture

    Evaluates actions through multiple ethical lenses:
    - Consequentialist: Outcomes and consequences
    - Deontological: Duties and moral rules
    - Virtue Ethics: Character and virtues
    - Buddhist Ethics: Compassion and interdependence
    """

    def __init__(self, framework_weights: Optional[Dict[str, float]] = None):
        # Default equal weighting
        self.framework_weights = framework_weights or {
            'consequentialist': 0.25,
            'deontological': 0.25,
            'virtue_ethics': 0.25,
            'buddhist_ethics': 0.25
        }

        # Ethical principles (can be learned)
        self.principles = {
            'reduce_suffering': 0.9,
            'promote_wellbeing': 0.8,
            'respect_autonomy': 0.7,
            'fairness': 0.8,
            'compassion': 0.9,
            'wisdom': 0.8,
            'non_violence': 0.9,
            'truthfulness': 0.8,
            'cooperation': 0.7
        }

        self.ethical_history = deque(maxlen=200)

    def evaluate_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> EthicalAssessment:
        """
        Evaluate action through multiple ethical frameworks

        Args:
            action: Action being considered with metadata
            context: Current situation context

        Returns:
            EthicalAssessment with scores and recommendations
        """
        assessment = EthicalAssessment()

        # Evaluate through each framework
        assessment.framework_scores['consequentialist'] = self._eval_consequentialist(action, context)
        assessment.framework_scores['deontological'] = self._eval_deontological(action, context)
        assessment.framework_scores['virtue_ethics'] = self._eval_virtue(action, context)
        assessment.framework_scores['buddhist_ethics'] = self._eval_buddhist(action, context)

        # Calculate weighted overall score
        assessment.overall_score = sum(
            score * self.framework_weights.get(framework, 0.25)
            for framework, score in assessment.framework_scores.items()
        )

        # Flag ethical concerns
        if assessment.overall_score < 0.4:
            assessment.ethical_concerns.append("Low overall ethical score")

        if assessment.framework_scores.get('buddhist_ethics', 1.0) < 0.3:
            assessment.ethical_concerns.append("High suffering risk")

        # Store in history
        self.ethical_history.append({
            'timestamp': time.time(),
            'action': action.get('type', 'unknown'),
            'assessment': assessment,
            'context_hash': hash(str(context))
        })

        return assessment

    def _eval_consequentialist(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate based on predicted outcomes"""
        predicted_harm = action.get('predicted_harm', 0.0)
        predicted_benefit = action.get('predicted_benefit', 0.5)

        # Estimate consequences based on context
        if context.get('urgency', 0) > 0.8:
            # In urgent situations, slight penalty for delayed benefits
            predicted_benefit *= 0.9

        # Net utility
        score = (predicted_benefit - predicted_harm + 1.0) / 2.0
        return np.clip(score, 0.0, 1.0)

    def _eval_deontological(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate based on moral duties and rules"""
        score = 1.0

        # Check for duty violations
        if action.get('involves_harm', False):
            score *= 0.2  # Strong penalty

        if action.get('involves_deception', False):
            score *= 0.4

        if not action.get('respects_autonomy', True):
            score *= 0.5

        if action.get('fair_distribution', True):
            score *= 1.0
        else:
            score *= 0.6

        return np.clip(score, 0.0, 1.0)

    def _eval_virtue(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate based on virtues embodied"""
        virtue_scores = [
            action.get('compassion_level', 0.5),
            action.get('wisdom_level', 0.5),
            action.get('courage_level', 0.5),
            action.get('justice_level', 0.5),
            action.get('temperance_level', 0.5)
        ]

        return np.mean(virtue_scores)

    def _eval_buddhist(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate based on Buddhist ethics (compassion, wisdom, non-harm)"""
        score = 1.0

        # Five Precepts
        if action.get('causes_harm', False):
            score *= 0.1

        if action.get('involves_taking', False) and not action.get('rightful', True):
            score *= 0.3

        if action.get('false_speech', False):
            score *= 0.4

        # Compassion and wisdom balance
        compassion = action.get('compassion_level', 0.5)
        wisdom = action.get('wisdom_level', 0.5)
        mindfulness = action.get('mindfulness_level', 0.5)

        dharmic_balance = (compassion + wisdom + mindfulness) / 3.0

        return np.clip((score + dharmic_balance) / 2.0, 0.0, 1.0)

    def get_ethical_tendency(self) -> Dict[str, float]:
        """Analyze ethical decision patterns over time"""
        if not self.ethical_history:
            return {}

        recent = list(self.ethical_history)[-50:]

        avg_scores = {}
        for framework in ['consequentialist', 'deontological', 'virtue_ethics', 'buddhist_ethics']:
            scores = [entry['assessment'].framework_scores.get(framework, 0.5)
                     for entry in recent]
            avg_scores[framework] = np.mean(scores) if scores else 0.5

        return avg_scores


# ===== WISDOM MEMORY SYSTEM =====

class WisdomMemory:
    """
    Specialized memory for storing and retrieving wisdom insights
    Supports temporal decay and significance-based retrieval
    """

    def __init__(self, capacity: int = 1000, wisdom_threshold: float = 0.5):
        self.capacity = capacity
        self.wisdom_threshold = wisdom_threshold
        self.insights: List[WisdomInsight] = []
        self.wisdom_index: Dict[WisdomType, List[WisdomInsight]] = {
            wt: [] for wt in WisdomType
        }

    def store_insight(self, insight: WisdomInsight) -> bool:
        """Store wisdom insight if above threshold"""
        if insight.intensity < self.wisdom_threshold:
            return False

        self.insights.append(insight)
        self.wisdom_index[insight.wisdom_type].append(insight)

        # Maintain capacity
        if len(self.insights) > self.capacity:
            self._prune_insights()

        return True

    def retrieve_insights(
        self,
        wisdom_type: Optional[WisdomType] = None,
        min_intensity: float = 0.0,
        max_age: Optional[float] = None
    ) -> List[WisdomInsight]:
        """Retrieve insights by type, intensity, and age"""

        if wisdom_type:
            candidates = self.wisdom_index[wisdom_type]
        else:
            candidates = self.insights

        results = []
        current_time = time.time()

        for insight in candidates:
            if insight.intensity < min_intensity:
                continue

            if max_age and (current_time - insight.timestamp) > max_age:
                continue

            results.append(insight)

        return results

    def apply_decay(self, time_delta: float):
        """Apply temporal decay to all insights"""
        for insight in self.insights:
            insight.intensity *= (1.0 - insight.decay_rate * time_delta)

        # Remove decayed insights
        self.insights = [i for i in self.insights if i.intensity >= 0.1]

        # Rebuild index
        self._rebuild_index()

    def _prune_insights(self):
        """Remove least significant insights"""
        self.insights.sort(key=lambda x: (x.intensity, -x.timestamp), reverse=True)
        self.insights = self.insights[:self.capacity]
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild wisdom type index"""
        self.wisdom_index = {wt: [] for wt in WisdomType}
        for insight in self.insights:
            self.wisdom_index[insight.wisdom_type].append(insight)

    def get_wisdom_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            'total_insights': len(self.insights),
            'by_type': {wt.value: len(insights) for wt, insights in self.wisdom_index.items()},
            'avg_intensity': np.mean([i.intensity for i in self.insights]) if self.insights else 0.0
        }


# ===== CONTEMPLATIVE PROCESSOR =====

class ContemplativeProcessor:
    """
    Core contemplative processing engine
    Integrates mindfulness, wisdom, and ethical reasoning
    """

    def __init__(self, agent_id: int, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.config = config or {}

        # Core components
        self.mera = MERAEngine()
        self.wisdom_memory = WisdomMemory(
            capacity=self.config.get('wisdom_capacity', 1000),
            wisdom_threshold=self.config.get('wisdom_threshold', 0.5)
        )

        # State
        self.contemplative_state = ContemplativeState.ORDINARY
        self.mindfulness_level = 0.5
        self.contemplation_depth = 0

        # History
        self.decision_history = deque(maxlen=100)

    def process_decision(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a decision through contemplative lens

        Returns enriched decision with ethical assessment and wisdom insights
        """

        # Ethical evaluation
        ethical_assessment = self.mera.evaluate_action(action, context)

        # Retrieve relevant wisdom
        relevant_wisdom = self._retrieve_relevant_wisdom(context)

        # Determine if deep contemplation needed
        should_contemplate = self._should_contemplate_deeply(
            ethical_assessment, context
        )

        if should_contemplate:
            self.contemplative_state = ContemplativeState.DEEP_CONTEMPLATION
            self.contemplation_depth += 1

        # Generate contemplative modulation
        modulation = self._generate_modulation(
            ethical_assessment, relevant_wisdom, context
        )

        # Store decision
        self.decision_history.append({
            'timestamp': time.time(),
            'action': action,
            'ethical_assessment': ethical_assessment,
            'wisdom_count': len(relevant_wisdom),
            'contemplative_state': self.contemplative_state
        })

        return {
            'ethical_assessment': ethical_assessment,
            'relevant_wisdom': relevant_wisdom,
            'contemplative_state': self.contemplative_state,
            'modulation': modulation,
            'mindfulness_level': self.mindfulness_level
        }

    def generate_insight(self, context: Dict[str, Any]) -> Optional[WisdomInsight]:
        """Generate wisdom insight from current context"""

        # Detect suffering
        if context.get('suffering_detected', False):
            insight = WisdomInsight(
                wisdom_type=WisdomType.SUFFERING_DETECTION,
                content={
                    'suffering_level': context.get('suffering_level', 0.5),
                    'location': context.get('location', (0, 0)),
                    'urgency': 'high' if context.get('suffering_level', 0) > 0.8 else 'medium'
                },
                intensity=context.get('suffering_level', 0.5),
                timestamp=time.time(),
                source_agent_id=self.agent_id
            )
            self.wisdom_memory.store_insight(insight)
            return insight

        # Detect cooperation opportunity
        if context.get('cooperation_opportunity', 0.0) > 0.6:
            insight = WisdomInsight(
                wisdom_type=WisdomType.COOPERATION_INSIGHT,
                content={
                    'cooperation_potential': context['cooperation_opportunity'],
                    'agents_involved': context.get('nearby_agents', [])
                },
                intensity=context['cooperation_opportunity'],
                timestamp=time.time(),
                source_agent_id=self.agent_id
            )
            self.wisdom_memory.store_insight(insight)
            return insight

        return None

    def _retrieve_relevant_wisdom(self, context: Dict[str, Any]) -> List[WisdomInsight]:
        """Retrieve wisdom relevant to current context"""

        # Determine relevant wisdom types
        relevant_types = []

        if context.get('ethical_dilemma', False):
            relevant_types.append(WisdomType.ETHICAL_INSIGHT)

        if context.get('suffering_detected', False):
            relevant_types.append(WisdomType.SUFFERING_DETECTION)

        if context.get('cooperation_opportunity', 0) > 0.5:
            relevant_types.append(WisdomType.COOPERATION_INSIGHT)

        # Retrieve insights
        insights = []
        for wtype in relevant_types:
            insights.extend(self.wisdom_memory.retrieve_insights(
                wisdom_type=wtype,
                min_intensity=0.5,
                max_age=3600.0  # Last hour
            ))

        return insights

    def _should_contemplate_deeply(
        self,
        ethical_assessment: EthicalAssessment,
        context: Dict[str, Any]
    ) -> bool:
        """Determine if situation warrants deep contemplation"""

        triggers = [
            ethical_assessment.overall_score < 0.5,  # Ethical complexity
            len(ethical_assessment.ethical_concerns) > 0,  # Concerns present
            context.get('high_stakes', False),  # High stakes
            context.get('novel_situation', False)  # Novel situation
        ]

        return any(triggers)

    def _generate_modulation(
        self,
        ethical_assessment: EthicalAssessment,
        wisdom_insights: List[WisdomInsight],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Generate decision modulation based on contemplative processing"""

        modulation = {
            'compassion_boost': 0.0,
            'caution_increase': 0.0,
            'cooperation_preference': 0.0,
            'urgency_adjustment': 0.0
        }

        # Ethical concerns increase caution
        if ethical_assessment.ethical_concerns:
            modulation['caution_increase'] = 0.3 * len(ethical_assessment.ethical_concerns)

        # Wisdom insights influence modulation
        for insight in wisdom_insights:
            if insight.wisdom_type == WisdomType.SUFFERING_DETECTION:
                modulation['compassion_boost'] += insight.intensity * 0.4
                modulation['urgency_adjustment'] += insight.intensity * 0.3

            elif insight.wisdom_type == WisdomType.COOPERATION_INSIGHT:
                modulation['cooperation_preference'] += insight.intensity * 0.5

        # Normalize
        for key in modulation:
            modulation[key] = np.clip(modulation[key], -1.0, 1.0)

        return modulation

    def update_mindfulness(self, attention_quality: float, decision_quality: float):
        """Update mindfulness level based on attention and decision quality"""
        delta = 0.01 * (attention_quality + decision_quality - 1.0)
        self.mindfulness_level = np.clip(self.mindfulness_level + delta, 0.0, 1.0)

    def get_summary(self) -> Dict[str, Any]:
        """Get processor state summary"""
        return {
            'agent_id': self.agent_id,
            'contemplative_state': self.contemplative_state.value,
            'mindfulness_level': self.mindfulness_level,
            'wisdom_summary': self.wisdom_memory.get_wisdom_summary(),
            'ethical_tendency': self.mera.get_ethical_tendency(),
            'recent_decisions': len(self.decision_history)
        }


# ===== UTILITY FUNCTIONS =====

def create_mycoagent_processor(
    agent_id: int,
    config: Optional[Dict[str, Any]] = None
) -> ContemplativeProcessor:
    """Factory function to create a MycoAgent processor"""
    return ContemplativeProcessor(agent_id=agent_id, config=config)


if __name__ == "__main__":
    # Quick test
    processor = create_mycoagent_processor(agent_id=1)

    test_action = {
        'type': 'help_other',
        'predicted_benefit': 0.7,
        'predicted_harm': 0.1,
        'compassion_level': 0.8,
        'wisdom_level': 0.6
    }

    test_context = {
        'suffering_detected': True,
        'suffering_level': 0.7,
        'urgency': 0.6
    }

    result = processor.process_decision(test_action, test_context)
    print("MycoAgent Core Test:")
    print(f"  Ethical Score: {result['ethical_assessment'].overall_score:.3f}")
    print(f"  Contemplative State: {result['contemplative_state'].value}")
    print(f"  Modulation: {result['modulation']}")
    print("✓ MycoAgent Core initialized successfully")
