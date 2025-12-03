#!/usr/bin/env python3
"""
Mindfulness Monitor - Refined Attention and Awareness Tracking
===============================================================
Enhanced mindfulness monitoring with:
- Multi-dimensional attention tracking
- Awareness quality metrics
- Contemplative depth measurement
- Mindfulness skill progression
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class MindfulnessState(Enum):
    """Levels of mindfulness"""
    DISTRACTED = "distracted"          # < 0.3
    ORDINARY = "ordinary"              # 0.3 - 0.6
    MINDFUL = "mindful"                # 0.6 - 0.8
    DEEPLY_MINDFUL = "deeply_mindful"  # > 0.8


@dataclass
class AttentionMetrics:
    """Multi-dimensional attention metrics"""
    focus_coherence: float = 0.5        # How unified is attention
    attention_stability: float = 0.5     # How steady over time
    awareness_breadth: float = 0.5       # How much is noticed
    meta_awareness: float = 0.5          # Awareness of awareness
    present_moment: float = 0.5          # Grounded in present
    equanimity: float = 0.5              # Non-reactive balance


@dataclass
class MindfulnessEvent:
    """Record of significant mindfulness event"""
    timestamp: float
    event_type: str  # 'distraction', 'return_to_focus', 'insight', 'deep_mindfulness'
    mindfulness_level: float
    context: Dict[str, Any] = field(default_factory=dict)


class MindfulnessMonitor:
    """
    Refined mindfulness monitoring system

    Tracks:
    - Attention quality and stability
    - Awareness depth
    - Mindfulness skill development
    - Contemplative states
    """

    def __init__(
        self,
        agent_id: int,
        capacity: int = 200,
        baseline_mindfulness: float = 0.5
    ):
        self.agent_id = agent_id
        self.capacity = capacity

        # Core state
        self.mindfulness_level = baseline_mindfulness
        self.mindfulness_state = self._classify_state(baseline_mindfulness)

        # Attention metrics
        self.current_metrics = AttentionMetrics()

        # History tracking
        self.attention_history = deque(maxlen=capacity)
        self.mindfulness_history = deque(maxlen=capacity)
        self.event_history = deque(maxlen=100)

        # Skill development
        self.baseline_mindfulness = baseline_mindfulness
        self.peak_mindfulness = baseline_mindfulness
        self.mindfulness_growth_rate = 0.0

        # Thresholds (can be tuned)
        self.mindful_threshold = 0.6
        self.deeply_mindful_threshold = 0.8

        # Statistics
        self.distraction_count = 0
        self.return_to_focus_count = 0
        self.time_in_mindful_state = 0.0
        self.last_update_time = time.time()

    def update(
        self,
        attention_focus: np.ndarray,
        decision_quality: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Update mindfulness state based on attention patterns

        Args:
            attention_focus: Attention distribution (array)
            decision_quality: Quality of recent decision (0-1)
            context: Additional context (optional)
        """
        current_time = time.time()
        time_delta = current_time - self.last_update_time

        context = context or {}

        # Calculate attention metrics
        self._update_attention_metrics(attention_focus, decision_quality, context)

        # Update mindfulness level
        self._update_mindfulness_level(time_delta)

        # Check for mindfulness events
        self._detect_events(context)

        # Record history
        self.attention_history.append({
            'timestamp': current_time,
            'metrics': self.current_metrics,
            'mindfulness_level': self.mindfulness_level
        })

        self.mindfulness_history.append({
            'timestamp': current_time,
            'level': self.mindfulness_level,
            'state': self.mindfulness_state.value
        })

        # Update statistics
        if self.mindfulness_level >= self.mindful_threshold:
            self.time_in_mindful_state += time_delta

        # Track peak
        if self.mindfulness_level > self.peak_mindfulness:
            self.peak_mindfulness = self.mindfulness_level

        # Update growth rate
        self._update_growth_rate()

        self.last_update_time = current_time

    def _update_attention_metrics(
        self,
        attention_focus: np.ndarray,
        decision_quality: float,
        context: Dict[str, Any]
    ):
        """Update multi-dimensional attention metrics"""

        # Focus coherence: How unified is attention
        if len(attention_focus) > 0:
            # Low variance = high coherence
            self.current_metrics.focus_coherence = 1.0 - min(1.0, np.var(attention_focus) * 2)
        else:
            self.current_metrics.focus_coherence = 0.5

        # Attention stability: Compare to recent history
        if len(self.attention_history) >= 10:
            recent_coherence = [
                h['metrics'].focus_coherence
                for h in list(self.attention_history)[-10:]
            ]
            coherence_variance = np.var(recent_coherence)
            self.current_metrics.attention_stability = 1.0 - min(1.0, coherence_variance * 3)
        else:
            self.current_metrics.attention_stability = 0.5

        # Awareness breadth: How much is noticed
        # Approximate from attention distribution entropy
        if len(attention_focus) > 1:
            # Normalize
            attention_norm = attention_focus / (np.sum(attention_focus) + 1e-8)
            # Calculate entropy
            entropy = -np.sum(attention_norm * np.log(attention_norm + 1e-8))
            max_entropy = np.log(len(attention_focus))
            self.current_metrics.awareness_breadth = entropy / max_entropy if max_entropy > 0 else 0.5
        else:
            self.current_metrics.awareness_breadth = 0.3

        # Meta-awareness: Awareness of being aware
        # Proxy: combination of decision quality and attention stability
        self.current_metrics.meta_awareness = (
            0.5 * decision_quality +
            0.5 * self.current_metrics.attention_stability
        )

        # Present moment: Not dwelling on past/future
        # Proxy: high focus coherence + high decision quality
        self.current_metrics.present_moment = (
            0.6 * self.current_metrics.focus_coherence +
            0.4 * decision_quality
        )

        # Equanimity: Non-reactive balance
        # Proxy: stability combined with breadth
        self.current_metrics.equanimity = (
            0.5 * self.current_metrics.attention_stability +
            0.3 * self.current_metrics.awareness_breadth +
            0.2 * self.current_metrics.meta_awareness
        )

    def _update_mindfulness_level(self, time_delta: float):
        """Update overall mindfulness level"""

        # Aggregate attention metrics
        metrics_aggregate = (
            self.current_metrics.focus_coherence * 0.25 +
            self.current_metrics.attention_stability * 0.20 +
            self.current_metrics.meta_awareness * 0.25 +
            self.current_metrics.present_moment * 0.20 +
            self.current_metrics.equanimity * 0.10
        )

        # Learning rate depends on current level
        # Easier to improve when low, harder when high
        if self.mindfulness_level < 0.5:
            learning_rate = 0.05
        elif self.mindfulness_level < 0.7:
            learning_rate = 0.03
        else:
            learning_rate = 0.01

        # Update with momentum
        delta = learning_rate * (metrics_aggregate - self.mindfulness_level)

        # Natural decay if not maintained
        decay = 0.01 * time_delta

        self.mindfulness_level = np.clip(
            self.mindfulness_level + delta - decay,
            0.0,
            1.0
        )

        # Update state classification
        old_state = self.mindfulness_state
        self.mindfulness_state = self._classify_state(self.mindfulness_level)

        # Log state transitions
        if old_state != self.mindfulness_state:
            logger.debug(
                f"Agent {self.agent_id} mindfulness: {old_state.value} -> {self.mindfulness_state.value}"
            )

    def _classify_state(self, level: float) -> MindfulnessState:
        """Classify mindfulness state from level"""
        if level < 0.3:
            return MindfulnessState.DISTRACTED
        elif level < 0.6:
            return MindfulnessState.ORDINARY
        elif level < 0.8:
            return MindfulnessState.MINDFUL
        else:
            return MindfulnessState.DEEPLY_MINDFUL

    def _detect_events(self, context: Dict[str, Any]):
        """Detect significant mindfulness events"""

        # Distraction event
        if (self.mindfulness_level < 0.3 and
            len(self.mindfulness_history) > 0 and
            self.mindfulness_history[-1]['level'] >= 0.3):

            self.distraction_count += 1
            self._record_event('distraction', context)

        # Return to focus
        if (self.mindfulness_level >= 0.6 and
            len(self.mindfulness_history) > 0 and
            self.mindfulness_history[-1]['level'] < 0.6):

            self.return_to_focus_count += 1
            self._record_event('return_to_focus', context)

        # Deep mindfulness achievement
        if (self.mindfulness_level >= 0.8 and
            len(self.mindfulness_history) > 0 and
            self.mindfulness_history[-1]['level'] < 0.8):

            self._record_event('deep_mindfulness', context)

        # Insight event (high meta-awareness spike)
        if self.current_metrics.meta_awareness > 0.85:
            self._record_event('insight', context)

    def _record_event(self, event_type: str, context: Dict[str, Any]):
        """Record a mindfulness event"""
        event = MindfulnessEvent(
            timestamp=time.time(),
            event_type=event_type,
            mindfulness_level=self.mindfulness_level,
            context=context
        )
        self.event_history.append(event)

        logger.debug(f"Agent {self.agent_id} mindfulness event: {event_type}")

    def _update_growth_rate(self):
        """Calculate mindfulness growth rate"""
        if len(self.mindfulness_history) < 20:
            self.mindfulness_growth_rate = 0.0
            return

        # Compare recent average to earlier average
        recent = [h['level'] for h in list(self.mindfulness_history)[-10:]]
        earlier = [h['level'] for h in list(self.mindfulness_history)[-20:-10]]

        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)

        self.mindfulness_growth_rate = recent_avg - earlier_avg

    def is_mindful(self) -> bool:
        """Check if agent is currently mindful"""
        return self.mindfulness_level >= self.mindful_threshold

    def is_deeply_mindful(self) -> bool:
        """Check if agent is in deep mindfulness"""
        return self.mindfulness_level >= self.deeply_mindful_threshold

    def get_attention_quality(self) -> float:
        """Get overall attention quality score"""
        return (
            self.current_metrics.focus_coherence * 0.4 +
            self.current_metrics.attention_stability * 0.3 +
            self.current_metrics.meta_awareness * 0.3
        )

    def get_contemplative_capacity(self) -> float:
        """
        Get capacity for deep contemplation

        Higher when:
        - High mindfulness
        - Stable attention
        - Good meta-awareness
        """
        return (
            self.mindfulness_level * 0.5 +
            self.current_metrics.attention_stability * 0.3 +
            self.current_metrics.meta_awareness * 0.2
        )

    def get_event_summary(self) -> Dict[str, int]:
        """Get summary of mindfulness events"""
        event_counts = {}
        for event in self.event_history:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        return event_counts

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive mindfulness statistics"""
        recent_levels = [h['level'] for h in list(self.mindfulness_history)[-50:]]

        return {
            'agent_id': self.agent_id,
            'current_mindfulness': self.mindfulness_level,
            'mindfulness_state': self.mindfulness_state.value,
            'baseline_mindfulness': self.baseline_mindfulness,
            'peak_mindfulness': self.peak_mindfulness,
            'growth_rate': self.mindfulness_growth_rate,
            'avg_recent_mindfulness': np.mean(recent_levels) if recent_levels else 0.5,
            'attention_quality': self.get_attention_quality(),
            'contemplative_capacity': self.get_contemplative_capacity(),
            'time_in_mindful_state': self.time_in_mindful_state,
            'distraction_count': self.distraction_count,
            'return_to_focus_count': self.return_to_focus_count,
            'event_summary': self.get_event_summary(),
            'current_metrics': {
                'focus_coherence': self.current_metrics.focus_coherence,
                'attention_stability': self.current_metrics.attention_stability,
                'awareness_breadth': self.current_metrics.awareness_breadth,
                'meta_awareness': self.current_metrics.meta_awareness,
                'present_moment': self.current_metrics.present_moment,
                'equanimity': self.current_metrics.equanimity
            }
        }

    def reset(self):
        """Reset mindfulness state to baseline"""
        self.mindfulness_level = self.baseline_mindfulness
        self.mindfulness_state = self._classify_state(self.baseline_mindfulness)
        self.current_metrics = AttentionMetrics()
        self.attention_history.clear()
        self.mindfulness_history.clear()
        self.event_history.clear()
        self.distraction_count = 0
        self.return_to_focus_count = 0
        self.time_in_mindful_state = 0.0


# ===== MONITOR FACTORY =====

def create_mindfulness_monitor(
    agent_id: int,
    baseline: float = 0.5,
    capacity: int = 200
) -> MindfulnessMonitor:
    """Factory function to create a mindfulness monitor"""
    return MindfulnessMonitor(
        agent_id=agent_id,
        capacity=capacity,
        baseline_mindfulness=baseline
    )


if __name__ == "__main__":
    # Quick test
    monitor = create_mindfulness_monitor(agent_id=1, baseline=0.5)

    # Simulate attention updates
    for i in range(20):
        # Good attention
        attention = np.random.dirichlet([5, 1, 1, 1, 1])  # Focused
        decision_quality = 0.7 + 0.2 * np.random.random()

        monitor.update(attention, decision_quality)

    stats = monitor.get_statistics()

    print("Mindfulness Monitor Test:")
    print(f"  Current mindfulness: {stats['current_mindfulness']:.3f}")
    print(f"  State: {stats['mindfulness_state']}")
    print(f"  Attention quality: {stats['attention_quality']:.3f}")
    print(f"  Contemplative capacity: {stats['contemplative_capacity']:.3f}")
    print(f"  Focus coherence: {stats['current_metrics']['focus_coherence']:.3f}")
    print(f"  Meta-awareness: {stats['current_metrics']['meta_awareness']:.3f}")
    print("✓ Mindfulness Monitor initialized successfully")
