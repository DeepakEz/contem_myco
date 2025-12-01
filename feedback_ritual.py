#!/usr/bin/env python3
"""
MODULE 3: AGENT FEEDBACK & RITUAL SYSTEMS
Advanced agent feedback mechanisms and collective ritual coordination
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import random
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

# ===== ENUMS AND DATA STRUCTURES =====

class RitualType(Enum):
    """Types of collective rituals"""
    SYNCHRONIZED_MEDITATION = "synchronized_meditation"
    WISDOM_CIRCLE = "wisdom_circle"
    HARMONY_RESONANCE = "harmony_resonance"
    COLLECTIVE_INSIGHT = "collective_insight"
    CONFLICT_RESOLUTION_CIRCLE = "conflict_resolution_circle"
    ENERGY_SHARING_RITUAL = "energy_sharing_ritual"
    INNOVATION_CATALYST = "innovation_catalyst"
    CONTEMPLATIVE_SILENCE = "contemplative_silence"

class FeedbackType(Enum):
    """Types of agent feedback"""
    MINDFULNESS_BOOST = "mindfulness_boost"
    COOPERATION_ENHANCEMENT = "cooperation_enhancement"
    WISDOM_RECEPTIVITY = "wisdom_receptivity"
    ENERGY_OPTIMIZATION = "energy_optimization"
    STRESS_REDUCTION = "stress_reduction"
    LEARNING_ACCELERATION = "learning_acceleration"
    EMOTIONAL_REGULATION = "emotional_regulation"
    INNOVATION_STIMULATION = "innovation_stimulation"

@dataclass
class FeedbackApplication:
    """Record of feedback applied to agents"""
    agent_id: str
    feedback_type: FeedbackType
    intensity: float
    timestamp: float
    baseline_values: Dict[str, float]
    target_values: Dict[str, float]
    actual_change: Dict[str, float] = field(default_factory=dict)
    effectiveness_score: float = 0.0

@dataclass
class RitualExecution:
    """Record of ritual execution"""
    ritual_type: RitualType
    participants: List[str]
    duration: float
    effectiveness: float
    emergence_effects: Dict[str, float]
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)

# ===== AGENT FEEDBACK INTERFACE =====

class AgentFeedbackInterface:
    """Advanced agent feedback system with adaptive delivery mechanisms"""
    
    def __init__(self):
        # Feedback tracking
        self.feedback_history = deque(maxlen=5000)
        self.agent_response_tracking = defaultdict(list)
        self.feedback_effectiveness = defaultdict(list)
        
        # Adaptive feedback parameters
        self.adaptation_rates = {
            FeedbackType.MINDFULNESS_BOOST: 0.1,
            FeedbackType.COOPERATION_ENHANCEMENT: 0.08,
            FeedbackType.WISDOM_RECEPTIVITY: 0.12,
            FeedbackType.ENERGY_OPTIMIZATION: 0.15,
            FeedbackType.STRESS_REDUCTION: 0.2,
            FeedbackType.LEARNING_ACCELERATION: 0.05,
            FeedbackType.EMOTIONAL_REGULATION: 0.1,
            FeedbackType.INNOVATION_STIMULATION: 0.06
        }
        
        # Feedback neural network for personalization
        self.feedback_network = self._create_feedback_network()
        
        # Agent personality models
        self.agent_profiles = {}
        self.receptivity_models = {}
        
    def _create_feedback_network(self) -> nn.Module:
        """Create neural network for personalized feedback delivery"""
        
        class FeedbackPersonalizationNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                # Input: agent state + feedback history + context
                self.agent_encoder = nn.Linear(12, 64)     # Agent attributes
                self.history_encoder = nn.Linear(20, 64)   # Feedback history
                self.context_encoder = nn.Linear(8, 32)    # Context features
                
                self.fusion_layer = nn.Linear(160, 128)
                self.effectiveness_predictor = nn.Linear(128, 1)
                self.intensity_adjuster = nn.Linear(128, 1)
                
            def forward(self, agent_features, history_features, context_features):
                agent_emb = F.relu(self.agent_encoder(agent_features))
                history_emb = F.relu(self.history_encoder(history_features))
                context_emb = F.relu(self.context_encoder(context_features))
                
                # Fuse all features
                combined = torch.cat([agent_emb, history_emb, context_emb], dim=-1)
                fused = F.relu(self.fusion_layer(combined))
                
                # Predict effectiveness and suggest intensity adjustment
                effectiveness = torch.sigmoid(self.effectiveness_predictor(fused))
                intensity_adj = torch.tanh(self.intensity_adjuster(fused))
                
                return effectiveness, intensity_adj
        
        return FeedbackPersonalizationNetwork()
    
    def apply_overmind_feedback(self, agents: List, feedback_type: str, 
                              intensity: float, overmind_action) -> Dict[str, Any]:
        """Apply feedback from overmind to agents with advanced personalization"""
        
        try:
            # Convert string to enum
            if isinstance(feedback_type, str):
                feedback_type = FeedbackType(feedback_type)
            
            results = {
                'total_agents_affected': 0,
                'feedback_applications': [],
                'overall_effectiveness': 0.0,
                'personalization_used': True,
                'adaptation_adjustments': 0
            }
            
            effectiveness_scores = []
            
            for agent in agents:
                try:
                    # Create agent profile if not exists
                    if not hasattr(agent, 'id'):
                        agent.id = id(agent)  # Fallback ID
                    
                    if agent.id not in self.agent_profiles:
                        self._create_agent_profile(agent)
                    
                    # Get personalized feedback parameters
                    personalized_intensity, predicted_effectiveness = self._personalize_feedback(
                        agent, feedback_type, intensity
                    )
                    
                    # Record baseline values
                    baseline_values = self._extract_agent_baseline(agent)
                    
                    # Apply feedback with personalization
                    actual_change = self._apply_feedback_to_agent(
                        agent, feedback_type, personalized_intensity
                    )
                    
                    # Calculate target values
                    target_values = self._calculate_target_values(
                        baseline_values, feedback_type, personalized_intensity
                    )
                    
                    # Record feedback application
                    feedback_app = FeedbackApplication(
                        agent_id=str(agent.id),
                        feedback_type=feedback_type,
                        intensity=personalized_intensity,
                        timestamp=time.time(),
                        baseline_values=baseline_values,
                        target_values=target_values,
                        actual_change=actual_change,
                        effectiveness_score=predicted_effectiveness
                    )
                    
                    self.feedback_history.append(feedback_app)
                    self.agent_response_tracking[agent.id].append(feedback_app)
                    
                    results['feedback_applications'].append({
                        'agent_id': str(agent.id),
                        'intensity_used': personalized_intensity,
                        'predicted_effectiveness': predicted_effectiveness,
                        'actual_change': actual_change
                    })
                    
                    effectiveness_scores.append(predicted_effectiveness)
                    results['total_agents_affected'] += 1
                    
                    # Update agent profile
                    self._update_agent_profile(agent, feedback_app)
                    
                except Exception as e:
                    logger.warning(f"Failed to apply feedback to agent {getattr(agent, 'id', 'unknown')}: {e}")
                    continue
            
            # Calculate overall effectiveness
            if effectiveness_scores:
                results['overall_effectiveness'] = float(np.mean(effectiveness_scores))
            
            # Update adaptation rates based on effectiveness
            self._update_adaptation_rates(feedback_type, effectiveness_scores)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to apply overmind feedback: {e}")
            return self._fallback_feedback_application(agents, feedback_type, intensity)
    
    def _create_agent_profile(self, agent):
        """Create comprehensive profile for agent"""
        
        profile = {
            'baseline_attributes': self._extract_agent_baseline(agent),
            'receptivity_scores': {ft: 0.5 for ft in FeedbackType},
            'adaptation_speed': 0.5,
            'feedback_retention': 0.7,
            'response_volatility': 0.3,
            'learning_curve': [],
            'personality_factors': self._analyze_agent_personality(agent)
        }
        
        self.agent_profiles[agent.id] = profile
    
    def _extract_agent_baseline(self, agent) -> Dict[str, float]:
        """Extract baseline values from agent"""
        
        return {
            'energy': getattr(agent, 'energy', 0.5),
            'health': getattr(agent, 'health', 0.5),
            'mindfulness_level': getattr(agent, 'mindfulness_level', 0.5),
            'cooperation_tendency': getattr(agent, 'cooperation_tendency', 0.5),
            'conflict_tendency': getattr(agent, 'conflict_tendency', 0.3),
            'learning_rate': getattr(agent, 'learning_rate', 0.5),
            'emotional_stability': getattr(agent, 'emotional_stability', 0.5),
            'stress_level': getattr(agent, 'stress_level', 0.3),
            'innovation_capacity': getattr(agent, 'innovation_capacity', 0.4),
            'wisdom_accumulated': getattr(agent, 'wisdom_accumulated', 0.0)
        }
    
    def _analyze_agent_personality(self, agent) -> Dict[str, float]:
        """Analyze agent personality for feedback personalization"""
        
        # Extract personality factors from agent attributes
        openness = getattr(agent, 'innovation_capacity', 0.4)
        conscientiousness = getattr(agent, 'cooperation_tendency', 0.5)
        extraversion = 1.0 - getattr(agent, 'mindfulness_level', 0.5)  # Inverse relationship
        agreeableness = 1.0 - getattr(agent, 'conflict_tendency', 0.3)
        neuroticism = getattr(agent, 'stress_level', 0.3)
        
        return {
            'openness': openness,
            'conscientiousness': conscientiousness,
            'extraversion': extraversion,
            'agreeableness': agreeableness,
            'neuroticism': neuroticism
        }
    
    def _personalize_feedback(self, agent, feedback_type: FeedbackType, 
                            base_intensity: float) -> Tuple[float, float]:
        """Personalize feedback for specific agent"""
        
        try:
            # Get agent features
            agent_features = self._get_agent_features(agent)
            history_features = self._get_agent_history_features(agent)
            context_features = self._get_context_features()
            
            # Use neural network for personalization
            with torch.no_grad():
                effectiveness, intensity_adjustment = self.feedback_network(
                    agent_features, history_features, context_features
                )
            
            # Adjust intensity
            adjusted_intensity = base_intensity * (1.0 + intensity_adjustment.item() * 0.3)
            adjusted_intensity = np.clip(adjusted_intensity, 0.1, 2.0)
            
            return adjusted_intensity, effectiveness.item()
            
        except Exception as e:
            logger.warning(f"Failed to personalize feedback: {e}")
            # Fallback to rule-based personalization
            return self._rule_based_personalization(agent, feedback_type, base_intensity)
    
    def _get_agent_features(self, agent) -> torch.Tensor:
        """Extract agent features for neural network"""
        
        baseline = self._extract_agent_baseline(agent)
        personality = self.agent_profiles.get(agent.id, {}).get('personality_factors', {})
        
        features = [
            baseline.get('energy', 0.5),
            baseline.get('health', 0.5),
            baseline.get('mindfulness_level', 0.5),
            baseline.get('cooperation_tendency', 0.5),
            baseline.get('conflict_tendency', 0.3),
            baseline.get('learning_rate', 0.5),
            baseline.get('emotional_stability', 0.5),
            baseline.get('stress_level', 0.3),
            personality.get('openness', 0.5),
            personality.get('conscientiousness', 0.5),
            personality.get('agreeableness', 0.5),
            personality.get('neuroticism', 0.3)
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _get_agent_history_features(self, agent) -> torch.Tensor:
        """Get agent's feedback history features"""
        
        recent_feedback = self.agent_response_tracking.get(agent.id, [])[-10:]
        
        if not recent_feedback:
            return torch.zeros(20)
        
        features = []
        
        # Average effectiveness of recent feedback
        avg_effectiveness = np.mean([f.effectiveness_score for f in recent_feedback])
        features.extend([avg_effectiveness] * 5)
        
        # Feedback type distribution
        type_counts = defaultdict(int)
        for feedback in recent_feedback:
            type_counts[feedback.feedback_type] += 1
        
        # Normalize counts
        total_feedback = len(recent_feedback)
        for fb_type in FeedbackType:
            normalized_count = type_counts[fb_type] / max(1, total_feedback)
            features.append(normalized_count)
        
        # Recent intensity levels
        recent_intensities = [f.intensity for f in recent_feedback[-5:]]
        while len(recent_intensities) < 5:
            recent_intensities.append(0.0)
        features.extend(recent_intensities)
        
        return torch.tensor(features[:20], dtype=torch.float32)
    
    def _get_context_features(self) -> torch.Tensor:
        """Get current context features"""
        
        # Simplified context features
        features = [
            0.5,  # crisis_level (would be passed from overmind)
            0.5,  # cooperation_rate
            0.3,  # conflict_rate
            0.5,  # collective_mindfulness
            0.7,  # resource_abundance
            0.2,  # environmental_stress
            0.6,  # social_harmony
            0.4   # innovation_pressure
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _rule_based_personalization(self, agent, feedback_type: FeedbackType, 
                                   base_intensity: float) -> Tuple[float, float]:
        """Fallback rule-based personalization"""
        
        profile = self.agent_profiles.get(agent.id, {})
        personality = profile.get('personality_factors', {})
        
        # Adjust based on personality and feedback type
        adjustment_factor = 1.0
        
        if feedback_type == FeedbackType.MINDFULNESS_BOOST:
            # More effective for agents with higher openness
            adjustment_factor *= (1.0 + personality.get('openness', 0.5) * 0.3)
        elif feedback_type == FeedbackType.COOPERATION_ENHANCEMENT:
            # More effective for agents with higher agreeableness
            adjustment_factor *= (1.0 + personality.get('agreeableness', 0.5) * 0.4)
        elif feedback_type == FeedbackType.STRESS_REDUCTION:
            # More needed for agents with higher neuroticism
            adjustment_factor *= (1.0 + personality.get('neuroticism', 0.3) * 0.5)
        
        # Adjust based on recent feedback effectiveness
        recent_effectiveness = self._get_recent_effectiveness(agent.id)
        if recent_effectiveness < 0.3:
            adjustment_factor *= 1.2  # Increase intensity if recent feedback was ineffective
        elif recent_effectiveness > 0.8:
            adjustment_factor *= 0.8  # Decrease intensity if recent feedback was very effective
        
        adjusted_intensity = base_intensity * adjustment_factor
        adjusted_intensity = np.clip(adjusted_intensity, 0.1, 2.0)
        
        predicted_effectiveness = 0.5 + (adjustment_factor - 1.0) * 0.3
        predicted_effectiveness = np.clip(predicted_effectiveness, 0.1, 0.9)
        
        return adjusted_intensity, predicted_effectiveness
    
    def _get_recent_effectiveness(self, agent_id: str) -> float:
        """Get recent feedback effectiveness for agent"""
        
        recent_feedback = self.agent_response_tracking.get(agent_id, [])[-5:]
        if not recent_feedback:
            return 0.5
        
        return np.mean([f.effectiveness_score for f in recent_feedback])
    
    def _apply_feedback_to_agent(self, agent, feedback_type: FeedbackType, 
                               intensity: float) -> Dict[str, float]:
        """Apply specific feedback to agent"""
        
        changes = {}
        adaptation_rate = self.adaptation_rates[feedback_type]
        
        try:
            if feedback_type == FeedbackType.MINDFULNESS_BOOST:
                old_value = getattr(agent, 'mindfulness_level', 0.5)
                change = intensity * adaptation_rate
                new_value = np.clip(old_value + change, 0.0, 1.0)
                agent.mindfulness_level = new_value
                changes['mindfulness_level'] = new_value - old_value
                
            elif feedback_type == FeedbackType.COOPERATION_ENHANCEMENT:
                old_value = getattr(agent, 'cooperation_tendency', 0.5)
                change = intensity * adaptation_rate
                new_value = np.clip(old_value + change, 0.0, 1.0)
                agent.cooperation_tendency = new_value
                changes['cooperation_tendency'] = new_value - old_value
                
                # Secondary effect on conflict tendency
                conflict_old = getattr(agent, 'conflict_tendency', 0.3)
                conflict_change = -intensity * adaptation_rate * 0.5
                conflict_new = np.clip(conflict_old + conflict_change, 0.0, 1.0)
                agent.conflict_tendency = conflict_new
                changes['conflict_tendency'] = conflict_new - conflict_old
                
            elif feedback_type == FeedbackType.ENERGY_OPTIMIZATION:
                old_value = getattr(agent, 'energy', 0.5)
                change = intensity * adaptation_rate
                new_value = np.clip(old_value + change, 0.0, 1.0)
                agent.energy = new_value
                changes['energy'] = new_value - old_value
                
            elif feedback_type == FeedbackType.STRESS_REDUCTION:
                old_value = getattr(agent, 'stress_level', 0.3)
                change = -intensity * adaptation_rate  # Negative change reduces stress
                new_value = np.clip(old_value + change, 0.0, 1.0)
                agent.stress_level = new_value
                changes['stress_level'] = new_value - old_value
                
                # Secondary effect on emotional stability
                stability_old = getattr(agent, 'emotional_stability', 0.5)
                stability_change = intensity * adaptation_rate * 0.3
                stability_new = np.clip(stability_old + stability_change, 0.0, 1.0)
                agent.emotional_stability = stability_new
                changes['emotional_stability'] = stability_new - stability_old
                
            elif feedback_type == FeedbackType.LEARNING_ACCELERATION:
                old_value = getattr(agent, 'learning_rate', 0.5)
                change = intensity * adaptation_rate
                new_value = np.clip(old_value + change, 0.0, 1.0)
                agent.learning_rate = new_value
                changes['learning_rate'] = new_value - old_value
                
            elif feedback_type == FeedbackType.INNOVATION_STIMULATION:
                old_value = getattr(agent, 'innovation_capacity', 0.4)
                change = intensity * adaptation_rate
                new_value = np.clip(old_value + change, 0.0, 1.0)
                agent.innovation_capacity = new_value
                changes['innovation_capacity'] = new_value - old_value
                
        except Exception as e:
            logger.warning(f"Failed to apply {feedback_type} to agent: {e}")
        
        return changes
    
    def _calculate_target_values(self, baseline: Dict[str, float], 
                               feedback_type: FeedbackType, intensity: float) -> Dict[str, float]:
        """Calculate target values for feedback"""
        
        targets = baseline.copy()
        adaptation_rate = self.adaptation_rates[feedback_type]
        
        if feedback_type == FeedbackType.MINDFULNESS_BOOST:
            targets['mindfulness_level'] = np.clip(
                baseline['mindfulness_level'] + intensity * adaptation_rate, 0.0, 1.0
            )
        elif feedback_type == FeedbackType.COOPERATION_ENHANCEMENT:
            targets['cooperation_tendency'] = np.clip(
                baseline['cooperation_tendency'] + intensity * adaptation_rate, 0.0, 1.0
            )
            targets['conflict_tendency'] = np.clip(
                baseline['conflict_tendency'] - intensity * adaptation_rate * 0.5, 0.0, 1.0
            )
        elif feedback_type == FeedbackType.ENERGY_OPTIMIZATION:
            targets['energy'] = np.clip(
                baseline['energy'] + intensity * adaptation_rate, 0.0, 1.0
            )
        elif feedback_type == FeedbackType.STRESS_REDUCTION:
            targets['stress_level'] = np.clip(
                baseline['stress_level'] - intensity * adaptation_rate, 0.0, 1.0
            )
            targets['emotional_stability'] = np.clip(
                baseline['emotional_stability'] + intensity * adaptation_rate * 0.3, 0.0, 1.0
            )
        
        return targets
    
    def _update_agent_profile(self, agent, feedback_app: FeedbackApplication):
        """Update agent profile based on feedback response"""
        
        if agent.id not in self.agent_profiles:
            return
        
        profile = self.agent_profiles[agent.id]
        
        # Calculate actual effectiveness based on change magnitude
        target_change = sum(abs(v - feedback_app.baseline_values.get(k, 0)) 
                          for k, v in feedback_app.target_values.items())
        actual_change = sum(abs(v) for v in feedback_app.actual_change.values())
        
        if target_change > 0:
            actual_effectiveness = min(1.0, actual_change / target_change)
        else:
            actual_effectiveness = 0.5
        
        # Update feedback effectiveness
        feedback_app.effectiveness_score = actual_effectiveness
        
        # Update receptivity scores
        feedback_type = feedback_app.feedback_type
        old_receptivity = profile['receptivity_scores'][feedback_type]
        new_receptivity = old_receptivity * 0.8 + actual_effectiveness * 0.2
        profile['receptivity_scores'][feedback_type] = new_receptivity
        
        # Update adaptation speed
        response_speed = sum(abs(v) for v in feedback_app.actual_change.values()) / max(0.1, feedback_app.intensity)
        profile['adaptation_speed'] = profile['adaptation_speed'] * 0.9 + response_speed * 0.1
    
    def _update_adaptation_rates(self, feedback_type: FeedbackType, effectiveness_scores: List[float]):
        """Update adaptation rates based on feedback effectiveness"""
        
        if not effectiveness_scores:
            return
        
        avg_effectiveness = np.mean(effectiveness_scores)
        
        # Adjust adaptation rate based on effectiveness
        current_rate = self.adaptation_rates[feedback_type]
        
        if avg_effectiveness < 0.4:
            # Low effectiveness - increase adaptation rate
            new_rate = min(0.3, current_rate * 1.1)
        elif avg_effectiveness > 0.8:
            # High effectiveness - slightly decrease to avoid overshoot
            new_rate = max(0.01, current_rate * 0.95)
        else:
            # Moderate effectiveness - small adjustment toward optimal
            optimal_rate = 0.1
            new_rate = current_rate * 0.95 + optimal_rate * 0.05
        
        self.adaptation_rates[feedback_type] = new_rate
    
    def _fallback_feedback_application(self, agents: List, feedback_type: str, 
                                     intensity: float) -> Dict[str, Any]:
        """Fallback simple feedback application"""
        
        affected = 0
        
        for agent in agents[:10]:  # Limit to 10 agents
            try:
                if feedback_type == 'mindfulness_boost' and hasattr(agent, 'mindfulness_level'):
                    agent.mindfulness_level = min(1.0, agent.mindfulness_level + intensity * 0.1)
                    affected += 1
                elif feedback_type == 'cooperation_enhancement' and hasattr(agent, 'cooperation_tendency'):
                    agent.cooperation_tendency = min(1.0, agent.cooperation_tendency + intensity * 0.08)
                    affected += 1
            except Exception:
                continue
        
        return {
            'total_agents_affected': affected,
            'feedback_applications': [],
            'overall_effectiveness': 0.5,
            'personalization_used': False
        }
    
    def get_feedback_effectiveness_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of feedback effectiveness by type"""
        
        summary = {}
        
        for feedback_type in FeedbackType:
            type_feedback = [f for f in self.feedback_history 
                           if f.feedback_type == feedback_type]
            
            if type_feedback:
                effectiveness_scores = [f.effectiveness_score for f in type_feedback]
                summary[feedback_type.value] = {
                    'average_effectiveness': float(np.mean(effectiveness_scores)),
                    'total_applications': len(type_feedback),
                    'current_adaptation_rate': self.adaptation_rates[feedback_type],
                    'effectiveness_trend': self._calculate_effectiveness_trend(type_feedback)
                }
        
        return summary
    
    def _calculate_effectiveness_trend(self, feedback_list: List[FeedbackApplication]) -> str:
        """Calculate trend in feedback effectiveness"""
        
        if len(feedback_list) < 6:
            return 'insufficient_data'
        
        recent = feedback_list[-3:]
        older = feedback_list[-6:-3]
        
        recent_avg = np.mean([f.effectiveness_score for f in recent])
        older_avg = np.mean([f.effectiveness_score for f in older])
        
        if recent_avg > older_avg + 0.1:
            return 'improving'
        elif recent_avg < older_avg - 0.1:
            return 'declining'
        else:
            return 'stable'
    
    def get_status(self) -> str:
        """Get feedback interface status"""
        return f"active ({len(self.feedback_history)} applications, {len(self.agent_profiles)} profiles)"

# ===== RITUAL PROTOCOL LAYER =====

class RitualProtocolLayer:
    """Advanced collective ritual coordination system"""
    
    def __init__(self):
        # Ritual tracking
        self.active_rituals = {}
        self.ritual_history = deque(maxlen=1000)
        self.ritual_effectiveness = defaultdict(list)
        
        # Ritual parameters
        self.ritual_requirements = {
            RitualType.SYNCHRONIZED_MEDITATION: {
                'min_participants': 5,
                'optimal_participants': 15,
                'duration_range': (30, 120),  # seconds
                'mindfulness_threshold': 0.4,
                'cooperation_boost': 0.15
            },
            RitualType.WISDOM_CIRCLE: {
                'min_participants': 8,
                'optimal_participants': 20,
                'duration_range': (60, 300),
                'wisdom_threshold': 2.0,
                'insight_amplification': 0.25
            },
            RitualType.CONFLICT_RESOLUTION_CIRCLE: {
                'min_participants': 6,
                'optimal_participants': 12,
                'duration_range': (90, 240),
                'conflict_threshold': 0.3,
                'harmony_restoration': 0.3
            },
            RitualType.ENERGY_SHARING_RITUAL: {
                'min_participants': 10,
                'optimal_participants': 25,
                'duration_range': (45, 150),
                'energy_threshold': 0.3,
                'energy_redistribution': 0.2
            }
        }
        
        # Emergence tracking
        self.emergence_patterns = defaultdict(list)
        self.collective_resonance = deque(maxlen=100)
        
    def assess_ritual_opportunities(self, agents: List, colony_metrics) -> List[RitualType]:
        """Assess which rituals would be beneficial given current state"""
        
        opportunities = []
        
        try:
            # Check each ritual type for viability
            for ritual_type, requirements in self.ritual_requirements.items():
                
                if self._assess_ritual_viability(ritual_type, agents, colony_metrics):
                    opportunities.append(ritual_type)
            
            # Sort by priority based on current needs
            opportunities.sort(key=lambda rt: self._calculate_ritual_priority(rt, colony_metrics), 
                             reverse=True)
            
            return opportunities[:3]  # Return top 3 opportunities
            
        except Exception as e:
            logger.warning(f"Failed to assess ritual opportunities: {e}")
            return []
    
    def _assess_ritual_viability(self, ritual_type: RitualType, agents: List, 
                               colony_metrics) -> bool:
        """Check if a ritual is viable given current conditions"""
        
        requirements = self.ritual_requirements[ritual_type]
        
        # Check minimum participants
        if len(agents) < requirements['min_participants']:
            return False
        
        # Check specific conditions for each ritual type
        if ritual_type == RitualType.SYNCHRONIZED_MEDITATION:
            avg_mindfulness = getattr(colony_metrics, 'collective_mindfulness', 0.5)
            return avg_mindfulness >= requirements['mindfulness_threshold']
            
        elif ritual_type == RitualType.WISDOM_CIRCLE:
            wise_agents = sum(1 for agent in agents 
                            if getattr(agent, 'wisdom_accumulated', 0) >= requirements['wisdom_threshold'])
            return wise_agents >= requirements['min_participants']
            
        elif ritual_type == RitualType.CONFLICT_RESOLUTION_CIRCLE:
            conflict_rate = getattr(colony_metrics, 'conflict_rate', 0.0)
            return conflict_rate >= requirements['conflict_threshold']
            
        elif ritual_type == RitualType.ENERGY_SHARING_RITUAL:
            low_energy_agents = sum(1 for agent in agents 
                                  if getattr(agent, 'energy', 0.5) <= requirements['energy_threshold'])
            return low_energy_agents >= requirements['min_participants'] // 2
        
        return True  # Default to viable for other ritual types
    
    def _calculate_ritual_priority(self, ritual_type: RitualType, colony_metrics) -> float:
        """Calculate priority score for ritual based on current needs"""
        
        priority = 0.0
        
        if ritual_type == RitualType.SYNCHRONIZED_MEDITATION:
            # Higher priority if mindfulness is low or stress is high
            mindfulness = getattr(colony_metrics, 'collective_mindfulness', 0.5)
            priority = (1.0 - mindfulness) * 2.0
            
        elif ritual_type == RitualType.CONFLICT_RESOLUTION_CIRCLE:
            # Higher priority if conflict rate is high
            conflict_rate = getattr(colony_metrics, 'conflict_rate', 0.0)
            priority = conflict_rate * 3.0
            
        elif ritual_type == RitualType.ENERGY_SHARING_RITUAL:
            # Higher priority if average energy is low
            avg_energy = getattr(colony_metrics, 'average_energy', 0.5)
            priority = (1.0 - avg_energy) * 2.5
            
        elif ritual_type == RitualType.WISDOM_CIRCLE:
            # Higher priority if wisdom sharing frequency is low
            wisdom_freq = getattr(colony_metrics, 'wisdom_sharing_frequency', 0.3)
            priority = (1.0 - wisdom_freq) * 1.5
        
        # Boost priority if ritual hasn't been used recently
        recent_usage = self._get_recent_ritual_usage(ritual_type)
        if recent_usage == 0:
            priority *= 1.3
        elif recent_usage > 3:
            priority *= 0.7
        
        return priority
    
    def _get_recent_ritual_usage(self, ritual_type: RitualType) -> int:
        """Get how many times ritual was used recently"""
        
        recent_time = time.time() - 300  # Last 5 minutes
        recent_rituals = [r for r in self.ritual_history 
                         if r.timestamp > recent_time and r.ritual_type == ritual_type]
        return len(recent_rituals)
    
    def execute_ritual(self, ritual_type: RitualType, agents: List, 
                      duration: Optional[float] = None) -> Dict[str, Any]:
        """Execute a collective ritual with participants"""
        
        try:
            requirements = self.ritual_requirements.get(ritual_type)
            if not requirements:
                return {'success': False, 'error': 'Unknown ritual type'}
            
            # Select participants
            participants = self._select_ritual_participants(ritual_type, agents)
            if len(participants) < requirements['min_participants']:
                return {'success': False, 'error': 'Insufficient participants'}
            
            # Determine duration
            if duration is None:
                min_dur, max_dur = requirements['duration_range']
                duration = random.uniform(min_dur, max_dur)
            
            # Execute ritual-specific logic
            ritual_effects = self._perform_ritual_effects(ritual_type, participants, duration)
            
            # Calculate emergence effects
            emergence_effects = self._calculate_emergence_effects(ritual_type, participants, ritual_effects)
            
            # Record ritual execution
            execution_record = RitualExecution(
                ritual_type=ritual_type,
                participants=[str(getattr(p, 'id', id(p))) for p in participants],
                duration=duration,
                effectiveness=ritual_effects.get('overall_effectiveness', 0.5),
                emergence_effects=emergence_effects,
                timestamp=time.time(),
                context={'participant_count': len(participants)}
            )
            
            self.ritual_history.append(execution_record)
            
            # Track active ritual
            ritual_id = f"{ritual_type.value}_{int(time.time())}"
            self.active_rituals[ritual_id] = {
                'type': ritual_type,
                'participants': participants,
                'start_time': time.time(),
                'duration': duration,
                'effects_applied': ritual_effects
            }
            
            # Schedule ritual completion
            self._schedule_ritual_completion(ritual_id, duration)
            
            return {
                'success': True,
                'ritual_id': ritual_id,
                'participants_count': len(participants),
                'duration': duration,
                'immediate_effects': ritual_effects,
                'emergence_effects': emergence_effects,
                'effectiveness': execution_record.effectiveness
            }
            
        except Exception as e:
            logger.error(f"Failed to execute ritual {ritual_type}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _select_ritual_participants(self, ritual_type: RitualType, agents: List) -> List:
        """Select optimal participants for ritual"""
        
        requirements = self.ritual_requirements[ritual_type]
        optimal_count = min(requirements['optimal_participants'], len(agents))
        
        # Score agents based on suitability for ritual type
        scored_agents = []
        
        for agent in agents:
            score = self._calculate_participant_suitability(agent, ritual_type)
            scored_agents.append((agent, score))
        
        # Sort by suitability and select top participants
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        selected = [agent for agent, score in scored_agents[:optimal_count]]
        
        return selected
    
    def _calculate_participant_suitability(self, agent, ritual_type: RitualType) -> float:
        """Calculate how suitable an agent is for a specific ritual"""
        
        score = 0.0
        
        if ritual_type == RitualType.SYNCHRONIZED_MEDITATION:
            score += getattr(agent, 'mindfulness_level', 0.5) * 0.4
            score += (1.0 - getattr(agent, 'stress_level', 0.3)) * 0.3
            score += getattr(agent, 'emotional_stability', 0.5) * 0.3
            
        elif ritual_type == RitualType.WISDOM_CIRCLE:
            score += min(1.0, getattr(agent, 'wisdom_accumulated', 0) / 5.0) * 0.5
            score += getattr(agent, 'mindfulness_level', 0.5) * 0.3
            score += getattr(agent, 'cooperation_tendency', 0.5) * 0.2
            
        elif ritual_type == RitualType.CONFLICT_RESOLUTION_CIRCLE:
            score += (1.0 - getattr(agent, 'conflict_tendency', 0.3)) * 0.4
            score += getattr(agent, 'emotional_stability', 0.5) * 0.3
            score += getattr(agent, 'cooperation_tendency', 0.5) * 0.3
            
        elif ritual_type == RitualType.ENERGY_SHARING_RITUAL:
            score += getattr(agent, 'energy', 0.5) * 0.3  # Mix of high and low energy
            score += getattr(agent, 'cooperation_tendency', 0.5) * 0.4
            score += getattr(agent, 'emotional_stability', 0.5) * 0.3
        
        return score
    
    def _perform_ritual_effects(self, ritual_type: RitualType, participants: List, 
                              duration: float) -> Dict[str, Any]:
        """Apply ritual-specific effects to participants"""
        
        effects = {
            'participants_affected': 0,
            'attribute_changes': defaultdict(list),
            'overall_effectiveness': 0.0
        }
        
        try:
            effectiveness_scores = []
            
            for participant in participants:
                participant_effects = self._apply_ritual_to_participant(
                    ritual_type, participant, duration, len(participants)
                )
                
                if participant_effects:
                    effects['participants_affected'] += 1
                    effectiveness_scores.append(participant_effects.get('effectiveness', 0.5))
                    
                    # Aggregate attribute changes
                    for attr, change in participant_effects.get('changes', {}).items():
                        effects['attribute_changes'][attr].append(change)
            
            # Calculate overall effectiveness
            if effectiveness_scores:
                effects['overall_effectiveness'] = float(np.mean(effectiveness_scores))
            
            return effects
            
        except Exception as e:
            logger.warning(f"Failed to perform ritual effects: {e}")
            return effects
    
    def _apply_ritual_to_participant(self, ritual_type: RitualType, participant, 
                                   duration: float, group_size: int) -> Dict[str, Any]:
        """Apply ritual effects to individual participant"""
        
        changes = {}
        
        # Base effectiveness modified by duration and group size
        base_effectiveness = min(1.0, duration / 60.0)  # Effectiveness increases with duration
        group_bonus = min(0.3, (group_size - 5) * 0.02)  # Bonus for larger groups
        effectiveness = base_effectiveness + group_bonus
        
        try:
            if ritual_type == RitualType.SYNCHRONIZED_MEDITATION:
                # Boost mindfulness and reduce stress
                if hasattr(participant, 'mindfulness_level'):
                    old_value = participant.mindfulness_level
                    boost = effectiveness * 0.15
                    participant.mindfulness_level = min(1.0, old_value + boost)
                    changes['mindfulness_level'] = participant.mindfulness_level - old_value
                
                if hasattr(participant, 'stress_level'):
                    old_value = participant.stress_level
                    reduction = effectiveness * 0.1
                    participant.stress_level = max(0.0, old_value - reduction)
                    changes['stress_level'] = participant.stress_level - old_value
                    
            elif ritual_type == RitualType.WISDOM_CIRCLE:
                # Share and amplify wisdom
                if hasattr(participant, 'wisdom_accumulated'):
                    old_value = participant.wisdom_accumulated
                    # Gain wisdom from sharing
                    wisdom_gain = effectiveness * 0.5
                    participant.wisdom_accumulated = old_value + wisdom_gain
                    changes['wisdom_accumulated'] = wisdom_gain
                
                # Boost learning rate temporarily
                if hasattr(participant, 'learning_rate'):
                    old_value = participant.learning_rate
                    boost = effectiveness * 0.1
                    participant.learning_rate = min(1.0, old_value + boost)
                    changes['learning_rate'] = participant.learning_rate - old_value
                    
            elif ritual_type == RitualType.CONFLICT_RESOLUTION_CIRCLE:
                # Reduce conflict tendency and boost cooperation
                if hasattr(participant, 'conflict_tendency'):
                    old_value = participant.conflict_tendency
                    reduction = effectiveness * 0.15
                    participant.conflict_tendency = max(0.0, old_value - reduction)
                    changes['conflict_tendency'] = participant.conflict_tendency - old_value
                
                if hasattr(participant, 'cooperation_tendency'):
                    old_value = participant.cooperation_tendency
                    boost = effectiveness * 0.1
                    participant.cooperation_tendency = min(1.0, old_value + boost)
                    changes['cooperation_tendency'] = participant.cooperation_tendency - old_value
                    
            elif ritual_type == RitualType.ENERGY_SHARING_RITUAL:
                # Redistribute energy among participants
                avg_energy = np.mean([getattr(p, 'energy', 0.5) for p in [participant]])
                current_energy = getattr(participant, 'energy', 0.5)
                
                if hasattr(participant, 'energy'):
                    # Move toward average energy level
                    energy_change = (avg_energy - current_energy) * effectiveness * 0.2
                    participant.energy = np.clip(current_energy + energy_change, 0.0, 1.0)
                    changes['energy'] = energy_change
            
            return {
                'effectiveness': effectiveness,
                'changes': changes
            }
            
        except Exception as e:
            logger.warning(f"Failed to apply ritual to participant: {e}")
            return {'effectiveness': 0.0, 'changes': {}}
    
    def _calculate_emergence_effects(self, ritual_type: RitualType, participants: List, 
                                   ritual_effects: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emergent effects from collective ritual"""
        
        emergence = {}
        
        try:
            participant_count = len(participants)
            effectiveness = ritual_effects.get('overall_effectiveness', 0.5)
            
            # Base emergence from group synchronization
            synchronization_factor = min(1.0, participant_count / 20.0)
            base_emergence = effectiveness * synchronization_factor
            
            # Ritual-specific emergence effects
            if ritual_type == RitualType.SYNCHRONIZED_MEDITATION:
                emergence['collective_awareness'] = base_emergence * 0.8
                emergence['group_coherence'] = base_emergence * 0.6
                emergence['peaceful_resonance'] = base_emergence * 0.7
                
            elif ritual_type == RitualType.WISDOM_CIRCLE:
                emergence['collective_intelligence'] = base_emergence * 0.9
                emergence['insight_amplification'] = base_emergence * 0.8
                emergence['knowledge_synthesis'] = base_emergence * 0.6
                
            elif ritual_type == RitualType.CONFLICT_RESOLUTION_CIRCLE:
                emergence['harmony_field'] = base_emergence * 0.8
                emergence['empathy_resonance'] = base_emergence * 0.7
                emergence['healing_energy'] = base_emergence * 0.6
                
            elif ritual_type == RitualType.ENERGY_SHARING_RITUAL:
                emergence['vitality_field'] = base_emergence * 0.7
                emergence['resource_optimization'] = base_emergence * 0.6
                emergence['collective_endurance'] = base_emergence * 0.8
            
            # Record emergence patterns
            self.emergence_patterns[ritual_type].append(emergence)
            self.collective_resonance.append(base_emergence)
            
            return emergence
            
        except Exception as e:
            logger.warning(f"Failed to calculate emergence effects: {e}")
            return {}
    
    def _schedule_ritual_completion(self, ritual_id: str, duration: float):
        """Schedule ritual completion (simplified - would use proper scheduler in production)"""
        
        # In a real implementation, this would use a proper task scheduler
        # For now, we'll just mark completion immediately
        if ritual_id in self.active_rituals:
            ritual_info = self.active_rituals[ritual_id]
            
            # Apply any final completion effects
            self._apply_ritual_completion_effects(ritual_info)
            
            # Remove from active rituals
            del self.active_rituals[ritual_id]
    
    def _apply_ritual_completion_effects(self, ritual_info: Dict[str, Any]):
        """Apply effects when ritual completes"""
        
        try:
            ritual_type = ritual_info['type']
            participants = ritual_info['participants']
            
            # Apply completion bonuses
            completion_bonus = 0.05  # Small bonus for completing ritual
            
            for participant in participants:
                if hasattr(participant, 'emotional_stability'):
                    participant.emotional_stability = min(
                        1.0, participant.emotional_stability + completion_bonus
                    )
                    
        except Exception as e:
            logger.warning(f"Failed to apply ritual completion effects: {e}")
    
    def get_ritual_effectiveness_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of ritual effectiveness by type"""
        
        summary = {}
        
        for ritual_type in RitualType:
            type_rituals = [r for r in self.ritual_history 
                          if r.ritual_type == ritual_type]
            
            if type_rituals:
                effectiveness_scores = [r.effectiveness for r in type_rituals]
                recent_rituals = type_rituals[-10:]  # Last 10
                
                summary[ritual_type.value] = {
                    'average_effectiveness': float(np.mean(effectiveness_scores)),
                    'total_executions': len(type_rituals),
                    'recent_effectiveness': float(np.mean([r.effectiveness for r in recent_rituals])),
                    'emergence_strength': self._calculate_emergence_strength(ritual_type)
                }
        
        return summary
    
    def _calculate_emergence_strength(self, ritual_type: RitualType) -> float:
        """Calculate average emergence strength for ritual type"""
        
        emergence_records = self.emergence_patterns.get(ritual_type, [])
        if not emergence_records:
            return 0.0
        
        # Average all emergence effect values
        all_values = []
        for record in emergence_records[-10:]:  # Last 10 records
            all_values.extend(record.values())
        
        return float(np.mean(all_values)) if all_values else 0.0
    
    def get_status(self) -> str:
        """Get ritual layer status"""
        return f"active ({len(self.active_rituals)} active, {len(self.ritual_history)} completed)"

if __name__ == "__main__":
    print("Agent Feedback & Ritual Systems Module - Ready for Integration")