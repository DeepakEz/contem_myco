#!/usr/bin/env python3
"""
MODULE 4: NEURAL ADAPTIVE SYSTEMS
Advanced neural alignment and adaptive threshold regulation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import logging
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

# ===== DATA STRUCTURES =====

@dataclass
class ThresholdConfiguration:
    """Configuration for adaptive thresholds"""
    initial_value: float
    min_value: float
    max_value: float
    adaptation_rate: float
    volatility_sensitivity: float
    
@dataclass
class NeuralAlignmentMetrics:
    """Metrics for neural alignment performance"""
    alignment_score: float = 0.0
    prediction_accuracy: float = 0.0
    adaptation_velocity: float = 0.0
    coherence_index: float = 0.0
    timestamp: float = field(default_factory=time.time)

# ===== THRESHOLD REGULATOR =====

class ThresholdRegulator:
    """Advanced adaptive threshold regulation system"""
    
    def __init__(self):
        # Initialize thresholds with configurations
        self.threshold_configs = {
            'intervention_threshold': ThresholdConfiguration(
                initial_value=0.6,
                min_value=0.1,
                max_value=0.9,
                adaptation_rate=0.01,
                volatility_sensitivity=0.05
            ),
            'crisis_detection_threshold': ThresholdConfiguration(
                initial_value=0.7,
                min_value=0.3,
                max_value=0.95,
                adaptation_rate=0.02,
                volatility_sensitivity=0.03
            ),
            'wisdom_significance_threshold': ThresholdConfiguration(
                initial_value=0.75,
                min_value=0.5,
                max_value=0.9,
                adaptation_rate=0.005,
                volatility_sensitivity=0.02
            )
        }
        
        # Current threshold values
        self.thresholds = {
            name: config.initial_value 
            for name, config in self.threshold_configs.items()
        }
        
        # Tracking systems
        self.threshold_history = defaultdict(deque)  # Track threshold changes
        self.outcome_history = defaultdict(deque)    # Track intervention outcomes
        self.performance_metrics = defaultdict(list) # Performance tracking
        
        # Adaptation neural network
        self.adaptation_network = self._create_adaptation_network()
        
        # Environmental state tracking
        self.environmental_volatility = deque(maxlen=50)
        self.context_stability = deque(maxlen=100)
        
    def _create_adaptation_network(self) -> nn.Module:
        """Create neural network for threshold adaptation"""
        
        class ThresholdAdaptationNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                # Input: outcome history + context features
                self.outcome_encoder = nn.Linear(10, 32)    # Last 10 outcomes
                self.context_encoder = nn.Linear(8, 32)     # Context features
                self.fusion_layer = nn.Linear(64, 64)
                self.adaptation_head = nn.Linear(64, 1)     # Threshold adjustment
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, outcome_features, context_features):
                # Encode inputs
                outcome_emb = F.relu(self.outcome_encoder(outcome_features))
                context_emb = F.relu(self.context_encoder(context_features))
                
                # Fuse features
                fused = torch.cat([outcome_emb, context_emb], dim=-1)
                fused = F.relu(self.fusion_layer(fused))
                fused = self.dropout(fused)
                
                # Output threshold adjustment
                adjustment = torch.tanh(self.adaptation_head(fused))
                return adjustment
        
        return ThresholdAdaptationNetwork()
    
    def get_threshold(self, threshold_name: str) -> float:
        """Get current threshold value"""
        return self.thresholds.get(threshold_name, 0.5)
    
    def record_intervention_outcome(self, threshold_type: str, 
                                  intervention_triggered: bool, 
                                  success: bool,
                                  context: Dict[str, Any] = None):
        """Record outcome of intervention for threshold adaptation"""
        
        outcome_record = {
            'timestamp': time.time(),
            'intervention_triggered': intervention_triggered,
            'success': success,
            'context': context or {},
            'threshold_value': self.thresholds.get(threshold_type, 0.5)
        }
        
        # Store outcome
        self.outcome_history[threshold_type].append(outcome_record)
        
        # Limit history size
        if len(self.outcome_history[threshold_type]) > 200:
            self.outcome_history[threshold_type].popleft()
        
        # Trigger adaptation
        self._adapt_threshold(threshold_type, outcome_record)
        
        # Update performance metrics
        self._update_performance_metrics(threshold_type, outcome_record)
    
    def _adapt_threshold(self, threshold_type: str, outcome_record: Dict[str, Any]):
        """Adapt threshold based on outcome using neural network"""
        
        if threshold_type not in self.threshold_configs:
            return
        
        config = self.threshold_configs[threshold_type]
        current_value = self.thresholds[threshold_type]
        
        try:
            # Prepare features for neural network
            outcome_features = self._extract_outcome_features(threshold_type)
            context_features = self._extract_context_features(outcome_record['context'])
            
            # Get neural network recommendation
            with torch.no_grad():
                adjustment = self.adaptation_network(outcome_features, context_features).item()
            
            # Apply adjustment with constraints
            neural_adjustment = adjustment * config.adaptation_rate
            
            # Apply rule-based adjustments
            rule_based_adjustment = self._calculate_rule_based_adjustment(
                threshold_type, outcome_record, config
            )
            
            # Combine adjustments
            total_adjustment = (neural_adjustment * 0.7 + rule_based_adjustment * 0.3)
            
            # Apply volatility sensitivity
            volatility = self._calculate_environmental_volatility()
            volatility_factor = 1.0 + (volatility * config.volatility_sensitivity)
            total_adjustment *= volatility_factor
            
            # Update threshold with constraints
            new_value = current_value + total_adjustment
            new_value = np.clip(new_value, config.min_value, config.max_value)
            
            # Store old value and update
            old_value = self.thresholds[threshold_type]
            self.thresholds[threshold_type] = new_value
            
            # Record change
            change_record = {
                'timestamp': time.time(),
                'old_value': old_value,
                'new_value': new_value,
                'adjustment': total_adjustment,
                'neural_component': neural_adjustment,
                'rule_component': rule_based_adjustment,
                'trigger_outcome': outcome_record
            }
            
            self.threshold_history[threshold_type].append(change_record)
            
            # Log significant changes
            if abs(total_adjustment) > 0.05:
                logger.info(f"Threshold {threshold_type} adapted: {old_value:.3f} -> {new_value:.3f}")
                
        except Exception as e:
            logger.warning(f"Failed to adapt threshold {threshold_type}: {e}")
            # Fallback to simple rule-based adaptation
            self._simple_threshold_adaptation(threshold_type, outcome_record)
    
    def _extract_outcome_features(self, threshold_type: str) -> torch.Tensor:
        """Extract features from outcome history"""
        
        outcomes = list(self.outcome_history[threshold_type])[-10:]  # Last 10 outcomes
        
        if not outcomes:
            return torch.zeros(10)
        
        # Pad or truncate to exactly 10 outcomes
        features = []
        for i in range(10):
            if i < len(outcomes):
                outcome = outcomes[i]
                # Encode as: [intervention_triggered, success, time_factor]
                time_factor = max(0, 1.0 - (time.time() - outcome['timestamp']) / (24 * 3600))
                feature = [
                    float(outcome['intervention_triggered']),
                    float(outcome['success']),
                    time_factor
                ]
                features.extend(feature)
            else:
                features.extend([0.0, 0.0, 0.0])  # Padding
        
        return torch.tensor(features[:10], dtype=torch.float32)
    
    def _extract_context_features(self, context: Dict[str, Any]) -> torch.Tensor:
        """Extract features from context"""
        
        features = [
            context.get('crisis_level', 0.0),
            context.get('cooperation_rate', 0.5),
            context.get('conflict_rate', 0.0),
            context.get('signal_entropy', 0.0),
            context.get('environmental_stability', 0.5),
            context.get('agent_count', 50) / 100.0,  # Normalize
            context.get('average_energy', 0.5),
            context.get('collective_mindfulness', 0.5)
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _calculate_rule_based_adjustment(self, threshold_type: str, 
                                       outcome_record: Dict[str, Any],
                                       config: ThresholdConfiguration) -> float:
        """Calculate rule-based threshold adjustment"""
        
        intervention_triggered = outcome_record['intervention_triggered']
        success = outcome_record['success']
        
        # Basic adaptation rules
        if intervention_triggered and success:
            # Good intervention - slightly lower threshold to catch more
            return -config.adaptation_rate * 0.5
        elif intervention_triggered and not success:
            # False positive - raise threshold to be more selective
            return config.adaptation_rate * 1.5
        elif not intervention_triggered and not success:
            # Potential missed opportunity - lower threshold
            return -config.adaptation_rate * 1.0
        else:
            # No intervention needed and none triggered - small adjustment
            return config.adaptation_rate * 0.1
    
    def _calculate_environmental_volatility(self) -> float:
        """Calculate current environmental volatility"""
        
        if len(self.environmental_volatility) < 2:
            return 0.0
        
        # Calculate variance in recent environmental conditions
        volatility_values = list(self.environmental_volatility)
        return float(np.var(volatility_values))
    
    def _simple_threshold_adaptation(self, threshold_type: str, outcome_record: Dict[str, Any]):
        """Fallback simple threshold adaptation"""
        
        if threshold_type not in self.threshold_configs:
            return
        
        config = self.threshold_configs[threshold_type]
        current_value = self.thresholds[threshold_type]
        
        intervention_triggered = outcome_record['intervention_triggered']
        success = outcome_record['success']
        
        # Simple adaptation logic
        if intervention_triggered and not success:
            # False positive - increase threshold
            adjustment = config.adaptation_rate
        elif not intervention_triggered and not success:
            # Missed opportunity - decrease threshold
            adjustment = -config.adaptation_rate
        else:
            # Correct decision - small adjustment toward optimum
            adjustment = config.adaptation_rate * 0.1 * (0.7 - current_value)
        
        # Apply adjustment
        new_value = np.clip(
            current_value + adjustment,
            config.min_value,
            config.max_value
        )
        
        self.thresholds[threshold_type] = new_value
    
    def _update_performance_metrics(self, threshold_type: str, outcome_record: Dict[str, Any]):
        """Update performance metrics for threshold"""
        
        # Calculate accuracy over recent outcomes
        recent_outcomes = list(self.outcome_history[threshold_type])[-20:]
        
        if len(recent_outcomes) >= 5:
            correct_decisions = sum(
                1 for outcome in recent_outcomes
                if (outcome['intervention_triggered'] and outcome['success']) or
                   (not outcome['intervention_triggered'] and outcome['success'])
            )
            
            accuracy = correct_decisions / len(recent_outcomes)
            
            metric = {
                'timestamp': time.time(),
                'accuracy': accuracy,
                'total_outcomes': len(recent_outcomes),
                'threshold_value': self.thresholds[threshold_type]
            }
            
            self.performance_metrics[threshold_type].append(metric)
            
            # Limit metrics history
            if len(self.performance_metrics[threshold_type]) > 100:
                self.performance_metrics[threshold_type].pop(0)
    
    def get_threshold_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive threshold analysis"""
        
        analysis = {}
        
        for threshold_name in self.thresholds:
            current_value = self.thresholds[threshold_name]
            config = self.threshold_configs[threshold_name]
            
            # Recent performance
            recent_metrics = self.performance_metrics[threshold_name][-10:]
            avg_accuracy = np.mean([m['accuracy'] for m in recent_metrics]) if recent_metrics else 0.5
            
            # Adaptation activity
            recent_changes = list(self.threshold_history[threshold_name])[-10:]
            adaptation_rate = len(recent_changes) / max(1, len(self.outcome_history[threshold_name]))
            
            # Volatility
            recent_values = [change['new_value'] for change in recent_changes]
            volatility = np.std(recent_values) if len(recent_values) > 1 else 0.0
            
            analysis[threshold_name] = {
                'current_value': current_value,
                'config': config,
                'performance': {
                    'accuracy': avg_accuracy,
                    'adaptation_rate': adaptation_rate,
                    'volatility': volatility
                },
                'recommendation': self._get_threshold_recommendation(
                    threshold_name, avg_accuracy, adaptation_rate, volatility
                )
            }
        
        return analysis
    
    def _get_threshold_recommendation(self, threshold_name: str, accuracy: float,
                                    adaptation_rate: float, volatility: float) -> str:
        """Get recommendation for threshold management"""
        
        if accuracy < 0.6:
            return "MAJOR_ADJUSTMENT_NEEDED"
        elif volatility > 0.1:
            return "REDUCE_ADAPTATION_SENSITIVITY"
        elif adaptation_rate > 0.5:
            return "DECREASE_ADAPTATION_RATE"
        elif accuracy > 0.8 and volatility < 0.05:
            return "OPTIMAL_PERFORMANCE"
        else:
            return "MINOR_TUNING_SUGGESTED"
    
    def get_status(self) -> str:
        """Get regulator status"""
        active_thresholds = len(self.thresholds)
        total_adaptations = sum(len(history) for history in self.threshold_history.values())
        return f"active ({active_thresholds} thresholds, {total_adaptations} adaptations)"

# ===== NEURAL ALIGNMENT SYSTEM =====

class NeuralAlignment:
    """Advanced neural alignment for contemplative decision making"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        
        # Neural networks
        self.insight_encoder = self._create_insight_encoder()
        self.context_encoder = self._create_context_encoder()
        self.alignment_network = self._create_alignment_network()
        
        # Alignment tracking
        self.alignment_history = deque(maxlen=1000)
        self.dharma_alignment_scores = deque(maxlen=500)
        
        # Performance metrics
        self.metrics_history = deque(maxlen=200)
        
    def _create_insight_encoder(self) -> nn.Module:
        """Create neural network for encoding wisdom insights"""
        
        class InsightEncoder(nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                # Simple text encoding (in practice would use pre-trained embeddings)
                self.text_projection = nn.Linear(50, embedding_dim)  # Simplified text features
                self.dharma_head = nn.Linear(embedding_dim, 1)
                self.significance_head = nn.Linear(embedding_dim, 1)
                
            def forward(self, text_features):
                embedding = F.relu(self.text_projection(text_features))
                dharma_score = torch.sigmoid(self.dharma_head(embedding))
                significance = torch.sigmoid(self.significance_head(embedding))
                return embedding, dharma_score, significance
        
        return InsightEncoder(self.embedding_dim)
    
    def _create_context_encoder(self) -> nn.Module:
        """Create neural network for encoding contextual information"""
        
        class ContextEncoder(nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.colony_encoder = nn.Linear(10, embedding_dim // 2)
                self.environment_encoder = nn.Linear(8, embedding_dim // 2)
                self.fusion_layer = nn.Linear(embedding_dim, embedding_dim)
                
            def forward(self, colony_features, env_features):
                colony_emb = F.relu(self.colony_encoder(colony_features))
                env_emb = F.relu(self.environment_encoder(env_features))
                
                combined = torch.cat([colony_emb, env_emb], dim=-1)
                context_embedding = F.relu(self.fusion_layer(combined))
                return context_embedding
        
        return ContextEncoder(self.embedding_dim)
    
    def _create_alignment_network(self) -> nn.Module:
        """Create neural network for computing alignment scores"""
        
        class AlignmentNetwork(nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.attention = nn.MultiheadAttention(embedding_dim, 4)
                self.alignment_head = nn.Linear(embedding_dim, 1)
                self.coherence_head = nn.Linear(embedding_dim, 1)
                
            def forward(self, insight_emb, context_emb):
                # Compute attention between insight and context
                attended_insight, attention_weights = self.attention(
                    insight_emb.unsqueeze(0),
                    context_emb.unsqueeze(0),
                    context_emb.unsqueeze(0)
                )
                
                # Compute alignment and coherence scores
                alignment_score = torch.sigmoid(self.alignment_head(attended_insight.squeeze(0)))
                coherence_score = torch.sigmoid(self.coherence_head(attended_insight.squeeze(0)))
                
                return alignment_score, coherence_score, attention_weights
        
        return AlignmentNetwork(self.embedding_dim)
    
    def encode_wisdom_insight(self, insight_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Encode wisdom insight with neural alignment"""
        
        try:
            # Extract text features (simplified - would use proper NLP in practice)
            text_features = self._extract_text_features(insight_text)
            
            # Encode insight
            with torch.no_grad():
                insight_embedding, dharma_score, significance = self.insight_encoder(text_features)
            
            # Extract context features
            colony_features, env_features = self._extract_context_features(context)
            
            # Encode context
            with torch.no_grad():
                context_embedding = self.context_encoder(colony_features, env_features)
            
            # Compute alignment
            with torch.no_grad():
                alignment_score, coherence_score, attention_weights = self.alignment_network(
                    insight_embedding, context_embedding
                )
            
            # Create result
            result = {
                'insight_embedding': insight_embedding,
                'context_embedding': context_embedding,
                'dharma_alignment': float(dharma_score.item()),
                'significance_score': float(significance.item()),
                'neural_alignment': float(alignment_score.item()),
                'coherence_score': float(coherence_score.item()),
                'attention_weights': attention_weights.squeeze().tolist() if attention_weights.dim() > 1 else [float(attention_weights.item())]
            }
            
            # Record alignment
            self._record_alignment(result, context)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to encode wisdom insight: {e}")
            return self._create_fallback_encoding(insight_text, context)
    
    def _extract_text_features(self, text: str) -> torch.Tensor:
        """Extract features from text (simplified implementation)"""
        
        # Simple feature extraction based on keywords and length
        features = []
        
        text_lower = text.lower()
        
        # Dharma-related keywords
        dharma_keywords = ['wisdom', 'compassion', 'mindfulness', 'harmony', 'balance', 'peace']
        dharma_count = sum(1 for keyword in dharma_keywords if keyword in text_lower)
        features.extend([dharma_count / len(dharma_keywords)] * 10)
        
        # Emotional tone
        positive_words = ['good', 'better', 'heal', 'grow', 'help', 'support']
        negative_words = ['bad', 'worse', 'harm', 'conflict', 'stress', 'fear']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        emotional_balance = (positive_count - negative_count) / max(1, positive_count + negative_count)
        features.extend([emotional_balance] * 10)
        
        # Text complexity
        word_count = len(text.split())
        complexity = min(1.0, word_count / 20.0)  # Normalize to 0-1
        features.extend([complexity] * 10)
        
        # Action orientation
        action_words = ['do', 'act', 'create', 'build', 'make', 'implement']
        action_count = sum(1 for word in action_words if word in text_lower)
        action_score = min(1.0, action_count / 3.0)
        features.extend([action_score] * 10)
        
        # Collective focus
        collective_words = ['we', 'us', 'together', 'collective', 'community', 'shared']
        collective_count = sum(1 for word in collective_words if word in text_lower)
        collective_score = min(1.0, collective_count / 3.0)
        features.extend([collective_score] * 10)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_context_features(self, context: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from context"""
        
        # Colony features
        colony_features = [
            context.get('crisis_level', 0.0),
            context.get('cooperation_rate', 0.5),
            context.get('conflict_rate', 0.0),
            context.get('collective_mindfulness', 0.5),
            context.get('average_energy', 0.5),
            context.get('average_health', 0.5),
            context.get('wisdom_sharing_frequency', 0.3),
            context.get('signal_entropy', 0.0),
            context.get('emotional_gradients', 0.0),
            context.get('agent_count', 50) / 100.0  # Normalize
        ]
        
        # Environmental features
        env_features = [
            context.get('temperature', 25.0) / 50.0,  # Normalize
            context.get('resource_abundance', 0.7),
            context.get('hazard_level', 0.2),
            context.get('environmental_stability', 0.5),
            context.get('season_factor', 0.5),  # Would be computed from season
            context.get('weather_volatility', 0.1),
            context.get('resource_regeneration_rate', 0.1),
            context.get('ecosystem_health', 0.7)
        ]
        
        return (
            torch.tensor(colony_features, dtype=torch.float32),
            torch.tensor(env_features, dtype=torch.float32)
        )
    
    def _record_alignment(self, result: Dict[str, Any], context: Dict[str, Any]):
        """Record alignment for tracking and learning"""
        
        alignment_record = {
            'timestamp': time.time(),
            'dharma_alignment': result['dharma_alignment'],
            'neural_alignment': result['neural_alignment'],
            'coherence_score': result['coherence_score'],
            'context_type': self._classify_context(context),
            'significance': result['significance_score']
        }
        
        self.alignment_history.append(alignment_record)
        self.dharma_alignment_scores.append(result['dharma_alignment'])
        
        # Update metrics
        self._update_alignment_metrics(alignment_record)
    
    def _classify_context(self, context: Dict[str, Any]) -> str:
        """Classify context type for analysis"""
        
        crisis_level = context.get('crisis_level', 0.0)
        cooperation_rate = context.get('cooperation_rate', 0.5)
        
        if crisis_level > 0.7:
            return 'crisis'
        elif cooperation_rate > 0.8:
            return 'harmonious'
        elif cooperation_rate < 0.3:
            return 'conflicted'
        else:
            return 'balanced'
    
    def _update_alignment_metrics(self, alignment_record: Dict[str, Any]):
        """Update performance metrics"""
        
        recent_alignments = list(self.alignment_history)[-20:]
        
        if len(recent_alignments) >= 10:
            metrics = NeuralAlignmentMetrics(
                alignment_score=np.mean([r['neural_alignment'] for r in recent_alignments]),
                prediction_accuracy=0.0,  # Would be computed from actual outcomes
                adaptation_velocity=self._calculate_adaptation_velocity(recent_alignments),
                coherence_index=np.mean([r['coherence_score'] for r in recent_alignments])
            )
            
            self.metrics_history.append(metrics)
    
    def _calculate_adaptation_velocity(self, recent_alignments: List[Dict[str, Any]]) -> float:
        """Calculate how quickly the system is adapting"""
        
        if len(recent_alignments) < 5:
            return 0.0
        
        # Calculate rate of change in alignment scores
        alignment_scores = [r['neural_alignment'] for r in recent_alignments]
        velocities = []
        
        for i in range(1, len(alignment_scores)):
            velocity = abs(alignment_scores[i] - alignment_scores[i-1])
            velocities.append(velocity)
        
        return float(np.mean(velocities))
    
    def _create_fallback_encoding(self, insight_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback encoding when neural processing fails"""
        
        # Simple rule-based encoding
        dharma_keywords = ['wisdom', 'compassion', 'mindfulness', 'harmony']
        dharma_score = sum(1 for keyword in dharma_keywords if keyword.lower() in insight_text.lower())
        dharma_alignment = min(1.0, dharma_score / 2.0)
        
        return {
            'dharma_alignment': dharma_alignment,
            'significance_score': 0.5,
            'neural_alignment': 0.5,
            'coherence_score': 0.5,
            'fallback_used': True
        }
    
    def get_alignment_summary(self) -> Dict[str, Any]:
        """Get summary of alignment performance"""
        
        if not self.dharma_alignment_scores:
            return {'status': 'no_data'}
        
        recent_scores = list(self.dharma_alignment_scores)[-50:]
        
        return {
            'average_dharma_alignment': float(np.mean(recent_scores)),
            'alignment_trend': self._calculate_alignment_trend(recent_scores),
            'coherence_stability': self._calculate_coherence_stability(),
            'total_insights_processed': len(self.alignment_history),
            'context_distribution': self._get_context_distribution()
        }
    
    def _calculate_alignment_trend(self, scores: List[float]) -> str:
        """Calculate trend in alignment scores"""
        
        if len(scores) < 10:
            return 'insufficient_data'
        
        # Simple linear trend
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)
        
        if second_avg > first_avg + 0.05:
            return 'improving'
        elif second_avg < first_avg - 0.05:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_coherence_stability(self) -> float:
        """Calculate stability of coherence scores"""
        
        recent_records = list(self.alignment_history)[-30:]
        if len(recent_records) < 5:
            return 0.0
        
        coherence_scores = [r['coherence_score'] for r in recent_records]
        return float(1.0 - np.std(coherence_scores))  # Higher stability = lower std dev
    
    def _get_context_distribution(self) -> Dict[str, int]:
        """Get distribution of context types processed"""
        
        recent_records = list(self.alignment_history)[-100:]
        distribution = defaultdict(int)
        
        for record in recent_records:
            context_type = record.get('context_type', 'unknown')
            distribution[context_type] += 1
        
        return dict(distribution)
    
    def get_status(self) -> str:
        """Get neural alignment status"""
        processed_insights = len(self.alignment_history)
        avg_alignment = np.mean(self.dharma_alignment_scores) if self.dharma_alignment_scores else 0.0
        return f"active ({processed_insights} insights, {avg_alignment:.3f} avg alignment)"

if __name__ == "__main__":
    print("Neural Adaptive Systems Module - Ready for Integration")