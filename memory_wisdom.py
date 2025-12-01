#!/usr/bin/env python3
"""
MODULE 2: MEMORY & WISDOM SYSTEMS
Advanced memory attention, wisdom archive with significance scoring,
and insight evolution tracking
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
class WisdomInsightEmbedding:
    """Embedded representation of wisdom insights"""
    insight_text: str
    embedding_vector: torch.Tensor
    dharma_alignment: float
    emergence_context: Dict[str, float]
    timestamp: float
    source_trace: Optional[Dict[str, Any]] = None
    significance_score: float = 0.0

@dataclass
class WisdomSignificanceMetrics:
    """Enhanced significance scoring for wisdom insights"""
    novelty_score: float = 0.0
    persistence_score: float = 0.0
    feedback_effect_score: float = 0.0
    cross_domain_relevance: float = 0.0
    emergence_complexity: float = 0.0
    
    def calculate_overall_significance(self) -> float:
        """Calculate weighted overall significance score"""
        weights = {
            'novelty': 0.25,
            'persistence': 0.20,
            'feedback_effect': 0.30,
            'cross_domain': 0.15,
            'emergence_complexity': 0.10
        }
        
        return (
            self.novelty_score * weights['novelty'] +
            self.persistence_score * weights['persistence'] +
            self.feedback_effect_score * weights['feedback_effect'] +
            self.cross_domain_relevance * weights['cross_domain'] +
            self.emergence_complexity * weights['emergence_complexity']
        )

# ===== WISDOM ARCHIVE WITH SIGNIFICANCE SCORING =====

class WisdomArchive:
    """Enhanced wisdom archive with significance scoring and decay detection"""
    
    def __init__(self, max_insights: int = 10000):
        self.max_insights = max_insights
        self.insights = {}  # insight_id -> WisdomInsightEmbedding
        self.insight_metadata = {}  # insight_id -> metadata
        self.significance_tracker = {}  # insight_id -> WisdomSignificanceMetrics
        self.reuse_tracking = defaultdict(list)  # insight_id -> reuse events
        self.historical_impact_scores = {}  # insight_id -> impact over time
        
        # Decay and evolution tracking
        self.decay_scores = {}  # insight_id -> current decay score
        self.novelty_detector = NoveltyDetector()
        
        # Categorization and retrieval
        self.insight_tags = defaultdict(set)  # tag -> set of insight_ids
        self.relevance_index = {}  # context_hash -> relevant insight_ids
    
    def archive_insight_with_significance(self, insight_obj: WisdomInsightEmbedding, 
                                        context: Dict[str, Any], tags: List[str] = None,
                                        confidence_score: float = 1.0) -> Optional[str]:
        """Archive insight with confidence threshold and significance scoring"""
        
        # Check confidence threshold
        if confidence_score < 0.7:
            logger.warning(f"Insight confidence {confidence_score:.3f} below archival threshold")
            return None
        
        try:
            # Calculate significance metrics
            significance = self._calculate_significance_metrics(insight_obj, context)
            insight_obj.significance_score = significance.calculate_overall_significance()
            
            # Detect novelty
            novelty_score = self.novelty_detector.calculate_novelty(
                insight_obj.embedding_vector, list(self.insights.values())
            )
            significance.novelty_score = novelty_score
            
            # Archive with enhanced metadata
            insight_id = self.store_insight(insight_obj, context, tags)
            
            if insight_id:
                self.significance_tracker[insight_id] = significance
                
                # Log significant insights
                overall_significance = significance.calculate_overall_significance()
                if overall_significance > 0.8:
                    logger.info(f"High significance insight archived: {insight_id} (score: {overall_significance:.3f})")
            
            return insight_id
            
        except Exception as e:
            logger.error(f"Failed to archive insight with significance: {e}")
            return None
    
    def store_insight(self, insight: WisdomInsightEmbedding, context: Dict[str, Any], 
                     tags: List[str] = None) -> str:
        """Store insight with full context tracking"""
        
        insight_id = f"insight_{len(self.insights)}_{int(time.time())}"
        
        # Store insight
        self.insights[insight_id] = insight
        
        # Create comprehensive metadata
        self.insight_metadata[insight_id] = {
            'creation_timestamp': time.time(),
            'creation_context': context,
            'tags': set(tags or []),
            'usage_count': 0,
            'last_accessed': time.time(),
            'success_rate': 0.0,
            'impact_measurements': [],
            'relevance_score': 1.0
        }
        
        # Initialize tracking
        self.historical_impact_scores[insight_id] = []
        self.decay_scores[insight_id] = 0.0
        
        # Add tags to index
        if tags:
            for tag in tags:
                self.insight_tags[tag].add(insight_id)
        
        # Add to relevance index
        context_hash = self._hash_context(context)
        if context_hash not in self.relevance_index:
            self.relevance_index[context_hash] = set()
        self.relevance_index[context_hash].add(insight_id)
        
        # Automatic tagging based on content
        auto_tags = self._generate_auto_tags(insight, context)
        for tag in auto_tags:
            self.insight_metadata[insight_id]['tags'].add(tag)
            self.insight_tags[tag].add(insight_id)
        
        logger.info(f"Archived insight {insight_id} with tags: {self.insight_metadata[insight_id]['tags']}")
        
        # Manage storage limits
        if len(self.insights) > self.max_insights:
            self._prune_old_insights()
        
        return insight_id
    
    def record_insight_reuse(self, insight_id: str, context: Dict[str, Any], success_score: float):
        """Record when and how an insight was reused"""
        
        if insight_id not in self.insights:
            return
        
        reuse_event = {
            'timestamp': time.time(),
            'context': context,
            'success_score': success_score,
            'context_similarity': self._calculate_context_similarity(insight_id, context)
        }
        
        self.reuse_tracking[insight_id].append(reuse_event)
        self.historical_impact_scores[insight_id].append(success_score)
        
        # Update metadata
        metadata = self.insight_metadata[insight_id]
        metadata['usage_count'] += 1
        metadata['last_accessed'] = time.time()
        
        # Update success rate
        metadata['impact_measurements'].append(reuse_event)
        if metadata['impact_measurements']:
            metadata['success_rate'] = np.mean([m['success_score'] for m in metadata['impact_measurements']])
        
        # Update decay score
        self._update_decay_score(insight_id)
    
    def detect_insight_decay(self, insight_id: str) -> Dict[str, Any]:
        """Detect if an insight is becoming outdated"""
        
        if insight_id not in self.insights:
            return {'error': 'Insight not found'}
        
        metadata = self.insight_metadata[insight_id]
        
        # Time-based decay
        age_days = (time.time() - metadata['creation_timestamp']) / (24 * 3600)
        time_decay = min(1.0, age_days / 365)  # Decay over a year
        
        # Usage-based relevance
        recent_usage = len([r for r in self.reuse_tracking[insight_id] 
                           if time.time() - r['timestamp'] < 30 * 24 * 3600])  # Last 30 days
        usage_relevance = 1.0 / (1.0 + np.exp(-recent_usage + 2))  # Sigmoid
        
        # Success trend
        recent_successes = self.historical_impact_scores[insight_id][-10:]  # Last 10 uses
        success_trend = np.mean(recent_successes) if recent_successes else 0.5
        
        # Combined decay score
        overall_decay = (
            time_decay * 0.4 +
            (1.0 - usage_relevance) * 0.3 +
            (1.0 - success_trend) * 0.3
        )
        
        return {
            'overall_decay_score': overall_decay,
            'time_decay': time_decay,
            'usage_relevance': usage_relevance,
            'success_trend': success_trend,
            'recommendation': self._get_decay_recommendation(overall_decay)
        }
    
    def get_insights_by_significance(self, min_significance: float = 0.7, 
                                   max_results: int = 10) -> List[Tuple[str, WisdomInsightEmbedding, float]]:
        """Get insights ranked by significance score"""
        
        significant_insights = []
        
        for insight_id, significance in self.significance_tracker.items():
            if insight_id in self.insights:
                score = significance.calculate_overall_significance()
                if score >= min_significance:
                    significant_insights.append((insight_id, self.insights[insight_id], score))
        
        # Sort by significance score
        significant_insights.sort(key=lambda x: x[2], reverse=True)
        
        return significant_insights[:max_results]
    
    def _calculate_significance_metrics(self, insight_obj: WisdomInsightEmbedding, 
                                      context: Dict[str, Any]) -> WisdomSignificanceMetrics:
        """Calculate comprehensive significance metrics"""
        
        significance = WisdomSignificanceMetrics()
        
        # Persistence score based on context stability
        context_volatility = self._calculate_context_volatility(context)
        significance.persistence_score = 1.0 - context_volatility
        
        # Cross-domain relevance based on context breadth
        context_breadth = len([k for k, v in context.items() if isinstance(v, (int, float)) and v > 0])
        significance.cross_domain_relevance = min(1.0, context_breadth / 10.0)
        
        # Emergence complexity based on context interactions
        significance.emergence_complexity = self._calculate_emergence_complexity(context)
        
        # Feedback effect score (will be updated as feedback is applied)
        significance.feedback_effect_score = 0.5  # Default, updated later
        
        return significance
    
    def _calculate_context_volatility(self, context: Dict[str, Any]) -> float:
        """Calculate how volatile/changing the context is"""
        
        volatile_indicators = [
            context.get('crisis_level', 0),
            context.get('conflict_rate', 0),
            abs(context.get('cooperation_rate', 0.5) - 0.5) * 2,
            context.get('signal_entropy', 0)
        ]
        
        return np.mean([v for v in volatile_indicators if isinstance(v, (int, float))])
    
    def _calculate_emergence_complexity(self, context: Dict[str, Any]) -> float:
        """Calculate complexity of conditions that led to insight emergence"""
        
        # Count non-zero context factors
        active_factors = sum(1 for v in context.values() 
                           if isinstance(v, (int, float)) and abs(v) > 0.1)
        
        # Calculate interaction complexity
        numeric_values = [v for v in context.values() if isinstance(v, (int, float))]
        if len(numeric_values) > 1:
            variance = np.var(numeric_values)
            mean_val = np.mean(numeric_values)
            complexity = (variance / (mean_val + 0.1)) * (active_factors / 20.0)
        else:
            complexity = active_factors / 20.0
        
        return min(1.0, complexity)
    
    def _generate_auto_tags(self, insight_obj: WisdomInsightEmbedding, context: Dict[str, Any]) -> List[str]:
        """Generate automatic tags based on content and context"""
        
        auto_tags = []
        text = insight_obj.insight_text.lower()
        
        # Content-based tags
        if 'cooperation' in text or 'together' in text:
            auto_tags.append('cooperation')
        if 'wisdom' in text or 'understanding' in text:
            auto_tags.append('wisdom')
        if 'balance' in text or 'harmony' in text:
            auto_tags.append('balance')
        if 'meditation' in text or 'mindful' in text:
            auto_tags.append('contemplative')
        
        # Context-based tags
        if context.get('crisis_level', 0) > 0.7:
            auto_tags.append('crisis_response')
        if context.get('cooperation_rate', 1) < 0.5:
            auto_tags.append('social_healing')
        
        # Dharma alignment tags
        if insight_obj.dharma_alignment > 0.8:
            auto_tags.append('high_dharma')
        elif insight_obj.dharma_alignment < 0.4:
            auto_tags.append('questionable')
        
        return auto_tags
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create hash of context for relevance indexing"""
        
        # Use key context elements for hashing
        key_elements = []
        for key in ['crisis_level', 'cooperation_rate', 'conflict_rate', 'step']:
            if key in context:
                key_elements.append(f"{key}:{context[key]}")
        
        return "_".join(key_elements)
    
    def _calculate_context_similarity(self, insight_id: str, current_context: Dict[str, Any]) -> float:
        """Calculate similarity between original and current context"""
        
        if insight_id not in self.insight_metadata:
            return 0.0
        
        original_context = self.insight_metadata[insight_id]['creation_context']
        
        # Simple similarity based on numerical context features
        similarity_scores = []
        
        common_keys = set(original_context.keys()) & set(current_context.keys())
        for key in common_keys:
            orig_val = original_context.get(key, 0)
            curr_val = current_context.get(key, 0)
            
            if isinstance(orig_val, (int, float)) and isinstance(curr_val, (int, float)):
                # Normalized difference
                max_val = max(abs(orig_val), abs(curr_val), 1.0)
                similarity = 1.0 - abs(orig_val - curr_val) / max_val
                similarity_scores.append(similarity)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def _update_decay_score(self, insight_id: str):
        """Update decay score based on usage patterns and age"""
        
        metadata = self.insight_metadata[insight_id]
        
        # Age factor
        age_days = (time.time() - metadata['creation_timestamp']) / (24 * 3600)
        age_decay = min(1.0, age_days / 365)  # Decay over a year
        
        # Usage recency factor
        time_since_last_use = (time.time() - metadata['last_accessed']) / (24 * 3600)
        recency_decay = min(1.0, time_since_last_use / 30)  # Decay if not used for 30 days
        
        # Success trend factor
        recent_impacts = self.historical_impact_scores[insight_id][-5:]
        if recent_impacts:
            trend_factor = 1.0 - np.mean(recent_impacts)
        else:
            trend_factor = 0.5
        
        # Combined decay score
        decay_score = (age_decay * 0.3 + recency_decay * 0.4 + trend_factor * 0.3)
        self.decay_scores[insight_id] = decay_score
        
        # Update relevance score
        metadata['relevance_score'] = max(0.1, 1.0 - decay_score)
    
    def _get_decay_recommendation(self, decay_score: float) -> str:
        """Get recommendation based on decay analysis"""
        
        if decay_score > 0.8:
            return "ARCHIVE_INSIGHT"
        elif decay_score > 0.6:
            return "REVISE_INSIGHT"
        elif decay_score > 0.4:
            return "REFRESH_CONTEXT"
        else:
            return "INSIGHT_HEALTHY"
    
    def _prune_old_insights(self):
        """Remove insights with high decay scores"""
        
        # Calculate value scores for all insights
        insight_values = []
        
        for insight_id in self.insights:
            metadata = self.insight_metadata[insight_id]
            decay_score = self.decay_scores[insight_id]
            
            # Value = usage * success_rate * (1 - decay)
            value_score = (
                metadata['usage_count'] * 
                metadata['success_rate'] * 
                (1.0 - decay_score)
            )
            
            insight_values.append((insight_id, value_score))
        
        # Sort by value and remove lowest 10%
        insight_values.sort(key=lambda x: x[1])
        remove_count = len(self.insights) // 10
        
        for insight_id, _ in insight_values[:remove_count]:
            self._remove_insight(insight_id)
    
    def _remove_insight(self, insight_id: str):
        """Remove insight and all associated data"""
        
        # Remove from all data structures
        if insight_id in self.insights:
            del self.insights[insight_id]
        if insight_id in self.insight_metadata:
            del self.insight_metadata[insight_id]
        if insight_id in self.historical_impact_scores:
            del self.historical_impact_scores[insight_id]
        if insight_id in self.decay_scores:
            del self.decay_scores[insight_id]
        if insight_id in self.reuse_tracking:
            del self.reuse_tracking[insight_id]
        if insight_id in self.significance_tracker:
            del self.significance_tracker[insight_id]
        
        # Remove from tag indices
        for tag_set in self.insight_tags.values():
            tag_set.discard(insight_id)
        
        # Remove from relevance index
        for insight_set in self.relevance_index.values():
            insight_set.discard(insight_id)
    
    def get_status(self) -> str:
        """Get archive status"""
        return f"active ({len(self.insights)} insights, {len(self.significance_tracker)} tracked)"

# ===== NOVELTY DETECTOR =====

class NoveltyDetector:
    """Detect novelty of new insights compared to existing ones"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    def calculate_novelty(self, new_embedding: torch.Tensor, 
                         existing_insights: List[WisdomInsightEmbedding]) -> float:
        """Calculate novelty score (0 = very similar to existing, 1 = very novel)"""
        
        if not existing_insights:
            return 1.0  # First insight is maximally novel
        
        # Calculate similarities to all existing insights
        similarities = []
        
        for existing_insight in existing_insights:
            try:
                similarity = F.cosine_similarity(
                    new_embedding.unsqueeze(0),
                    existing_insight.embedding_vector.unsqueeze(0)
                ).item()
                similarities.append(similarity)
            except Exception as e:
                logger.warning(f"Failed to calculate similarity: {e}")
                continue
        
        if not similarities:
            return 1.0
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity
        
        # Apply threshold for binary novelty detection
        if max_similarity > self.similarity_threshold:
            logger.info(f"Low novelty detected: max similarity {max_similarity:.3f}")
        
        return max(0.0, novelty)

# ===== MEMORY ATTENTION MECHANISM =====

class MemoryAttentionMechanism:
    """Advanced memory attention system for weighting historical decisions"""
    
    def __init__(self, memory_capacity: int = 1000):
        self.memory_capacity = memory_capacity
        self.intervention_memories = deque(maxlen=memory_capacity)
        self.impact_tracking = defaultdict(list)  # Track delayed impacts
        self.memory_embeddings = {}  # Store embedded representations
        
        # Attention weights
        self.attention_weights = {
            'recent_weight': 0.4,      # Last 10 decisions
            'medium_term_weight': 0.3, # Last 50 decisions  
            'long_term_weight': 0.2,   # Older decisions
            'crisis_memory_weight': 0.1 # Crisis-specific memories
        }
        
        # Attention neural network
        self.attention_network = self._create_attention_network()
        
    def _create_attention_network(self) -> nn.Module:
        """Create neural network for computing attention weights"""
        
        class AttentionNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                # Input: current context + memory features
                self.context_encoder = nn.Linear(20, 64)  # Current state
                self.memory_encoder = nn.Linear(15, 64)   # Memory features
                self.attention_head = nn.MultiheadAttention(64, 4)
                self.output_layer = nn.Linear(64, 1)
                
            def forward(self, current_context, memory_features):
                # Encode current context
                context_emb = F.relu(self.context_encoder(current_context))
                
                # Encode memories
                memory_emb = F.relu(self.memory_encoder(memory_features))
                
                # Compute attention
                attended_memory, attention_weights = self.attention_head(
                    context_emb.unsqueeze(0), memory_emb.unsqueeze(0), memory_emb.unsqueeze(0)
                )
                
                # Output attention score
                attention_score = torch.sigmoid(self.output_layer(attended_memory.squeeze(0)))
                return attention_score, attention_weights
        
        return AttentionNetwork()
    
    def add_intervention_memory(self, decision_record: Dict[str, Any], 
                              immediate_impact: Dict[str, float]):
        """Add intervention to memory with immediate impact assessment"""
        
        memory_entry = {
            'timestamp': time.time(),
            'decision': decision_record,
            'immediate_impact': immediate_impact,
            'delayed_impacts': [],  # Will be filled over time
            'attention_score': 1.0,  # Initial high attention
            'crisis_context': decision_record.get('urgency', 0),
            'intervention_type': decision_record.get('chosen_action'),
            'success_probability': decision_record.get('success_probability', 0.5)
        }
        
        self.intervention_memories.append(memory_entry)
        
        # Compute embedding for this memory
        self._compute_memory_embedding(memory_entry)
    
    def _compute_memory_embedding(self, memory_entry: Dict[str, Any]):
        """Compute embedding representation of memory"""
        
        # Extract key features for embedding
        features = []
        
        # Context features
        crisis_level = memory_entry['crisis_context']
        success_prob = memory_entry['success_probability']
        
        context_features = torch.tensor([
            crisis_level, success_prob,
            len(memory_entry['delayed_impacts']),
            memory_entry['attention_score']
        ])
        features.append(context_features)
        
        # Impact features
        immediate_impact = memory_entry['immediate_impact']
        impact_vector = torch.tensor([
            immediate_impact.get('agents_affected', 0) / 100.0,
            immediate_impact.get('implementation_fidelity', 0),
            immediate_impact.get('effectiveness', 0)
        ])
        features.append(impact_vector)
        
        # Combine all features
        full_embedding = torch.cat(features)
        memory_key = id(memory_entry)
        self.memory_embeddings[memory_key] = full_embedding
    
    def compute_weighted_memory_influence(self, current_context) -> Dict[str, float]:
        """Compute weighted influence of memories on current decision"""
        
        if not self.intervention_memories:
            return {'memory_influence': 0.0, 'confidence_boost': 0.0}
        
        # Categorize memories by age
        recent_memories = list(self.intervention_memories)[-10:]
        medium_memories = list(self.intervention_memories)[-50:-10] if len(self.intervention_memories) > 10 else []
        long_memories = list(self.intervention_memories)[:-50] if len(self.intervention_memories) > 50 else []
        
        # Compute influence for each category
        influences = {}
        
        for category, memories, weight in [
            ('recent', recent_memories, self.attention_weights['recent_weight']),
            ('medium', medium_memories, self.attention_weights['medium_term_weight']),
            ('long', long_memories, self.attention_weights['long_term_weight'])
        ]:
            if memories:
                category_influence = self._compute_category_influence(memories, current_context)
                influences[category] = category_influence * weight
            else:
                influences[category] = 0.0
        
        # Crisis memory influence
        crisis_memories = [m for m in self.intervention_memories if m['crisis_context'] > 0.7]
        if crisis_memories:
            crisis_influence = self._compute_category_influence(crisis_memories, current_context)
            influences['crisis'] = crisis_influence * self.attention_weights['crisis_memory_weight']
        else:
            influences['crisis'] = 0.0
        
        total_influence = sum(influences.values())
        confidence_boost = min(0.3, total_influence * 0.5)  # Cap confidence boost
        
        return {
            'memory_influence': total_influence,
            'confidence_boost': confidence_boost,
            'category_influences': influences
        }
    
    def _compute_category_influence(self, memories: List[Dict[str, Any]], 
                                  current_context) -> float:
        """Compute influence of a category of memories"""
        
        if not memories:
            return 0.0
        
        total_influence = 0.0
        
        for memory in memories:
            # Similarity to current context (simplified)
            if isinstance(current_context, dict):
                context_crisis = current_context.get('colony_metrics', type('', (), {'crisis_level': lambda: 0})).crisis_level()
                memory_crisis = memory['crisis_context']
                similarity = 1.0 - abs(context_crisis - memory_crisis)
            else:
                similarity = 0.5  # Default similarity
            
            # Weight by attention score and similarity
            influence = memory['attention_score'] * max(0, similarity)
            total_influence += influence
        
        return total_influence / len(memories)
    
    def get_status(self) -> str:
        """Get memory system status"""
        return f"active ({len(self.intervention_memories)} memories, {len(self.memory_embeddings)} embeddings)"

if __name__ == "__main__":
    print("Memory & Wisdom Systems Module - Ready for Integration")