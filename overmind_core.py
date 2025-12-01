#!/usr/bin/env python3
"""
ENHANCED MODULE 1: CORE OVERMIND SYSTEM
Main overmind class with parallel thought streams, self-evolution logic,
and async processing capabilities
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import logging
import time
import json
import asyncio
import uuid
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== ENUMS AND DATA STRUCTURES =====

class OvermindActionType(Enum):
    """Actions available to the overmind"""
    NO_ACTION = 0
    TRIGGER_COLLECTIVE_MEDITATION = 1
    PROMOTE_COOPERATION = 2
    ENHANCE_WISDOM_PROPAGATION = 3
    INCREASE_RESOURCE_REGENERATION = 4
    REDUCE_ENVIRONMENTAL_HAZARDS = 5
    REDISTRIBUTE_RESOURCES = 6
    IMPROVE_COMMUNICATION = 7
    INITIATE_HEALING_PROTOCOL = 8
    ENCOURAGE_EXPLORATION = 9
    FACILITATE_KNOWLEDGE_TRANSFER = 10

class RitualType(Enum):
    """Types of collective rituals"""
    SYNCHRONIZED_MEDITATION = "synchronized_meditation"
    WISDOM_CIRCLE = "wisdom_circle"
    HARMONY_RESONANCE = "harmony_resonance"
    COLLECTIVE_INSIGHT = "collective_insight"
    CONFLICT_RESOLUTION_CIRCLE = "conflict_resolution_circle"

@dataclass
class OvermindDecision:
    """Enhanced decision with meta-evaluation capabilities"""
    chosen_action: OvermindActionType
    confidence: float = 0.7
    urgency: float = 0.5
    success_probability: float = 0.7
    justification: str = ""
    expected_impact: Dict[str, float] = field(default_factory=dict)
    neural_alignment_score: float = 0.5
    memory_influence: Dict[str, Any] = field(default_factory=dict)
    emotional_gradients: float = 0.0
    signal_entropy: float = 0.0
    meta_evaluation: Dict[str, Any] = field(default_factory=dict)
    parallel_insights: List[str] = field(default_factory=list)

class ColonyMetrics:
    """Enhanced colony state metrics with trend analysis"""
    def __init__(self, agents: List):
        self.total_population = len(agents)
        self.average_energy = np.mean([getattr(a, 'energy', 0.5) for a in agents])
        self.average_health = np.mean([getattr(a, 'health', 0.5) for a in agents])
        self.collective_mindfulness = np.mean([getattr(a, 'mindfulness_level', 0.5) for a in agents])
        self.cooperation_rate = self._calculate_cooperation_rate(agents)
        self.conflict_rate = self._calculate_conflict_rate(agents)
        self.wisdom_sharing_frequency = self._calculate_wisdom_sharing(agents)
        
        # Enhanced metrics
        self.innovation_capacity = np.mean([getattr(a, 'innovation_capacity', 0.4) for a in agents])
        self.emotional_stability = np.mean([getattr(a, 'emotional_stability', 0.5) for a in agents])
        self.adaptation_rate = self._calculate_adaptation_rate(agents)
        self.emergence_potential = self._calculate_emergence_potential(agents)
        
    def crisis_level(self) -> float:
        """Enhanced crisis calculation with multiple factors"""
        health_crisis = max(0, 1.0 - self.average_health)
        energy_crisis = max(0, 1.0 - self.average_energy)
        conflict_crisis = self.conflict_rate
        stability_crisis = max(0, 1.0 - self.emotional_stability)
        
        return min(1.0, (health_crisis + energy_crisis + conflict_crisis + stability_crisis) / 3.0)
    
    def overall_wellbeing(self) -> float:
        """Enhanced wellbeing calculation"""
        return (self.average_health * 0.25 + 
                self.average_energy * 0.25 + 
                self.collective_mindfulness * 0.2 +
                (1.0 - self.conflict_rate) * 0.15 +
                self.emotional_stability * 0.15)
    
    def _calculate_cooperation_rate(self, agents: List) -> float:
        """Calculate cooperation rate among agents"""
        if len(agents) < 2:
            return 1.0
        cooperative_actions = sum(1 for a in agents if getattr(a, 'cooperation_tendency', 0.5) > 0.6)
        return cooperative_actions / len(agents)
    
    def _calculate_conflict_rate(self, agents: List) -> float:
        """Calculate conflict rate among agents"""
        if len(agents) < 2:
            return 0.0
        conflict_count = sum(1 for a in agents if getattr(a, 'conflict_tendency', 0.2) > 0.5)
        return conflict_count / len(agents)
    
    def _calculate_wisdom_sharing(self, agents: List) -> float:
        """Calculate wisdom sharing frequency"""
        sharing_agents = sum(1 for a in agents if getattr(a, 'wisdom_accumulated', 0) > 2.0)
        return sharing_agents / max(1, len(agents))
    
    def _calculate_adaptation_rate(self, agents: List) -> float:
        """Calculate how quickly agents adapt to changes"""
        adaptation_scores = [getattr(a, 'learning_rate', 0.5) * getattr(a, 'emotional_stability', 0.5) 
                           for a in agents]
        return np.mean(adaptation_scores)
    
    def _calculate_emergence_potential(self, agents: List) -> float:
        """Calculate potential for emergent behaviors"""
        diversity = len(set(round(getattr(a, 'mindfulness_level', 0.5), 1) for a in agents)) / len(agents)
        innovation = np.mean([getattr(a, 'innovation_capacity', 0.4) for a in agents])
        return (diversity + innovation) / 2.0

class EnvironmentalState:
    """Enhanced environmental conditions with dynamics"""
    def __init__(self, temperature: float = 25.0, resource_abundance: float = 0.7):
        self.temperature = temperature
        self.resource_abundance = resource_abundance
        self.hazard_level = 0.2
        self.season = "balanced"
        self.dynamics = self._calculate_dynamics()
    
    def _calculate_dynamics(self) -> Dict[str, float]:
        """Calculate environmental dynamics"""
        return {
            'stability': 1.0 - abs(self.temperature - 25.0) / 25.0,
            'favorability': (self.resource_abundance + (1.0 - self.hazard_level)) / 2.0,
            'change_rate': np.random.uniform(0.1, 0.3)
        }

# ===== ENHANCED CORE OVERMIND CLASS =====

class ProductionReadyContemplativeOvermind:
    """
    Enhanced Production-ready Phase III Contemplative Overmind
    Main orchestrator with parallel thought streams and self-evolution
    """
    
    def __init__(self, environment, wisdom_signal_grid, overmind_id: str = None,
                 debug_mode: bool = False, json_logging: bool = False):
        
        # Core identification
        self.overmind_id = overmind_id or f"overmind_{uuid.uuid4().hex[:8]}"
        self.debug_mode = debug_mode
        self.json_logging = json_logging
        
        # Environment and external systems
        self.environment = environment
        self.wisdom_signal_grid = wisdom_signal_grid
        
        # Enhanced parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.parallel_tasks = {}
        self.thought_streams = {
            'insight_generation': asyncio.Queue(),
            'feedback_processing': asyncio.Queue(),
            'ritual_coordination': asyncio.Queue(),
            'meta_evaluation': asyncio.Queue()
        }
        
        # Initialize components (imported from other modules)
        self._initialize_components()
        
        # Enhanced tracking
        self.decision_history = deque(maxlen=1000)
        self.intervention_frequency_tracker = deque(maxlen=100)
        self.meta_evaluation_history = deque(maxlen=200)
        self.self_evolution_metrics = {
            'adaptation_score': 0.5,
            'learning_velocity': 0.0,
            'wisdom_accumulation_rate': 0.0,
            'decision_quality_trend': 0.0
        }
        
        # Performance metrics
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_interventions': 0,
            'failed_interventions': 0,
            'average_processing_time': 0.0,
            'meta_evaluations_completed': 0,
            'self_corrections_made': 0
        }
        
        # State management
        self.system_state = {
            'status': 'active',
            'last_decision_step': -1,
            'initialization_time': time.time(),
            'total_runtime': 0.0,
            'evolution_phase': 'learning',
            'parallel_streams_active': 0
        }
        
        logger.info(f"Enhanced Production Overmind '{self.overmind_id}' initialized with parallel processing")
    
    def _initialize_components(self):
        """Initialize all component modules with enhanced features"""
        try:
            # Import enhanced modules
            from memory_wisdom import WisdomArchive, MemoryAttentionMechanism
            from feedback_ritual import AgentFeedbackInterface, RitualProtocolLayer
            from monitoring_logging import (EnhancedLogger, PerformanceMonitor, 
                                           StatusReporter, TestSuite)
            from neural_adaptive import ThresholdRegulator, NeuralAlignment
            
            # Enhanced memory and wisdom systems
            self.wisdom_archive = WisdomArchive()
            self.memory_attention = MemoryAttentionMechanism()
            
            # Enhanced agent interaction systems
            self.agent_feedback = AgentFeedbackInterface()
            self.ritual_layer = RitualProtocolLayer()
            
            # Enhanced monitoring and logging
            self.logger = EnhancedLogger(self.overmind_id, self.json_logging)
            self.performance_monitor = PerformanceMonitor()
            self.status_reporter = StatusReporter(self)
            self.test_suite = TestSuite(self)
            
            # Enhanced adaptive systems
            self.threshold_regulator = ThresholdRegulator()
            self.neural_alignment = NeuralAlignment()
            
            # Meta-evaluation system
            self.meta_evaluator = self._create_meta_evaluator()
            
            logger.info("All enhanced components initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Some enhanced modules not available: {e}")
            # Initialize fallback components
            self._initialize_fallback_components()
    
    def _create_meta_evaluator(self):
        """Create meta-evaluation system for self-reflection"""
        
        class MetaEvaluator:
            def __init__(self, overmind):
                self.overmind = overmind
                self.evaluation_criteria = {
                    'decision_accuracy': 0.0,
                    'intervention_timing': 0.0,
                    'resource_efficiency': 0.0,
                    'agent_satisfaction': 0.0,
                    'emergence_facilitation': 0.0
                }
                self.reflection_history = deque(maxlen=100)
            
            def evaluate_recent_decisions(self, window_size: int = 10) -> Dict[str, float]:
                """Evaluate recent decisions for self-improvement"""
                
                recent_decisions = list(self.overmind.decision_history)[-window_size:]
                if not recent_decisions:
                    return self.evaluation_criteria.copy()
                
                evaluation = {}
                
                # Decision accuracy (success rate)
                successful = sum(1 for d in recent_decisions 
                               if d.get('decision') and d['decision'].chosen_action != OvermindActionType.NO_ACTION)
                evaluation['decision_accuracy'] = successful / len(recent_decisions)
                
                # Intervention timing (based on urgency vs actual need)
                timing_scores = []
                for d in recent_decisions:
                    if d.get('decision'):
                        predicted_urgency = d['decision'].urgency
                        # Simulate actual urgency (would be measured from outcomes)
                        actual_urgency = predicted_urgency + np.random.uniform(-0.2, 0.2)
                        timing_score = 1.0 - abs(predicted_urgency - actual_urgency)
                        timing_scores.append(max(0, timing_score))
                
                evaluation['intervention_timing'] = np.mean(timing_scores) if timing_scores else 0.5
                
                # Resource efficiency (processing time vs decision quality)
                efficiency_scores = []
                for d in recent_decisions:
                    processing_time = d.get('processing_time', 1.0)
                    decision_quality = d.get('decision', type('', (), {'confidence': 0.5})).confidence
                    efficiency = decision_quality / max(0.1, processing_time)
                    efficiency_scores.append(min(1.0, efficiency))
                
                evaluation['resource_efficiency'] = np.mean(efficiency_scores) if efficiency_scores else 0.5
                
                # Agent satisfaction (simulated based on feedback effectiveness)
                satisfaction_scores = []
                for d in recent_decisions:
                    feedback_results = getattr(d.get('decision'), 'feedback_results', {})
                    if feedback_results:
                        effectiveness = feedback_results.get('overall_effectiveness', 0.5)
                        satisfaction_scores.append(effectiveness)
                
                evaluation['agent_satisfaction'] = np.mean(satisfaction_scores) if satisfaction_scores else 0.5
                
                # Emergence facilitation (growth in collective metrics)
                evaluation['emergence_facilitation'] = 0.6  # Placeholder - would measure actual emergence
                
                return evaluation
            
            def generate_self_corrections(self, evaluation: Dict[str, float]) -> List[str]:
                """Generate self-correction recommendations"""
                
                corrections = []
                
                if evaluation['decision_accuracy'] < 0.6:
                    corrections.append("Increase observation period before making decisions")
                    corrections.append("Enhance context analysis with additional data sources")
                
                if evaluation['intervention_timing'] < 0.5:
                    corrections.append("Recalibrate urgency assessment algorithms")
                    corrections.append("Implement predictive timing models")
                
                if evaluation['resource_efficiency'] < 0.4:
                    corrections.append("Optimize parallel processing allocation")
                    corrections.append("Implement decision confidence thresholds")
                
                if evaluation['agent_satisfaction'] < 0.5:
                    corrections.append("Enhance agent feedback integration mechanisms")
                    corrections.append("Improve communication protocols with agents")
                
                if evaluation['emergence_facilitation'] < 0.6:
                    corrections.append("Expand wisdom propagation strategies")
                    corrections.append("Increase ritual diversity and frequency")
                
                return corrections
        
        return MetaEvaluator(self)
    
    def _initialize_fallback_components(self):
        """Initialize basic fallback components if enhanced modules unavailable"""
        from overmind_core import (BasicWisdomArchive, BasicMemoryAttention, 
                                   BasicAgentFeedback, BasicRitualLayer, BasicThresholdRegulator)
        
        self.wisdom_archive = BasicWisdomArchive()
        self.memory_attention = BasicMemoryAttention()
        self.agent_feedback = BasicAgentFeedback()
        self.ritual_layer = BasicRitualLayer()
        self.threshold_regulator = BasicThresholdRegulator()
        self.meta_evaluator = None
    
    async def process_colony_state_with_parallel_streams(self, agents: List, step: int) -> Optional[OvermindDecision]:
        """
        Enhanced processing with parallel thought streams
        """
        start_time = time.time()
        
        try:
            # Start parallel thought streams
            parallel_tasks = await self._start_parallel_thought_streams(agents, step)
            
            # Main decision processing
            decision = await self._process_main_decision_stream(agents, step, parallel_tasks)
            
            # Integrate parallel insights
            if decision:
                decision.parallel_insights = await self._integrate_parallel_insights(parallel_tasks)
            
            # Meta-evaluation in background
            if self.meta_evaluator and len(self.decision_history) > 5:
                asyncio.create_task(self._perform_meta_evaluation_async())
            
            # Update systems
            self._update_systems(decision, self._build_complete_context(agents, step), step)
            
            # Record metrics
            processing_time = time.time() - start_time
            self._record_decision_metrics(decision, processing_time, step)
            
            return decision
            
        except Exception as e:
            return self._handle_processing_exception(e, step, time.time() - start_time)
    
    async def _start_parallel_thought_streams(self, agents: List, step: int) -> Dict[str, asyncio.Task]:
        """Start parallel processing streams for different aspects"""
        
        tasks = {}
        
        # Insight generation stream
        tasks['insight_generation'] = asyncio.create_task(
            self._generate_insights_async(agents, step)
        )
        
        # Feedback processing stream
        tasks['feedback_processing'] = asyncio.create_task(
            self._process_agent_feedback_async(agents, step)
        )
        
        # Ritual coordination stream
        tasks['ritual_coordination'] = asyncio.create_task(
            self._coordinate_rituals_async(agents, step)
        )
        
        # Environmental analysis stream
        tasks['environmental_analysis'] = asyncio.create_task(
            self._analyze_environment_async(step)
        )
        
        self.system_state['parallel_streams_active'] = len(tasks)
        
        return tasks
    
    async def _generate_insights_async(self, agents: List, step: int) -> List[Dict[str, Any]]:
        """Async insight generation stream"""
        
        insights = []
        
        try:
            # Extract wisdom insights from agents
            for agent in agents[:10]:  # Process subset to avoid overwhelming
                if hasattr(agent, 'recent_insights'):
                    for insight_text in getattr(agent, 'recent_insights', [])[-2:]:
                        context = {
                            'agent_id': getattr(agent, 'id', 'unknown'),
                            'agent_mindfulness': getattr(agent, 'mindfulness_level', 0.5),
                            'step': step,
                            'emergence_context': 'parallel_stream'
                        }
                        
                        # Use neural alignment if available
                        if hasattr(self, 'neural_alignment'):
                            embedding = await self._encode_insight_async(insight_text, context)
                            insights.append({
                                'text': insight_text,
                                'embedding': embedding,
                                'source': 'agent_insight'
                            })
                        else:
                            insights.append({
                                'text': insight_text,
                                'source': 'agent_insight'
                            })
                        
                        # Limit insights per stream
                        if len(insights) >= 5:
                            break
                    
                    if len(insights) >= 5:
                        break
            
            # Generate emergent insights from collective patterns
            collective_insight = self._generate_collective_insight(agents)
            if collective_insight:
                insights.append({
                    'text': collective_insight,
                    'source': 'collective_emergence'
                })
            
        except Exception as e:
            logger.warning(f"Insight generation stream error: {e}")
            insights.append({
                'text': "Insight generation encountered challenges, maintaining awareness",
                'source': 'system_fallback'
            })
        
        return insights
    
    async def _encode_insight_async(self, insight_text: str, context: Dict[str, Any]):
        """Async insight encoding"""
        
        loop = asyncio.get_event_loop()
        
        def _encode():
            return self.neural_alignment.encode_wisdom_insight(insight_text, context)
        
        return await loop.run_in_executor(self.executor, _encode)
    
    def _generate_collective_insight(self, agents: List) -> Optional[str]:
        """Generate insight from collective agent patterns"""
        
        # Analyze collective state
        avg_mindfulness = np.mean([getattr(a, 'mindfulness_level', 0.5) for a in agents])
        avg_cooperation = np.mean([getattr(a, 'cooperation_tendency', 0.5) for a in agents])
        avg_energy = np.mean([getattr(a, 'energy', 0.5) for a in agents])
        
        # Generate insight based on collective state
        if avg_mindfulness > 0.7 and avg_cooperation > 0.7:
            return "Collective harmony emerges when mindfulness and cooperation align"
        elif avg_energy < 0.3:
            return "Sustainable practices arise naturally when energy becomes scarce"
        elif abs(avg_mindfulness - avg_cooperation) < 0.1:
            return "Balance creates resonance between inner awareness and outer action"
        
        return None
    
    async def _process_agent_feedback_async(self, agents: List, step: int) -> Dict[str, Any]:
        """Async agent feedback processing stream"""
        
        feedback_analysis = {
            'effectiveness_trends': {},
            'agent_responsiveness': {},
            'recommended_adjustments': []
        }
        
        try:
            if hasattr(self.agent_feedback, 'get_feedback_effectiveness_summary'):
                effectiveness = self.agent_feedback.get_feedback_effectiveness_summary()
                feedback_analysis['effectiveness_trends'] = effectiveness
                
                # Analyze trends and generate recommendations
                for feedback_type, metrics in effectiveness.items():
                    if metrics.get('average_effectiveness', 0) < 0.5:
                        feedback_analysis['recommended_adjustments'].append(
                            f"Improve {feedback_type} delivery methods"
                        )
            
            # Analyze agent responsiveness
            responsive_agents = sum(1 for agent in agents 
                                  if getattr(agent, 'learning_rate', 0.5) > 0.6)
            feedback_analysis['agent_responsiveness']['responsive_rate'] = responsive_agents / len(agents)
            
        except Exception as e:
            logger.warning(f"Feedback processing stream error: {e}")
            feedback_analysis['error'] = str(e)
        
        return feedback_analysis
    
    async def _coordinate_rituals_async(self, agents: List, step: int) -> Dict[str, Any]:
        """Async ritual coordination stream"""
        
        ritual_analysis = {
            'opportunities': [],
            'active_rituals': 0,
            'effectiveness_predictions': {}
        }
        
        try:
            # Assess ritual opportunities
            if hasattr(self.ritual_layer, 'assess_ritual_opportunities'):
                colony_metrics = ColonyMetrics(agents)
                opportunities = self.ritual_layer.assess_ritual_opportunities(agents, colony_metrics)
                ritual_analysis['opportunities'] = [op.value for op in opportunities]
            
            # Count active rituals
            if hasattr(self.ritual_layer, 'active_rituals'):
                ritual_analysis['active_rituals'] = len(self.ritual_layer.active_rituals)
            
            # Predict effectiveness
            for opportunity in ritual_analysis['opportunities'][:3]:
                ritual_analysis['effectiveness_predictions'][opportunity] = np.random.uniform(0.4, 0.9)
            
        except Exception as e:
            logger.warning(f"Ritual coordination stream error: {e}")
            ritual_analysis['error'] = str(e)
        
        return ritual_analysis
    
    async def _analyze_environment_async(self, step: int) -> Dict[str, Any]:
        """Async environmental analysis stream"""
        
        env_analysis = {
            'stability_trend': 0.0,
            'resource_projection': 0.0,
            'hazard_assessment': 0.0,
            'adaptation_requirements': []
        }
        
        try:
            # Analyze environmental dynamics
            if hasattr(self.environment, 'dynamics'):
                dynamics = self.environment.dynamics
                env_analysis['stability_trend'] = dynamics.get('stability', 0.5)
                env_analysis['resource_projection'] = self.environment.resource_abundance
                env_analysis['hazard_assessment'] = getattr(self.environment, 'hazard_level', 0.2)
            
            # Generate adaptation requirements
            if env_analysis['resource_projection'] < 0.4:
                env_analysis['adaptation_requirements'].append('resource_conservation_protocols')
            
            if env_analysis['hazard_assessment'] > 0.6:
                env_analysis['adaptation_requirements'].append('hazard_mitigation_strategies')
            
        except Exception as e:
            logger.warning(f"Environmental analysis stream error: {e}")
            env_analysis['error'] = str(e)
        
        return env_analysis
    
    async def _process_main_decision_stream(self, agents: List, step: int, 
                                          parallel_tasks: Dict[str, asyncio.Task]) -> Optional[OvermindDecision]:
        """Process main decision while parallel streams are running"""
        
        # Build context
        context = self._build_complete_context(agents, step)
        
        # Memory attention influence
        memory_influence = self.memory_attention.compute_weighted_memory_influence(context)
        
        # Assess intervention necessity
        intervention_needed = self._assess_intervention_necessity(context)
        
        # Make decision with enhanced context
        decision = self._make_enhanced_decision(
            context, memory_influence, intervention_needed, agents, step
        )
        
        return decision
    
    async def _integrate_parallel_insights(self, parallel_tasks: Dict[str, asyncio.Task]) -> List[str]:
        """Integrate insights from parallel processing streams"""
        
        integrated_insights = []
        
        try:
            # Wait for all tasks to complete (with timeout)
            results = await asyncio.wait_for(
                asyncio.gather(*parallel_tasks.values(), return_exceptions=True),
                timeout=2.0
            )
            
            task_names = list(parallel_tasks.keys())
            
            for i, result in enumerate(results):
                task_name = task_names[i]
                
                if isinstance(result, Exception):
                    integrated_insights.append(f"{task_name}_stream_error: {str(result)[:50]}")
                    continue
                
                # Extract insights based on task type
                if task_name == 'insight_generation' and isinstance(result, list):
                    for insight in result[:2]:  # Limit insights
                        integrated_insights.append(f"Generated: {insight.get('text', 'Unknown')[:100]}")
                
                elif task_name == 'feedback_processing' and isinstance(result, dict):
                    if result.get('recommended_adjustments'):
                        integrated_insights.append(f"Feedback: {result['recommended_adjustments'][0]}")
                
                elif task_name == 'ritual_coordination' and isinstance(result, dict):
                    if result.get('opportunities'):
                        integrated_insights.append(f"Ritual: {result['opportunities'][0]} available")
                
                elif task_name == 'environmental_analysis' and isinstance(result, dict):
                    if result.get('adaptation_requirements'):
                        integrated_insights.append(f"Environment: {result['adaptation_requirements'][0]} needed")
        
        except asyncio.TimeoutError:
            integrated_insights.append("Parallel processing timeout - using partial results")
        except Exception as e:
            integrated_insights.append(f"Integration error: {str(e)[:50]}")
        
        return integrated_insights[:5]  # Limit to top 5 insights
    
    async def _perform_meta_evaluation_async(self):
        """Perform meta-evaluation in background"""
        
        try:
            if self.meta_evaluator:
                evaluation = self.meta_evaluator.evaluate_recent_decisions()
                corrections = self.meta_evaluator.generate_self_corrections(evaluation)
                
                # Record meta-evaluation
                meta_record = {
                    'timestamp': time.time(),
                    'evaluation': evaluation,
                    'corrections': corrections,
                    'overall_score': np.mean(list(evaluation.values()))
                }
                
                self.meta_evaluation_history.append(meta_record)
                self.performance_metrics['meta_evaluations_completed'] += 1
                
                # Apply self-corrections if needed
                if meta_record['overall_score'] < 0.5:
                    await self._apply_self_corrections(corrections)
                
                # Update evolution metrics
                self._update_self_evolution_metrics(evaluation)
                
        except Exception as e:
            logger.warning(f"Meta-evaluation error: {e}")
    
    async def _apply_self_corrections(self, corrections: List[str]):
        """Apply self-corrections to improve performance"""
        
        corrections_applied = 0
        
        for correction in corrections[:3]:  # Limit corrections per cycle
            try:
                if "observation period" in correction.lower():
                    # Increase processing time allocation
                    self.system_state['min_observation_time'] = getattr(
                        self.system_state, 'min_observation_time', 0.1
                    ) + 0.05
                    corrections_applied += 1
                
                elif "confidence threshold" in correction.lower():
                    # Adjust decision confidence requirements
                    if hasattr(self.threshold_regulator, 'thresholds'):
                        current = self.threshold_regulator.thresholds.get('intervention_threshold', 0.6)
                        self.threshold_regulator.thresholds['intervention_threshold'] = min(0.9, current + 0.05)
                        corrections_applied += 1
                
                elif "feedback integration" in correction.lower():
                    # Enhance feedback processing
                    if hasattr(self.agent_feedback, 'adaptation_rates'):
                        for feedback_type in self.agent_feedback.adaptation_rates:
                            current_rate = self.agent_feedback.adaptation_rates[feedback_type]
                            self.agent_feedback.adaptation_rates[feedback_type] = min(1.0, current_rate * 1.1)
                        corrections_applied += 1
                
            except Exception as e:
                logger.warning(f"Failed to apply correction '{correction}': {e}")
        
        if corrections_applied > 0:
            self.performance_metrics['self_corrections_made'] += corrections_applied
            logger.info(f"Applied {corrections_applied} self-corrections")
    
    def _update_self_evolution_metrics(self, evaluation: Dict[str, float]):
        """Update self-evolution tracking metrics"""
        
        # Calculate adaptation score
        current_performance = np.mean(list(evaluation.values()))
        
        if len(self.meta_evaluation_history) > 1:
            previous_performance = self.meta_evaluation_history[-2]['overall_score']
            learning_velocity = current_performance - previous_performance
            self.self_evolution_metrics['learning_velocity'] = learning_velocity
        
        self.self_evolution_metrics['adaptation_score'] = current_performance
        
        # Calculate decision quality trend
        if len(self.decision_history) >= 10:
            recent_confidences = [d.get('decision', type('', (), {'confidence': 0.5})).confidence 
                                for d in list(self.decision_history)[-10:]]
            self.self_evolution_metrics['decision_quality_trend'] = np.mean(recent_confidences)
        
        # Update evolution phase
        if current_performance > 0.8:
            self.system_state['evolution_phase'] = 'mastery'
        elif current_performance > 0.6:
            self.system_state['evolution_phase'] = 'integration'
        else:
            self.system_state['evolution_phase'] = 'learning'
    
    # Existing methods with enhanced functionality
    def process_colony_state_fully_integrated(self, agents: List, step: int) -> Optional[OvermindDecision]:
        """
        Main processing method - can run sync or async based on system state
        """
        
        # Use async processing if parallel streams are enabled
        if self.system_state.get('parallel_streams_enabled', True):
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    self.process_colony_state_with_parallel_streams(agents, step)
                )
            except Exception as e:
                logger.warning(f"Async processing failed: {e}, falling back to sync")
        
        # Fallback to synchronous processing
        return self._process_synchronously(agents, step)
    
    def _process_synchronously(self, agents: List, step: int) -> Optional[OvermindDecision]:
        """Synchronous processing fallback"""
        
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_inputs(agents, step)
            
            # Build context
            context = self._build_complete_context(agents, step)
            
            # Memory influence
            memory_influence = self.memory_attention.compute_weighted_memory_influence(context)
            
            # Assess intervention
            intervention_needed = self._assess_intervention_necessity(context)
            
            # Make decision
            decision = self._make_enhanced_decision(
                context, memory_influence, intervention_needed, agents, step
            )
            
            # Apply feedback
            if decision and decision.chosen_action != OvermindActionType.NO_ACTION:
                feedback_results = self._apply_agent_feedback(decision, agents)
                decision.feedback_results = feedback_results
            
            # Update systems
            self._update_systems(decision, context, step)
            
            # Record metrics
            processing_time = time.time() - start_time
            self._record_decision_metrics(decision, processing_time, step)
            
            return decision
            
        except Exception as e:
            return self._handle_processing_exception(e, step, time.time() - start_time)
    
    # Continue with rest of existing methods...#!/usr/bin/env python3
"""
MODULE 1: CORE OVERMIND SYSTEM
Main overmind class with decision-making logic and integration points
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import logging
import time
import json
from collections import deque, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== ENUMS AND DATA STRUCTURES =====

class OvermindActionType(Enum):
    """Actions available to the overmind"""
    NO_ACTION = 0
    TRIGGER_COLLECTIVE_MEDITATION = 1
    PROMOTE_COOPERATION = 2
    ENHANCE_WISDOM_PROPAGATION = 3
    INCREASE_RESOURCE_REGENERATION = 4
    REDUCE_ENVIRONMENTAL_HAZARDS = 5
    REDISTRIBUTE_RESOURCES = 6
    IMPROVE_COMMUNICATION = 7
    INITIATE_HEALING_PROTOCOL = 8
    ENCOURAGE_EXPLORATION = 9
    FACILITATE_KNOWLEDGE_TRANSFER = 10

class RitualType(Enum):
    """Types of collective rituals"""
    SYNCHRONIZED_MEDITATION = "synchronized_meditation"
    WISDOM_CIRCLE = "wisdom_circle"
    HARMONY_RESONANCE = "harmony_resonance"
    COLLECTIVE_INSIGHT = "collective_insight"
    CONFLICT_RESOLUTION_CIRCLE = "conflict_resolution_circle"

@dataclass
class OvermindDecision:
    """Decision made by overmind with comprehensive metadata"""
    chosen_action: OvermindActionType
    confidence: float = 0.7
    urgency: float = 0.5
    success_probability: float = 0.7
    justification: str = ""
    expected_impact: Dict[str, float] = field(default_factory=dict)
    neural_alignment_score: float = 0.5
    memory_influence: Dict[str, Any] = field(default_factory=dict)
    emotional_gradients: float = 0.0
    signal_entropy: float = 0.0

class ColonyMetrics:
    """Comprehensive colony state metrics"""
    def __init__(self, agents: List):
        self.total_population = len(agents)
        self.average_energy = np.mean([getattr(a, 'energy', 0.5) for a in agents])
        self.average_health = np.mean([getattr(a, 'health', 0.5) for a in agents])
        self.collective_mindfulness = np.mean([getattr(a, 'mindfulness_level', 0.5) for a in agents])
        self.cooperation_rate = self._calculate_cooperation_rate(agents)
        self.conflict_rate = self._calculate_conflict_rate(agents)
        self.wisdom_sharing_frequency = self._calculate_wisdom_sharing(agents)
        
    def crisis_level(self) -> float:
        """Calculate overall crisis level"""
        health_crisis = max(0, 1.0 - self.average_health)
        energy_crisis = max(0, 1.0 - self.average_energy)
        conflict_crisis = self.conflict_rate
        return min(1.0, (health_crisis + energy_crisis + conflict_crisis) / 2.0)
    
    def overall_wellbeing(self) -> float:
        """Calculate overall colony wellbeing"""
        return (self.average_health * 0.3 + 
                self.average_energy * 0.3 + 
                self.collective_mindfulness * 0.2 +
                (1.0 - self.conflict_rate) * 0.2)
    
    def _calculate_cooperation_rate(self, agents: List) -> float:
        """Calculate cooperation rate among agents"""
        if len(agents) < 2:
            return 1.0
        cooperative_actions = sum(1 for a in agents if getattr(a, 'cooperation_tendency', 0.5) > 0.6)
        return cooperative_actions / len(agents)
    
    def _calculate_conflict_rate(self, agents: List) -> float:
        """Calculate conflict rate among agents"""
        if len(agents) < 2:
            return 0.0
        conflict_count = sum(1 for a in agents if getattr(a, 'conflict_tendency', 0.2) > 0.5)
        return conflict_count / len(agents)
    
    def _calculate_wisdom_sharing(self, agents: List) -> float:
        """Calculate wisdom sharing frequency"""
        sharing_agents = sum(1 for a in agents if getattr(a, 'wisdom_accumulated', 0) > 2.0)
        return sharing_agents / max(1, len(agents))

class EnvironmentalState:
    """Environmental conditions"""
    def __init__(self, temperature: float = 25.0, resource_abundance: float = 0.7):
        self.temperature = temperature
        self.resource_abundance = resource_abundance
        self.hazard_level = 0.2
        self.season = "balanced"

# ===== CORE OVERMIND CLASS =====

class ProductionReadyContemplativeOvermind:
    """
    Production-ready Phase III Contemplative Overmind
    Main orchestrator with comprehensive error handling and monitoring
    """
    
    def __init__(self, environment, wisdom_signal_grid, overmind_id: str = None,
                 debug_mode: bool = False, json_logging: bool = False):
        
        # Core identification
        self.overmind_id = overmind_id or f"overmind_{uuid.uuid4().hex[:8]}"
        self.debug_mode = debug_mode
        self.json_logging = json_logging
        
        # Environment and external systems
        self.environment = environment
        self.wisdom_signal_grid = wisdom_signal_grid
        
        # Initialize components (imported from other modules)
        self._initialize_components()
        
        # Core tracking
        self.decision_history = deque(maxlen=1000)
        self.intervention_frequency_tracker = deque(maxlen=100)
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_interventions': 0,
            'failed_interventions': 0,
            'average_processing_time': 0.0
        }
        
        # State management
        self.system_state = {
            'status': 'active',
            'last_decision_step': -1,
            'initialization_time': time.time(),
            'total_runtime': 0.0
        }
        
        logger.info(f"Production Overmind '{self.overmind_id}' initialized successfully")
    
    def _initialize_components(self):
        """Initialize all component modules"""
        # These will be imported from other modules
        from memory_wisdom import WisdomArchive, MemoryAttentionMechanism
        from feedback_ritual import AgentFeedbackInterface, RitualProtocolLayer
        from monitoring_logging import (EnhancedLogger, PerformanceMonitor, 
                                       StatusReporter, TestSuite)
        
        try:
            # Memory and wisdom systems
            self.wisdom_archive = WisdomArchive()
            self.memory_attention = MemoryAttentionMechanism()
            
            # Agent interaction systems
            self.agent_feedback = AgentFeedbackInterface()
            self.ritual_layer = RitualProtocolLayer()
            
            # Monitoring and logging
            self.logger = EnhancedLogger(self.overmind_id, self.json_logging)
            self.performance_monitor = PerformanceMonitor()
            self.status_reporter = StatusReporter(self)
            self.test_suite = TestSuite(self)
            
            # Adaptive thresholds (imported from neural module)
            from neural_adaptive import ThresholdRegulator, NeuralAlignment
            self.threshold_regulator = ThresholdRegulator()
            self.neural_alignment = NeuralAlignment()
            
            logger.info("All components initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Some modules not available: {e}")
            # Initialize fallback components
            self._initialize_fallback_components()
    
    def _initialize_fallback_components(self):
        """Initialize basic fallback components if modules unavailable"""
        self.wisdom_archive = BasicWisdomArchive()
        self.memory_attention = BasicMemoryAttention()
        self.agent_feedback = BasicAgentFeedback()
        self.ritual_layer = BasicRitualLayer()
        self.threshold_regulator = BasicThresholdRegulator()
        
    def process_colony_state_fully_integrated(self, agents: List, step: int) -> Optional[OvermindDecision]:
        """
        Main processing method with full error handling and monitoring
        """
        start_time = time.time()
        
        try:
            with self.performance_monitor.measure_operation("full_processing"):
                # 1. Validate inputs
                self._validate_inputs(agents, step)
                
                # 2. Build comprehensive context
                context = self._build_complete_context(agents, step)
                
                # 3. Memory attention influence
                memory_influence = self.memory_attention.compute_weighted_memory_influence(context)
                
                # 4. Assess ritual opportunities
                ritual_opportunities = self.ritual_layer.assess_ritual_opportunities(agents, context['colony_metrics'])
                
                # 5. Adaptive threshold check
                intervention_needed = self._assess_intervention_necessity(context)
                
                # 6. Make decision with all inputs
                decision = self._make_integrated_decision(
                    context, memory_influence, ritual_opportunities, 
                    intervention_needed, agents, step
                )
                
                # 7. Apply agent feedback if intervention made
                if decision and decision.chosen_action != OvermindActionType.NO_ACTION:
                    feedback_results = self._apply_agent_feedback(decision, agents)
                    decision.feedback_results = feedback_results
                
                # 8. Update systems
                self._update_systems(decision, context, step)
                
                # 9. Record metrics
                processing_time = time.time() - start_time
                self._record_decision_metrics(decision, processing_time, step)
                
                return decision
                
        except Exception as e:
            return self._handle_processing_exception(e, step, time.time() - start_time)
    
    def _validate_inputs(self, agents: List, step: int):
        """Validate inputs with comprehensive checks"""
        if not agents:
            raise ValueError("Agent list cannot be empty")
        
        if step < 0:
            raise ValueError(f"Step must be non-negative, got {step}")
        
        # Validate agent attributes
        required_attrs = ['energy', 'health', 'mindfulness_level']
        for i, agent in enumerate(agents[:5]):  # Check first 5 agents
            for attr in required_attrs:
                if not hasattr(agent, attr):
                    logger.warning(f"Agent {i} missing attribute {attr}")
    
    def _build_complete_context(self, agents: List, step: int) -> Dict[str, Any]:
        """Build comprehensive context for decision making"""
        context = {
            'step': step,
            'agents': agents,
            'colony_metrics': ColonyMetrics(agents),
            'environmental_state': self.environment,
            'wisdom_insights': self._extract_recent_insights(agents),
            'signal_entropy': self._calculate_signal_entropy(agents),
            'emotional_gradients': self._calculate_emotional_gradients(agents)
        }
        
        # Add system state
        context['system_state'] = self.system_state.copy()
        context['active_rituals'] = len(getattr(self.ritual_layer, 'active_rituals', {}))
        
        return context
    
    def _extract_recent_insights(self, agents: List) -> List[str]:
        """Extract recent wisdom insights from agents"""
        insights = []
        for agent in agents:
            if hasattr(agent, 'recent_insights'):
                insights.extend(getattr(agent, 'recent_insights', [])[-2:])
        return insights[-10:]  # Last 10 insights
    
    def _calculate_signal_entropy(self, agents: List) -> float:
        """Calculate signal entropy for emergency detection"""
        signals = []
        for agent in agents:
            agent_signal = [
                getattr(agent, 'energy', 0.5),
                getattr(agent, 'health', 0.5),
                getattr(agent, 'mindfulness_level', 0.5),
                getattr(agent, 'cooperation_tendency', 0.5)
            ]
            signals.extend(agent_signal)
        
        if not signals:
            return 0.0
        
        # Simple entropy calculation
        signals_array = np.array(signals)
        unique_vals, counts = np.unique(np.round(signals_array, 1), return_counts=True)
        probabilities = counts / len(signals_array)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy / np.log2(len(unique_vals))  # Normalize
    
    def _calculate_emotional_gradients(self, agents: List) -> float:
        """Calculate emotional state gradients"""
        if not hasattr(self, '_previous_emotional_states'):
            self._previous_emotional_states = {}
        
        current_states = {}
        gradients = []
        
        for agent in agents:
            agent_id = getattr(agent, 'id', id(agent))
            emotional_state = (
                getattr(agent, 'energy', 0.5) * 0.4 +
                getattr(agent, 'health', 0.5) * 0.3 +
                getattr(agent, 'mindfulness_level', 0.5) * 0.3
            )
            current_states[agent_id] = emotional_state
            
            if agent_id in self._previous_emotional_states:
                gradient = emotional_state - self._previous_emotional_states[agent_id]
                gradients.append(gradient)
        
        self._previous_emotional_states = current_states
        return np.mean(gradients) if gradients else 0.0
    
    def _assess_intervention_necessity(self, context: Dict[str, Any]) -> bool:
        """Assess if intervention is necessary using adaptive thresholds"""
        crisis_level = context['colony_metrics'].crisis_level()
        signal_entropy = context['signal_entropy']
        
        intervention_threshold = self.threshold_regulator.get_threshold('intervention_threshold')
        crisis_threshold = self.threshold_regulator.get_threshold('crisis_detection_threshold')
        
        return crisis_level > crisis_threshold or signal_entropy > intervention_threshold
    
    def _make_integrated_decision(self, context: Dict[str, Any], memory_influence: Dict[str, float],
                                ritual_opportunities: List, intervention_needed: bool,
                                agents: List, step: int) -> Optional[OvermindDecision]:
        """Make decision integrating all available information"""
        
        colony_metrics = context['colony_metrics']
        
        # Determine action based on priority
        chosen_action = OvermindActionType.NO_ACTION
        justification = "No intervention needed"
        base_confidence = 0.6
        
        if intervention_needed:
            # Crisis response
            if colony_metrics.crisis_level() > 0.7:
                if colony_metrics.average_energy < 0.3:
                    chosen_action = OvermindActionType.INCREASE_RESOURCE_REGENERATION
                    justification = f"Energy crisis: {colony_metrics.average_energy:.2f}"
                elif colony_metrics.conflict_rate > 0.6:
                    chosen_action = OvermindActionType.PROMOTE_COOPERATION
                    justification = f"Conflict crisis: {colony_metrics.conflict_rate:.2f}"
                else:
                    chosen_action = OvermindActionType.REDUCE_ENVIRONMENTAL_HAZARDS
                    justification = "General crisis intervention"
                base_confidence = 0.8
            
            # Ritual opportunities
            elif ritual_opportunities:
                chosen_action = OvermindActionType.TRIGGER_COLLECTIVE_MEDITATION
                justification = f"Ritual opportunity: {ritual_opportunities[0].value}"
                base_confidence = 0.7
            
            # Proactive improvements
            elif colony_metrics.collective_mindfulness < 0.5:
                chosen_action = OvermindActionType.TRIGGER_COLLECTIVE_MEDITATION
                justification = "Mindfulness enhancement needed"
                base_confidence = 0.6
        
        # Apply memory influence
        confidence_boost = memory_influence.get('confidence_boost', 0.0)
        final_confidence = min(1.0, base_confidence + confidence_boost)
        
        # Calculate urgency and success probability
        urgency = context['colony_metrics'].crisis_level()
        success_probability = final_confidence * 0.8 + (1.0 - context['signal_entropy']) * 0.2
        
        # Create decision
        decision = OvermindDecision(
            chosen_action=chosen_action,
            confidence=final_confidence,
            urgency=urgency,
            success_probability=success_probability,
            justification=justification,
            memory_influence=memory_influence,
            emotional_gradients=context['emotional_gradients'],
            signal_entropy=context['signal_entropy']
        )
        
        return decision
    
    def _apply_agent_feedback(self, decision: OvermindDecision, agents: List) -> Dict[str, Any]:
        """Apply feedback to agents based on decision"""
        feedback_type = self._determine_feedback_type(decision.chosen_action)
        intensity = decision.confidence * decision.urgency
        
        return self.agent_feedback.apply_overmind_feedback(
            agents[:20], feedback_type, intensity, decision.chosen_action
        )
    
    def _determine_feedback_type(self, action: OvermindActionType) -> str:
        """Determine feedback type based on action"""
        feedback_mapping = {
            OvermindActionType.TRIGGER_COLLECTIVE_MEDITATION: 'mindfulness_boost',
            OvermindActionType.PROMOTE_COOPERATION: 'cooperation_enhancement',
            OvermindActionType.ENHANCE_WISDOM_PROPAGATION: 'wisdom_receptivity',
        }
        return feedback_mapping.get(action, 'general_guidance')
    
    def _update_systems(self, decision: OvermindDecision, context: Dict[str, Any], step: int):
        """Update all subsystems with decision results"""
        
        # Update memory system
        if decision and decision.chosen_action != OvermindActionType.NO_ACTION:
            impact_data = {
                'agents_affected': len(context['agents']) // 4,
                'implementation_fidelity': 0.8,
                'effectiveness': decision.success_probability
            }
            self.memory_attention.add_intervention_memory(decision.__dict__, impact_data)
        
        # Update threshold regulator
        self.threshold_regulator.record_intervention_outcome(
            'intervention_threshold',
            context['colony_metrics'].crisis_level() > 0.6,
            decision.success_probability > 0.7 if decision else False
        )
        
        # Update system state
        self.system_state['last_decision_step'] = step
        self.system_state['total_runtime'] = time.time() - self.system_state['initialization_time']
    
    def _record_decision_metrics(self, decision: OvermindDecision, processing_time: float, step: int):
        """Record comprehensive decision metrics"""
        
        self.performance_metrics['total_decisions'] += 1
        
        if decision and decision.chosen_action != OvermindActionType.NO_ACTION:
            self.performance_metrics['successful_interventions'] += 1
        
        # Update average processing time
        current_avg = self.performance_metrics['average_processing_time']
        total_decisions = self.performance_metrics['total_decisions']
        self.performance_metrics['average_processing_time'] = (
            (current_avg * (total_decisions - 1) + processing_time) / total_decisions
        )
        
        # Record in decision history
        decision_record = {
            'step': step,
            'timestamp': time.time(),
            'decision': decision,
            'processing_time': processing_time
        }
        self.decision_history.append(decision_record)
        
        # Log decision
        if self.logger:
            self.logger.log_decision(decision, step, processing_time)
    
    def _handle_processing_exception(self, exception: Exception, step: int, 
                                   processing_time: float) -> Optional[OvermindDecision]:
        """Handle processing exceptions gracefully"""
        
        error_msg = f"Processing error at step {step}: {exception}"
        logger.error(error_msg)
        
        if self.debug_mode:
            logger.debug(f"Exception trace: {traceback.format_exc()}")
        
        # Record failed intervention
        self.performance_metrics['failed_interventions'] += 1
        
        # Return safe default action in crisis
        return OvermindDecision(
            chosen_action=OvermindActionType.NO_ACTION,
            confidence=0.1,
            urgency=0.0,
            success_probability=0.0,
            justification=f"Emergency fallback due to error: {str(exception)[:100]}"
        )
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'overmind_id': self.overmind_id,
            'system_status': self.system_state['status'],
            'total_decisions': self.performance_metrics['total_decisions'],
            'success_rate': self._calculate_success_rate(),
            'average_processing_time': self.performance_metrics['average_processing_time'],
            'runtime_hours': self.system_state['total_runtime'] / 3600,
            'component_status': self._get_component_status(),
            'recent_decisions': [d['decision'].chosen_action.name 
                               for d in list(self.decision_history)[-5:]]
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate intervention success rate"""
        total = self.performance_metrics['total_decisions']
        if total == 0:
            return 0.0
        
        successful = self.performance_metrics['successful_interventions']
        return successful / total
    
    def _get_component_status(self) -> Dict[str, str]:
        """Get status of all components"""
        status = {}
        
        components = [
            ('wisdom_archive', self.wisdom_archive),
            ('memory_attention', self.memory_attention),
            ('agent_feedback', self.agent_feedback),
            ('ritual_layer', self.ritual_layer),
            ('threshold_regulator', self.threshold_regulator)
        ]
        
        for name, component in components:
            try:
                if hasattr(component, 'get_status'):
                    status[name] = component.get_status()
                else:
                    status[name] = 'active'
            except Exception as e:
                status[name] = f'error: {e}'
        
        return status

# ===== BASIC FALLBACK COMPONENTS =====

class BasicWisdomArchive:
    """Basic fallback wisdom archive"""
    def __init__(self):
        self.insights = deque(maxlen=100)
    
    def store_insight(self, insight: str, context: Dict[str, Any]) -> str:
        insight_id = f"insight_{len(self.insights)}"
        self.insights.append({'id': insight_id, 'text': insight, 'context': context})
        return insight_id
    
    def get_status(self) -> str:
        return f"active ({len(self.insights)} insights)"

class BasicMemoryAttention:
    """Basic fallback memory attention"""
    def __init__(self):
        self.memories = deque(maxlen=50)
    
    def add_intervention_memory(self, decision: Dict, impact: Dict):
        self.memories.append({'decision': decision, 'impact': impact})
    
    def compute_weighted_memory_influence(self, context: Dict) -> Dict[str, float]:
        return {'memory_influence': 0.1, 'confidence_boost': 0.05}
    
    def get_status(self) -> str:
        return f"active ({len(self.memories)} memories)"

class BasicAgentFeedback:
    """Basic fallback agent feedback"""
    def __init__(self):
        self.feedback_count = 0
    
    def apply_overmind_feedback(self, agents: List, feedback_type: str, 
                              intensity: float, action: OvermindActionType) -> Dict[str, Any]:
        affected = 0
        for agent in agents[:10]:  # Limit to 10 agents
            if hasattr(agent, 'mindfulness_level') and feedback_type == 'mindfulness_boost':
                agent.mindfulness_level = min(1.0, agent.mindfulness_level + intensity * 0.1)
                affected += 1
        
        self.feedback_count += 1
        return {'total_agents_affected': affected, 'feedback_applications': []}
    
    def get_status(self) -> str:
        return f"active ({self.feedback_count} applications)"

class BasicRitualLayer:
    """Basic fallback ritual layer"""
    def __init__(self):
        self.ritual_count = 0
        self.active_rituals = {}
    
    def assess_ritual_opportunities(self, agents: List, colony_metrics) -> List:
        # Simple heuristic
        if colony_metrics.collective_mindfulness < 0.5:
            return [RitualType.SYNCHRONIZED_MEDITATION]
        return []
    
    def get_status(self) -> str:
        return f"active ({len(self.active_rituals)} active rituals)"

class BasicThresholdRegulator:
    """Basic fallback threshold regulator"""
    def __init__(self):
        self.thresholds = {
            'intervention_threshold': 0.6,
            'crisis_detection_threshold': 0.7
        }
        self.adjustment_count = 0
    
    def get_threshold(self, threshold_name: str) -> float:
        return self.thresholds.get(threshold_name, 0.5)
    
    def record_intervention_outcome(self, threshold_type: str, 
                                  intervention_triggered: bool, success: bool):
        self.adjustment_count += 1
        # Simple adaptation
        if intervention_triggered and not success:
            self.thresholds[threshold_type] = min(0.9, self.thresholds[threshold_type] + 0.01)
        elif not intervention_triggered and success:
            self.thresholds[threshold_type] = max(0.1, self.thresholds[threshold_type] - 0.01)
    
    def get_status(self) -> str:
        return f"active ({self.adjustment_count} adjustments)"

if __name__ == "__main__":
    print("Core Overmind System Module - Ready for Integration")
    print("Import this module and create environment + agents to run overmind")
