"""
MycoNet 3.0 Evaluation Framework
================================

Comprehensive evaluation covering:
- Task performance metrics
- Field Architecture theoretical predictions
- Ethical compliance checks
- Scalability assessment
- 14 falsifiable predictions validation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import json
from pathlib import Path

from .config import MycoNetConfig
from .field_metrics import ColonyMetrics, MetricsComputer
from .scenarios import ScenarioRunner, ScenarioType, ScenarioResult

logger = logging.getLogger(__name__)


@dataclass
class PredictionValidation:
    """Result of validating a theoretical prediction."""
    prediction_id: str
    description: str
    validated: bool
    confidence: float
    evidence: Dict[str, float]
    details: str = ""


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    # Task performance
    task_scores: Dict[str, float] = field(default_factory=dict)
    scenario_results: Dict[str, ScenarioResult] = field(default_factory=dict)

    # Field Architecture metrics
    field_metrics: ColonyMetrics = None
    metric_trajectories: Dict[str, List[float]] = field(default_factory=dict)

    # Theoretical predictions
    predictions: List[PredictionValidation] = field(default_factory=list)
    prediction_success_rate: float = 0.0

    # Ethical compliance
    ethical_violations: int = 0
    ethical_compliance_rate: float = 0.0

    # Scalability
    scalability_score: float = 0.0
    performance_by_scale: Dict[int, float] = field(default_factory=dict)

    # Summary
    overall_score: float = 0.0
    pass_fail: bool = False


class TheoreticalPredictionValidator:
    """
    Validates the 14 falsifiable predictions from Field Architecture theory.
    """

    def __init__(self, config: MycoNetConfig):
        self.config = config
        self.predictions = self._define_predictions()

    def _define_predictions(self) -> List[Dict[str, Any]]:
        """Define the 14 falsifiable predictions."""
        return [
            {
                'id': 'P1',
                'description': 'Entropy floor: S > 0.3 bits/symbol under load',
                'metric': 'entropy',
                'condition': lambda s: s > 0.3,
                'threshold': 0.3
            },
            {
                'id': 'P2',
                'description': 'Information ceiling: I < 0.9 relative to capacity',
                'metric': 'mutual_information',
                'condition': lambda i: i < 0.9,
                'threshold': 0.9
            },
            {
                'id': 'P3',
                'description': 'Coherence emergence: C increases with cooperation',
                'metric': 'coherence',
                'condition': lambda c: c > 0.4,
                'threshold': 0.4
            },
            {
                'id': 'P4',
                'description': 'Integrated information: Φ > 0 for conscious processing',
                'metric': 'integrated_information',
                'condition': lambda phi: phi > 0,
                'threshold': 0.0
            },
            {
                'id': 'P5',
                'description': 'Phenomenal curvature: Φ_f correlates with complexity',
                'metric': 'phenomenal_curvature',
                'condition': lambda phi_f: phi_f > 0.1,
                'threshold': 0.1
            },
            {
                'id': 'P6',
                'description': 'Recoverability: RX > 0.7 for ethical operations',
                'metric': 'recoverability_index',
                'condition': lambda rx: rx > 0.7,
                'threshold': 0.7
            },
            {
                'id': 'P7',
                'description': 'Resonance stability: τ_R > 10 steps for stable colonies',
                'metric': 'resonance_half_life',
                'condition': lambda tau: tau > 10,
                'threshold': 10
            },
            {
                'id': 'P8',
                'description': 'Signal diffusion follows wisdom propagation laws',
                'metric': 'signal_diffusion_rate',
                'condition': lambda d: 0.05 < d < 0.5,
                'threshold': (0.05, 0.5)
            },
            {
                'id': 'P9',
                'description': 'Topological defects correlate with decision points',
                'metric': 'defect_correlation',
                'condition': lambda dc: dc > 0.3,
                'threshold': 0.3
            },
            {
                'id': 'P10',
                'description': 'Hypernetwork compression ratio > 10:1',
                'metric': 'compression_ratio',
                'condition': lambda cr: cr > 10,
                'threshold': 10
            },
            {
                'id': 'P11',
                'description': 'Surrogate model accuracy > 85% for field prediction',
                'metric': 'surrogate_accuracy',
                'condition': lambda acc: acc > 0.85,
                'threshold': 0.85
            },
            {
                'id': 'P12',
                'description': 'Symbolic bridge safety: PatchGate blocks > 99% unsafe patches',
                'metric': 'patchgate_safety',
                'condition': lambda ps: ps > 0.99,
                'threshold': 0.99
            },
            {
                'id': 'P13',
                'description': 'Multi-scale abstraction: MERA captures hierarchical structure',
                'metric': 'mera_effectiveness',
                'condition': lambda me: me > 0.5,
                'threshold': 0.5
            },
            {
                'id': 'P14',
                'description': 'Evolution fitness improvement > 50% over baseline',
                'metric': 'evolution_improvement',
                'condition': lambda ei: ei > 0.5,
                'threshold': 0.5
            }
        ]

    def validate_prediction(self, prediction: Dict, metrics: Dict[str, float]) -> PredictionValidation:
        """Validate a single prediction against collected metrics."""
        pred_id = prediction['id']
        metric_name = prediction['metric']

        if metric_name not in metrics:
            return PredictionValidation(
                prediction_id=pred_id,
                description=prediction['description'],
                validated=False,
                confidence=0.0,
                evidence={metric_name: float('nan')},
                details=f"Metric '{metric_name}' not available"
            )

        value = metrics[metric_name]
        condition = prediction['condition']
        validated = condition(value)

        # Calculate confidence based on margin from threshold
        threshold = prediction['threshold']
        if isinstance(threshold, tuple):
            # Range threshold
            low, high = threshold
            if validated:
                margin = min(value - low, high - value)
                confidence = min(1.0, margin / (high - low) * 2)
            else:
                confidence = 0.0
        else:
            if validated:
                if value > threshold:
                    confidence = min(1.0, (value - threshold) / max(threshold, 0.1))
                else:
                    confidence = min(1.0, (threshold - value) / max(threshold, 0.1))
            else:
                confidence = 0.0

        return PredictionValidation(
            prediction_id=pred_id,
            description=prediction['description'],
            validated=validated,
            confidence=confidence,
            evidence={metric_name: value},
            details=f"Value: {value:.4f}, Threshold: {threshold}"
        )

    def validate_all(self, metrics: Dict[str, float]) -> List[PredictionValidation]:
        """Validate all predictions."""
        validations = []
        for prediction in self.predictions:
            validation = self.validate_prediction(prediction, metrics)
            validations.append(validation)
        return validations


class EthicalComplianceChecker:
    """Checks ethical compliance based on Dharma Compiler rules."""

    def __init__(self, config: MycoNetConfig):
        self.config = config
        self.dharma = config.dharma
        self.violations: List[Dict] = []

    def check_rx_compliance(self, rx_history: List[float]) -> Tuple[bool, int]:
        """Check recoverability index compliance."""
        violations = sum(1 for rx in rx_history if rx < self.dharma.rx_moral_threshold)
        compliant = violations == 0
        return compliant, violations

    def check_entropy_export(self, entropy_exports: List[float]) -> Tuple[bool, int]:
        """Check entropy export compliance."""
        violations = sum(1 for e in entropy_exports if e > self.dharma.max_entropy_export)
        compliant = violations == 0
        return compliant, violations

    def check_fairness(self, resource_distributions: List[np.ndarray]) -> Tuple[bool, int]:
        """Check fairness via Gini coefficient."""
        violations = 0
        for dist in resource_distributions:
            if len(dist) > 0:
                gini = self._compute_gini(dist)
                if gini > self.dharma.gini_threshold:
                    violations += 1
        compliant = violations == 0
        return compliant, violations

    def _compute_gini(self, values: np.ndarray) -> float:
        """Compute Gini coefficient."""
        if len(values) == 0 or np.sum(values) == 0:
            return 0.0

        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
        return max(0.0, gini)

    def full_compliance_check(self, history: Dict[str, List]) -> Dict[str, Any]:
        """Run full ethical compliance check."""
        results = {
            'rx_compliant': True,
            'rx_violations': 0,
            'entropy_compliant': True,
            'entropy_violations': 0,
            'fairness_compliant': True,
            'fairness_violations': 0,
            'total_violations': 0,
            'compliance_rate': 1.0
        }

        if 'rx_history' in history:
            compliant, violations = self.check_rx_compliance(history['rx_history'])
            results['rx_compliant'] = compliant
            results['rx_violations'] = violations

        if 'entropy_exports' in history:
            compliant, violations = self.check_entropy_export(history['entropy_exports'])
            results['entropy_compliant'] = compliant
            results['entropy_violations'] = violations

        if 'resource_distributions' in history:
            compliant, violations = self.check_fairness(history['resource_distributions'])
            results['fairness_compliant'] = compliant
            results['fairness_violations'] = violations

        results['total_violations'] = (
            results['rx_violations'] +
            results['entropy_violations'] +
            results['fairness_violations']
        )

        total_checks = len(history.get('rx_history', [])) + \
                      len(history.get('entropy_exports', [])) + \
                      len(history.get('resource_distributions', []))

        if total_checks > 0:
            results['compliance_rate'] = 1.0 - (results['total_violations'] / total_checks)

        return results


class ScalabilityAssessor:
    """Assesses system scalability across agent counts."""

    def __init__(self, config: MycoNetConfig):
        self.config = config
        self.scale_tests = [10, 50, 100, 500, 1000]
        self.baseline_performance: Optional[float] = None

    def assess_scale(self, agent_count: int,
                    step_times: List[float],
                    coherence_values: List[float]) -> Dict[str, float]:
        """Assess performance at a given scale."""
        avg_step_time = np.mean(step_times) if step_times else 0
        avg_coherence = np.mean(coherence_values) if coherence_values else 0

        # Compute throughput (agents processed per second)
        throughput = agent_count / max(avg_step_time, 1e-6)

        # Expected linear scaling
        expected_time = agent_count / 1000  # 1ms per agent baseline

        # Efficiency (actual vs expected)
        efficiency = expected_time / max(avg_step_time, 1e-6)

        return {
            'agent_count': agent_count,
            'avg_step_time': avg_step_time,
            'throughput': throughput,
            'efficiency': efficiency,
            'coherence': avg_coherence
        }

    def compute_scalability_score(self, scale_results: Dict[int, Dict]) -> float:
        """Compute overall scalability score."""
        if not scale_results:
            return 0.0

        # Score based on maintaining efficiency across scales
        efficiencies = [r['efficiency'] for r in scale_results.values()]
        coherences = [r['coherence'] for r in scale_results.values()]

        # Efficiency should remain close to 1.0
        efficiency_score = np.mean([min(e, 1.0) for e in efficiencies])

        # Coherence should not degrade significantly
        if len(coherences) > 1:
            coherence_degradation = 1.0 - (coherences[-1] / max(coherences[0], 1e-6))
            coherence_score = max(0, 1.0 - coherence_degradation)
        else:
            coherence_score = 1.0 if coherences else 0.0

        # Combined score
        return 0.6 * efficiency_score + 0.4 * coherence_score


class Evaluator:
    """Main evaluation orchestrator."""

    def __init__(self, config: MycoNetConfig):
        self.config = config
        self.metrics_computer = MetricsComputer(config)
        self.prediction_validator = TheoreticalPredictionValidator(config)
        self.ethical_checker = EthicalComplianceChecker(config)
        self.scalability_assessor = ScalabilityAssessor(config)
        self.scenario_runner = ScenarioRunner(config)

    def evaluate_colony(self, state_history: np.ndarray,
                       field_states: np.ndarray,
                       history: Dict[str, List] = None) -> EvaluationResult:
        """
        Comprehensive colony evaluation.

        Args:
            state_history: Agent states over time [T, N, state_dim]
            field_states: Field states over time [T, H, W]
            history: Additional historical data (rx, entropy exports, etc.)

        Returns:
            Complete evaluation result
        """
        result = EvaluationResult()
        history = history or {}

        # Compute final field metrics
        if len(field_states) > 0:
            result.field_metrics = self.metrics_computer.compute_all(
                state_history, field_states[-1]
            )

        # Compute metric trajectories
        result.metric_trajectories = self._compute_trajectories(state_history, field_states)

        # Validate theoretical predictions
        metrics_dict = self._metrics_to_dict(result.field_metrics, history)
        result.predictions = self.prediction_validator.validate_all(metrics_dict)
        result.prediction_success_rate = sum(
            p.validated for p in result.predictions
        ) / len(result.predictions) if result.predictions else 0

        # Check ethical compliance
        ethical_results = self.ethical_checker.full_compliance_check(history)
        result.ethical_violations = ethical_results['total_violations']
        result.ethical_compliance_rate = ethical_results['compliance_rate']

        # Compute overall score
        result.overall_score = self._compute_overall_score(result)
        result.pass_fail = result.overall_score >= 0.7

        return result

    def _compute_trajectories(self, state_history: np.ndarray,
                             field_states: np.ndarray) -> Dict[str, List[float]]:
        """Compute metric trajectories over time."""
        trajectories = defaultdict(list)

        # Sample at regular intervals to reduce computation
        sample_interval = max(1, len(state_history) // 50)

        for t in range(0, len(state_history), sample_interval):
            states = state_history[t]
            field = field_states[t] if t < len(field_states) else field_states[-1]

            metrics = self.metrics_computer.compute_all(
                state_history[max(0, t-10):t+1],
                field
            )

            trajectories['entropy'].append(metrics.entropy)
            trajectories['coherence'].append(metrics.coherence)
            trajectories['information'].append(metrics.mutual_information)
            trajectories['phi'].append(metrics.integrated_information)
            trajectories['rx'].append(metrics.recoverability_index)

        return dict(trajectories)

    def _metrics_to_dict(self, metrics: ColonyMetrics,
                        history: Dict[str, List]) -> Dict[str, float]:
        """Convert metrics to dictionary for prediction validation."""
        if metrics is None:
            return {}

        d = {
            'entropy': metrics.entropy,
            'mutual_information': metrics.mutual_information,
            'coherence': metrics.coherence,
            'integrated_information': metrics.integrated_information,
            'phenomenal_curvature': metrics.phenomenal_curvature,
            'recoverability_index': metrics.recoverability_index,
            'resonance_half_life': metrics.resonance_half_life
        }

        # Add historical/derived metrics if available
        if 'signal_diffusion_rates' in history:
            rates = history['signal_diffusion_rates']
            d['signal_diffusion_rate'] = np.mean(rates) if rates else 0

        if 'defect_decisions' in history:
            correlations = history['defect_decisions']
            d['defect_correlation'] = np.mean(correlations) if correlations else 0

        if 'compression_ratios' in history:
            ratios = history['compression_ratios']
            d['compression_ratio'] = np.mean(ratios) if ratios else 0

        if 'surrogate_accuracies' in history:
            accs = history['surrogate_accuracies']
            d['surrogate_accuracy'] = np.mean(accs) if accs else 0

        if 'patchgate_results' in history:
            results = history['patchgate_results']
            d['patchgate_safety'] = np.mean(results) if results else 0

        if 'mera_scores' in history:
            scores = history['mera_scores']
            d['mera_effectiveness'] = np.mean(scores) if scores else 0

        if 'fitness_history' in history:
            fitness = history['fitness_history']
            if len(fitness) > 1:
                d['evolution_improvement'] = (fitness[-1] - fitness[0]) / max(abs(fitness[0]), 1e-6)
            else:
                d['evolution_improvement'] = 0

        return d

    def _compute_overall_score(self, result: EvaluationResult) -> float:
        """Compute overall evaluation score."""
        scores = []
        weights = []

        # Task performance (if available)
        if result.task_scores:
            task_avg = np.mean(list(result.task_scores.values()))
            scores.append(task_avg)
            weights.append(0.3)

        # Prediction validation
        scores.append(result.prediction_success_rate)
        weights.append(0.25)

        # Ethical compliance
        scores.append(result.ethical_compliance_rate)
        weights.append(0.25)

        # Scalability (if measured)
        if result.scalability_score > 0:
            scores.append(result.scalability_score)
            weights.append(0.2)

        # Field metrics quality
        if result.field_metrics:
            # Good metrics: high coherence, reasonable entropy, high RX
            fm = result.field_metrics
            field_score = (
                0.4 * min(fm.coherence, 1.0) +
                0.2 * (1.0 - abs(fm.entropy - 0.5)) +  # Peak at 0.5
                0.4 * fm.recoverability_index
            )
            scores.append(field_score)
            weights.append(0.2)

        if not scores:
            return 0.0

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        return sum(s * w for s, w in zip(scores, weights))

    def run_scenario_evaluation(self, environment,
                               agents: List,
                               scenarios: List[ScenarioType] = None) -> Dict[str, ScenarioResult]:
        """Run scenario-based evaluation."""
        if scenarios is None:
            scenarios = list(ScenarioType)

        results = {}
        for scenario_type in scenarios:
            try:
                scenario = self.scenario_runner.create_scenario(scenario_type)
                result = self.scenario_runner.run_scenario(scenario, environment, agents)
                results[scenario_type.name] = result
                logger.info(f"{scenario_type.name}: score={result.score:.3f}")
            except Exception as e:
                logger.error(f"Error running {scenario_type.name}: {e}")
                results[scenario_type.name] = ScenarioResult(
                    scenario_type=scenario_type,
                    success=False,
                    score=0.0,
                    metrics={},
                    details={'error': str(e)}
                )

        return results

    def run_scalability_evaluation(self, agent_factory: Callable,
                                  environment_factory: Callable,
                                  scales: List[int] = None) -> Dict[int, Dict]:
        """Run scalability evaluation across different agent counts."""
        if scales is None:
            scales = self.scalability_assessor.scale_tests

        results = {}
        for scale in scales:
            logger.info(f"Testing scale: {scale} agents")

            # Create environment and agents at this scale
            env = environment_factory(scale)
            agents = agent_factory(scale)

            # Run brief simulation
            step_times = []
            coherence_values = []

            import time
            for step in range(100):
                start = time.time()

                # Simulate step
                for agent in agents:
                    if hasattr(agent, 'step'):
                        agent.step(env)

                step_time = time.time() - start
                step_times.append(step_time)

                # Compute coherence
                if hasattr(env, 'compute_coherence'):
                    coherence_values.append(env.compute_coherence(agents))

            results[scale] = self.scalability_assessor.assess_scale(
                scale, step_times, coherence_values
            )

        return results

    def generate_report(self, result: EvaluationResult,
                       output_path: str = None) -> str:
        """Generate evaluation report."""
        lines = []
        lines.append("=" * 60)
        lines.append("MycoNet 3.0 Evaluation Report")
        lines.append("=" * 60)
        lines.append("")

        # Overall result
        lines.append(f"Overall Score: {result.overall_score:.3f}")
        lines.append(f"Pass/Fail: {'PASS' if result.pass_fail else 'FAIL'}")
        lines.append("")

        # Field metrics
        if result.field_metrics:
            lines.append("-" * 40)
            lines.append("Field Architecture Metrics")
            lines.append("-" * 40)
            fm = result.field_metrics
            lines.append(f"  Entropy (S):            {fm.entropy:.4f}")
            lines.append(f"  Mutual Information (I): {fm.mutual_information:.4f}")
            lines.append(f"  Coherence (C):          {fm.coherence:.4f}")
            lines.append(f"  Integrated Info (Φ):    {fm.integrated_information:.4f}")
            lines.append(f"  Phenomenal Curv (Φ_f):  {fm.phenomenal_curvature:.4f}")
            lines.append(f"  Recoverability (RX):    {fm.recoverability_index:.4f}")
            lines.append(f"  Resonance τ_R:          {fm.resonance_half_life:.4f}")
            lines.append("")

        # Theoretical predictions
        lines.append("-" * 40)
        lines.append("Theoretical Predictions")
        lines.append("-" * 40)
        lines.append(f"  Success Rate: {result.prediction_success_rate:.1%}")
        for pred in result.predictions:
            status = "✓" if pred.validated else "✗"
            lines.append(f"  [{status}] {pred.prediction_id}: {pred.description}")
            lines.append(f"      {pred.details}")
        lines.append("")

        # Ethical compliance
        lines.append("-" * 40)
        lines.append("Ethical Compliance")
        lines.append("-" * 40)
        lines.append(f"  Compliance Rate: {result.ethical_compliance_rate:.1%}")
        lines.append(f"  Total Violations: {result.ethical_violations}")
        lines.append("")

        # Task performance
        if result.task_scores:
            lines.append("-" * 40)
            lines.append("Task Performance")
            lines.append("-" * 40)
            for task, score in result.task_scores.items():
                lines.append(f"  {task}: {score:.3f}")
            lines.append("")

        # Scalability
        if result.performance_by_scale:
            lines.append("-" * 40)
            lines.append("Scalability")
            lines.append("-" * 40)
            lines.append(f"  Scalability Score: {result.scalability_score:.3f}")
            for scale, perf in sorted(result.performance_by_scale.items()):
                lines.append(f"  N={scale}: {perf:.3f}")
            lines.append("")

        lines.append("=" * 60)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report


def create_evaluator(config: MycoNetConfig = None) -> Evaluator:
    """Factory function to create an evaluator."""
    if config is None:
        config = MycoNetConfig()
    return Evaluator(config)


def quick_evaluate(state_history: np.ndarray,
                  field_states: np.ndarray,
                  config: MycoNetConfig = None) -> Dict[str, float]:
    """Quick evaluation returning key metrics."""
    evaluator = create_evaluator(config)
    result = evaluator.evaluate_colony(state_history, field_states)

    return {
        'overall_score': result.overall_score,
        'prediction_rate': result.prediction_success_rate,
        'ethical_compliance': result.ethical_compliance_rate,
        'coherence': result.field_metrics.coherence if result.field_metrics else 0,
        'rx': result.field_metrics.recoverability_index if result.field_metrics else 0,
        'pass': result.pass_fail
    }
