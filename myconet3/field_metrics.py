"""
MycoNet 3.0 Field Architecture Metrics Module
=============================================

Implements all Field Architecture cognitive metrics:
- Entropy (Ŝ): S(Ω) = -Σ p(x)log p(x)
- Information (Î): I(A;B) = H(A) + H(B) - H(A,B)
- Coherence (Ĉ): R = |1/N Σ e^(iθ_j)| (Kuramoto order parameter)
- Phenomenal Curvature (Φ_f): Second-order differential of prediction error
- Recoverability Index (RX): RX = 1 - (D_∞ - D_t0)/(D_max - D_t0)
- Resonance Half-life (τ_R): Time constant for coherence decay
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from collections import deque
import logging

from .config import MetricsConfig

logger = logging.getLogger(__name__)


@dataclass
class ColonyMetrics:
    """Container for all Field Architecture metrics."""
    # Core Field Architecture metrics
    entropy: float = 0.0  # Ŝ
    information: float = 0.0  # Î (mutual information)
    coherence: float = 0.0  # Ĉ (Kuramoto order parameter)
    integrated_information: float = 0.0  # Φ (IIT approximation)
    phenomenal_curvature: float = 0.0  # Φ_f (prediction error curvature)
    recoverability: float = 1.0  # RX
    resonance_halflife: float = float('inf')  # τ_R

    # Derived metrics
    crisis_level: float = 0.0  # Composite stress indicator
    stability_index: float = 1.0  # Overall system stability
    emergence_score: float = 0.0  # Novel pattern emergence

    # Colony state metrics
    total_population: int = 0
    average_energy: float = 0.0
    average_health: float = 0.0
    collective_mindfulness: float = 0.0
    cooperation_rate: float = 0.0
    conflict_rate: float = 0.0
    average_wisdom: float = 0.0
    sustainability_index: float = 0.0

    # Metadata
    timestamp: int = 0

    # Aliases for compatibility
    @property
    def mutual_information(self) -> float:
        """Alias for information."""
        return self.information

    @property
    def recoverability_index(self) -> float:
        """Alias for recoverability."""
        return self.recoverability

    @property
    def resonance_half_life(self) -> float:
        """Alias for resonance_halflife."""
        return self.resonance_halflife

    def overall_wellbeing(self) -> float:
        """Compute overall wellbeing score."""
        return (
                0.2 * self.average_energy +
                0.2 * self.average_health +
                0.2 * self.collective_mindfulness +
                0.2 * self.cooperation_rate +
                0.2 * (1.0 - self.crisis_level)
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'entropy': self.entropy,
            'information': self.information,
            'coherence': self.coherence,
            'integrated_information': self.integrated_information,
            'phenomenal_curvature': self.phenomenal_curvature,
            'recoverability': self.recoverability,
            'resonance_halflife': self.resonance_halflife,
            'crisis_level': self.crisis_level,
            'stability_index': self.stability_index,
            'total_population': self.total_population,
            'overall_wellbeing': self.overall_wellbeing()
        }


class MetricsComputer:
    """
    Computes all Field Architecture metrics from system state.
    """

    def __init__(self, config: MetricsConfig):
        self.config = config

        # History buffers for temporal metrics
        self.coherence_history: deque = deque(maxlen=1000)
        self.entropy_history: deque = deque(maxlen=1000)
        self.damage_log: deque = deque(maxlen=500)
        self.state_history: deque = deque(maxlen=500)

        # Baseline state for RX computation
        self.baseline_state: Optional[ColonyMetrics] = None
        self.baseline_window: List[ColonyMetrics] = []

        # Perturbation tracking
        self.perturbation_start: Optional[int] = None
        self.max_damage: float = 0.0

    def compute_all_metrics(self,
                            agent_states: List[Dict[str, Any]],
                            signal_grid: Optional[Any] = None,
                            field_state: Optional[Any] = None,
                            time_step: int = 0) -> ColonyMetrics:
        """
        Compute all Field Architecture metrics from current system state.
        """
        metrics = ColonyMetrics(timestamp=time_step)

        # Basic colony statistics
        metrics = self._compute_colony_stats(metrics, agent_states)

        # Core Field Architecture metrics
        metrics.entropy = self._compute_entropy(agent_states, signal_grid)
        metrics.information = self._compute_mutual_information(agent_states)
        metrics.coherence = self._compute_coherence(agent_states, field_state)
        metrics.integrated_information = self._compute_integrated_information(agent_states)
        metrics.phenomenal_curvature = self._compute_phenomenal_curvature(agent_states)

        # Temporal metrics
        metrics.recoverability = self._compute_recoverability(metrics)
        metrics.resonance_halflife = self._measure_resonance_halflife()

        # Derived metrics
        metrics.crisis_level = self._compute_crisis_level(metrics)
        metrics.stability_index = self._compute_stability_index(metrics)

        # Update history
        self.coherence_history.append(metrics.coherence)
        self.entropy_history.append(metrics.entropy)
        self.state_history.append(metrics)

        # Update baseline if needed
        self._update_baseline(metrics)

        return metrics

    def compute_all(self, state_history: np.ndarray, field_state: np.ndarray) -> ColonyMetrics:
        """
        Simplified interface for computing metrics from numpy arrays.

        Args:
            state_history: Array of shape [T, N, state_dim] or [N, state_dim]
            field_state: Array of shape [H, W] for the UPRT field

        Returns:
            ColonyMetrics with all computed values
        """
        # Handle different input shapes
        if state_history.ndim == 3:
            # [T, N, state_dim] - use latest timestep
            current_states = state_history[-1]
        else:
            current_states = state_history

        # Convert numpy arrays to list of dicts for compatibility
        agent_states = []
        for i, state in enumerate(current_states):
            agent_state = {
                'id': i,
                'energy': float(state[0]) if len(state) > 0 else 0.5,
                'health': float(state[1]) if len(state) > 1 else 1.0,
                'mindfulness': float(state[2]) if len(state) > 2 else 0.5,
                'position': (float(state[3]), float(state[4])) if len(state) > 4 else (0.0, 0.0),
                'alive': True,
                'full_state': state
            }
            agent_states.append(agent_state)

        return self.compute_all_metrics(
            agent_states=agent_states,
            signal_grid=None,
            field_state=field_state,
            time_step=len(self.state_history)
        )

    def _compute_colony_stats(self, metrics: ColonyMetrics,
                              agent_states: List[Dict[str, Any]]) -> ColonyMetrics:
        """Compute basic colony statistics."""
        if not agent_states:
            return metrics

        metrics.total_population = len(agent_states)

        # Average statistics
        energies = [a.get('energy', 0.5) for a in agent_states]
        healths = [a.get('health', 0.5) for a in agent_states]
        mindfulness = [a.get('mindfulness_level', 0.5) for a in agent_states]
        wisdom = [a.get('wisdom_accumulated', 0) for a in agent_states]

        metrics.average_energy = float(np.mean(energies))
        metrics.average_health = float(np.mean(healths))
        metrics.collective_mindfulness = float(np.mean(mindfulness))
        metrics.average_wisdom = float(np.mean(wisdom)) if wisdom else 0.0

        # Cooperation and conflict rates
        cooperating = sum(1 for a in agent_states if a.get('is_cooperating', False))
        conflicting = sum(1 for a in agent_states if a.get('in_conflict', False))

        metrics.cooperation_rate = cooperating / max(len(agent_states), 1)
        metrics.conflict_rate = conflicting / max(len(agent_states), 1)

        # Sustainability index
        resources = [a.get('resources_consumed', 0) for a in agent_states]
        production = [a.get('resources_produced', 0) for a in agent_states]
        total_consumed = sum(resources)
        total_produced = sum(production)

        if total_consumed > 0:
            metrics.sustainability_index = min(1.0, total_produced / total_consumed)
        else:
            metrics.sustainability_index = 1.0

        return metrics

    def _compute_entropy(self, agent_states: List[Dict[str, Any]],
                         signal_grid: Optional[Any] = None) -> float:
        """
        Compute multiscale sample entropy.

        S(Ω) = -Σ p(x)log p(x)

        Combines:
        - Agent state entropy
        - Signal distribution entropy
        - Position entropy
        """
        if not agent_states:
            return 0.0

        entropies = []

        # Agent energy distribution entropy
        energies = [a.get('energy', 0.5) for a in agent_states]
        energy_entropy = self._compute_distribution_entropy(energies)
        entropies.append(energy_entropy)

        # Agent position entropy
        positions = [(a.get('x', 0), a.get('y', 0)) for a in agent_states]
        position_entropy = self._compute_spatial_entropy(positions)
        entropies.append(position_entropy)

        # Agent action diversity entropy
        actions = [a.get('last_action', 0) for a in agent_states]
        action_entropy = self._compute_distribution_entropy(actions)
        entropies.append(action_entropy)

        # Signal grid entropy if available
        if signal_grid is not None:
            signal_entropy = self._compute_signal_entropy(signal_grid)
            entropies.append(signal_entropy)

        # Multiscale combination
        return float(np.mean(entropies))

    def _compute_distribution_entropy(self, values: List[float], bins: int = None) -> float:
        """Compute Shannon entropy of a distribution."""
        if not values:
            return 0.0

        bins = bins or self.config.entropy_bins

        # Create histogram
        hist, _ = np.histogram(values, bins=bins, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / np.sum(hist)  # Normalize

        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        # Normalize by max entropy
        max_entropy = np.log2(bins)
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _compute_spatial_entropy(self, positions: List[Tuple[float, float]]) -> float:
        """Compute entropy of spatial distribution."""
        if len(positions) < 2:
            return 0.0

        # Create 2D histogram
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        bins = int(np.sqrt(len(positions)))
        bins = max(bins, 5)

        hist, _, _ = np.histogram2d(xs, ys, bins=bins, density=True)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)

        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        max_entropy = np.log2(bins * bins)

        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _compute_signal_entropy(self, signal_grid: Any) -> float:
        """Compute entropy of wisdom signal distribution."""
        if not hasattr(signal_grid, 'layers'):
            return 0.0

        all_values = []
        for layer in signal_grid.layers.values():
            all_values.extend(layer.flatten())

        return self._compute_distribution_entropy(all_values)

    def _compute_mutual_information(self, agent_states: List[Dict[str, Any]]) -> float:
        """
        Compute mutual information between agent groups.

        I(A;B) = H(A) + H(B) - H(A,B)

        Uses binning for continuous states.
        """
        if len(agent_states) < 4:
            return 0.0

        # Split agents into two groups
        mid = len(agent_states) // 2
        group_a = agent_states[:mid]
        group_b = agent_states[mid:]

        # Get state vectors
        def get_state_vector(agent):
            return [
                agent.get('energy', 0.5),
                agent.get('health', 0.5),
                agent.get('mindfulness_level', 0.5)
            ]

        states_a = [get_state_vector(a) for a in group_a]
        states_b = [get_state_vector(a) for a in group_b]

        # Discretize states
        bins = 10

        def discretize(states):
            arr = np.array(states)
            discretized = np.floor(arr * bins).astype(int)
            discretized = np.clip(discretized, 0, bins - 1)
            # Create single value from multi-dimensional state
            return [sum(d[i] * (bins ** i) for i in range(len(d))) for d in discretized]

        discrete_a = discretize(states_a)
        discrete_b = discretize(states_b)

        # Compute entropies
        H_A = self._compute_distribution_entropy(discrete_a, bins=bins ** 3)
        H_B = self._compute_distribution_entropy(discrete_b, bins=bins ** 3)

        # Joint entropy (simplified approximation)
        joint = [(a, b) for a, b in zip(discrete_a, discrete_b)]
        joint_values = [a * 1000 + b for a, b in joint]
        H_AB = self._compute_distribution_entropy(joint_values, bins=bins ** 3)

        # Mutual information
        mi = max(0, H_A + H_B - H_AB)

        return float(mi)

    def _compute_coherence(self, agent_states: List[Dict[str, Any]],
                           field_state: Optional[Any] = None) -> float:
        """
        Compute Kuramoto order parameter for phase synchronization.

        Ĉ = R = |1/N Σ e^(iθ_j)|

        Uses agent decision cycles or field oscillations.
        """
        if not agent_states:
            return 0.0

        phases = []

        # Extract phases from agent states
        for agent in agent_states:
            # Use decision timing as phase
            decision_time = agent.get('last_decision_time', 0)
            period = agent.get('decision_period', 10)
            if period > 0:
                phase = 2 * np.pi * (decision_time % period) / period
            else:
                phase = np.random.uniform(0, 2 * np.pi)
            phases.append(phase)

        # Alternatively, use field phases if available
        if field_state is not None and hasattr(field_state, 'phase_map'):
            # Sample field phases at agent locations
            for i, agent in enumerate(agent_states):
                x = int(agent.get('x', 0)) % field_state.phase_map.shape[1]
                y = int(agent.get('y', 0)) % field_state.phase_map.shape[0]
                phases[i] = field_state.phase_map[y, x]

        # Kuramoto order parameter
        complex_phases = np.exp(1j * np.array(phases))
        R = np.abs(np.mean(complex_phases))

        return float(R)

    def _compute_integrated_information(self, agent_states: List[Dict[str, Any]]) -> float:
        """
        Approximate IIT's Φ using partition methods.

        Measure information that can't be explained by independent parts.
        This is a simplified approximation of the full IIT computation.
        """
        if len(agent_states) < 2:
            return 0.0

        # Get state vectors
        def get_state_vector(agent):
            return np.array([
                agent.get('energy', 0.5),
                agent.get('health', 0.5),
                agent.get('mindfulness_level', 0.5),
                agent.get('x', 0) / 64.0,
                agent.get('y', 0) / 64.0
            ])

        states = np.array([get_state_vector(a) for a in agent_states])

        # Compute whole system entropy
        H_whole = self._compute_matrix_entropy(states)

        # Compute partition entropies (minimum information partition)
        min_partition_info = float('inf')

        num_samples = min(self.config.phi_partition_samples, 2 ** len(agent_states))

        for _ in range(num_samples):
            # Random partition
            mid = np.random.randint(1, len(agent_states))
            partition_a = states[:mid]
            partition_b = states[mid:]

            H_A = self._compute_matrix_entropy(partition_a)
            H_B = self._compute_matrix_entropy(partition_b)

            partition_info = H_A + H_B
            min_partition_info = min(min_partition_info, partition_info)

        # Φ = H_whole - min(partition entropies)
        phi = max(0, H_whole - min_partition_info)

        return float(phi)

    def _compute_matrix_entropy(self, states: np.ndarray) -> float:
        """Compute entropy of state matrix using eigenvalue method."""
        if len(states) < 2:
            return 0.0

        # Compute covariance matrix
        cov = np.cov(states.T) + np.eye(states.shape[1]) * 1e-6

        # Eigenvalue entropy
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        eigenvalues = eigenvalues / np.sum(eigenvalues)

        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))

        return float(entropy)

    def _compute_phenomenal_curvature(self, agent_states: List[Dict[str, Any]]) -> float:
        """
        Compute global Φ_f from aggregated prediction error curvature.

        Φ_f = ||∂²PE/∂θ²||

        Approximates the Hessian of prediction error.
        """
        if len(self.state_history) < 3:
            return 0.0

        # Get prediction errors from agents
        prediction_errors = []
        for agent in agent_states:
            pe = agent.get('prediction_error', 0.0)
            prediction_errors.append(pe)

        if not prediction_errors:
            return 0.0

        # Current mean prediction error
        current_pe = np.mean(prediction_errors)

        # Historical prediction errors
        if len(self.state_history) >= 2:
            prev_states = list(self.state_history)[-3:]
            historical_pes = []

            for state in prev_states:
                # Approximate PE from entropy and coherence change
                pe_approx = abs(state.entropy - current_pe) if hasattr(state, 'entropy') else 0
                historical_pes.append(pe_approx)

            if len(historical_pes) >= 3:
                # Second derivative approximation
                d2_pe = historical_pes[2] - 2 * historical_pes[1] + historical_pes[0]
                return float(abs(d2_pe))

        return 0.0

    def _compute_recoverability(self, current_metrics: ColonyMetrics) -> float:
        """
        Compute Recoverability Index.

        RX = 1 - (D_∞ - D_t0)/(D_max - D_t0)

        Measures ability to recover from perturbations.
        """
        if self.baseline_state is None:
            return 1.0

        # Compute current damage (deviation from baseline)
        D_current = self._compute_damage(current_metrics, self.baseline_state)

        # Update max damage
        self.max_damage = max(self.max_damage, D_current)

        if self.max_damage <= 0:
            return 1.0

        # RX formula
        D_baseline = 0.0  # D_t0 = 0 by definition
        rx = 1.0 - (D_current - D_baseline) / (self.max_damage - D_baseline + 1e-6)

        return float(np.clip(rx, 0.0, 1.0))

    def _compute_damage(self, current: ColonyMetrics, baseline: ColonyMetrics) -> float:
        """Compute damage as weighted distance from baseline."""
        damage = 0.0

        # Population loss
        if baseline.total_population > 0:
            pop_loss = max(0, baseline.total_population - current.total_population)
            damage += pop_loss / baseline.total_population * 0.3

        # Energy/health degradation
        energy_loss = max(0, baseline.average_energy - current.average_energy)
        health_loss = max(0, baseline.average_health - current.average_health)
        damage += (energy_loss + health_loss) * 0.2

        # Coherence loss
        coherence_loss = max(0, baseline.coherence - current.coherence)
        damage += coherence_loss * 0.3

        # Crisis increase
        crisis_increase = max(0, current.crisis_level - baseline.crisis_level)
        damage += crisis_increase * 0.2

        return damage

    def _measure_resonance_halflife(self) -> float:
        """
        Measure resonance half-life from coherence decay.

        Fits exponential decay: C(t) = C_0 exp(-t/τ_R)
        Returns τ_R.
        """
        if len(self.coherence_history) < self.config.tau_min_samples:
            return float('inf')

        coherence_series = np.array(list(self.coherence_history)[-self.config.tau_fitting_window:])

        # Find perturbation (significant drop in coherence)
        if len(coherence_series) < 5:
            return float('inf')

        # Look for decay pattern
        diffs = np.diff(coherence_series)
        if np.all(diffs >= 0):  # No decay
            return float('inf')

        # Fit exponential decay
        try:
            t = np.arange(len(coherence_series))
            C0 = coherence_series[0]

            if C0 <= 0:
                return float('inf')

            # Log-linear fit: log(C/C0) = -t/τ
            log_ratio = np.log(np.maximum(coherence_series / C0, 1e-6))
            slope, _, _, _, _ = stats.linregress(t, log_ratio)

            if slope >= 0:
                return float('inf')

            tau = -1.0 / slope
            return float(np.clip(tau, 1.0, 1000.0))

        except Exception:
            return float('inf')

    def _compute_crisis_level(self, metrics: ColonyMetrics) -> float:
        """Compute composite crisis level indicator."""
        crisis = 0.0

        # Low resources
        if metrics.average_energy < 0.3:
            crisis += 0.3 * (0.3 - metrics.average_energy) / 0.3

        # Low health
        if metrics.average_health < 0.3:
            crisis += 0.3 * (0.3 - metrics.average_health) / 0.3

        # High conflict
        crisis += 0.2 * metrics.conflict_rate

        # Low coherence (fragmentation)
        if metrics.coherence < 0.3:
            crisis += 0.2 * (0.3 - metrics.coherence) / 0.3

        return float(np.clip(crisis, 0.0, 1.0))

    def _compute_stability_index(self, metrics: ColonyMetrics) -> float:
        """Compute overall system stability."""
        # Combine recoverability and coherence
        stability = (
                0.4 * metrics.recoverability +
                0.3 * metrics.coherence +
                0.2 * (1.0 - metrics.crisis_level) +
                0.1 * metrics.sustainability_index
        )

        return float(np.clip(stability, 0.0, 1.0))

    def _update_baseline(self, metrics: ColonyMetrics):
        """Update baseline state for RX computation."""
        self.baseline_window.append(metrics)

        if len(self.baseline_window) > self.config.rx_baseline_window:
            self.baseline_window.pop(0)

        if len(self.baseline_window) >= 10:
            # Compute average baseline
            avg_metrics = ColonyMetrics()
            avg_metrics.total_population = int(np.mean([m.total_population for m in self.baseline_window]))
            avg_metrics.average_energy = np.mean([m.average_energy for m in self.baseline_window])
            avg_metrics.average_health = np.mean([m.average_health for m in self.baseline_window])
            avg_metrics.coherence = np.mean([m.coherence for m in self.baseline_window])
            avg_metrics.crisis_level = np.mean([m.crisis_level for m in self.baseline_window])

            self.baseline_state = avg_metrics

    def detect_perturbation(self, current: ColonyMetrics, threshold: float = None) -> bool:
        """Detect if a perturbation has occurred."""
        if threshold is None:
            threshold = self.config.rx_perturbation_threshold

        if self.baseline_state is None:
            return False

        damage = self._compute_damage(current, self.baseline_state)
        return damage > threshold

    def reset(self):
        """Reset all history and baselines."""
        self.coherence_history.clear()
        self.entropy_history.clear()
        self.damage_log.clear()
        self.state_history.clear()
        self.baseline_state = None
        self.baseline_window = []
        self.perturbation_start = None
        self.max_damage = 0.0


# Utility functions for metric computation

def compute_entropy(signal_grid: Any, agent_states: List[Dict[str, Any]]) -> float:
    """
    Multiscale sample entropy.
    Compute Shannon entropy of signal distributions + agent state entropy.
    """
    computer = MetricsComputer(MetricsConfig())
    return computer._compute_entropy(agent_states, signal_grid)


def compute_mutual_information(agent_states: List[Dict[str, Any]]) -> float:
    """
    I(A;B) between different agent groups or agent-environment.
    Use binning or KDE for continuous states.
    """
    computer = MetricsComputer(MetricsConfig())
    return computer._compute_mutual_information(agent_states)


def compute_coherence(agent_states: List[Dict[str, Any]], field_phase_map: Optional[np.ndarray] = None) -> float:
    """
    Kuramoto order parameter: R = |1/N Σ exp(iθ_j)|
    Extract phases from agent decision cycles or field oscillations.
    """
    computer = MetricsComputer(MetricsConfig())

    class MockFieldState:
        def __init__(self, phase_map):
            self.phase_map = phase_map

    field_state = MockFieldState(field_phase_map) if field_phase_map is not None else None
    return computer._compute_coherence(agent_states, field_state)


def compute_integrated_information(agent_states: List[Dict[str, Any]]) -> float:
    """
    Approximate IIT's Φ using partition methods.
    Measure information that can't be explained by independent parts.
    """
    computer = MetricsComputer(MetricsConfig())
    return computer._compute_integrated_information(agent_states)


def compute_phenomenal_curvature(prediction_errors: List[float], model_params: Optional[Any] = None) -> float:
    """
    Global Φ_f from aggregated prediction error curvature.
    ||∂²PE/∂θ²|| using Hessian approximations.
    """
    if len(prediction_errors) < 3:
        return 0.0

    # Second derivative of prediction error sequence
    pe = np.array(prediction_errors)
    d2_pe = np.diff(pe, n=2)

    return float(np.mean(np.abs(d2_pe)))


def compute_recoverability(damage_log: List[float], current_state: float, baseline_state: float) -> float:
    """
    RX = 1 - (persistent_damage / max_damage)
    Track irreversible state changes.
    """
    if not damage_log:
        return 1.0

    max_damage = max(damage_log)
    current_damage = abs(current_state - baseline_state)

    if max_damage <= 0:
        return 1.0

    rx = 1.0 - current_damage / max_damage
    return float(np.clip(rx, 0.0, 1.0))


def measure_resonance_halflife(coherence_timeseries: List[float]) -> float:
    """
    Fit exponential decay C(t) = C_0 exp(-t/τ_R) after perturbation.
    Return τ_R.
    """
    if len(coherence_timeseries) < 10:
        return float('inf')

    try:
        t = np.arange(len(coherence_timeseries))
        C = np.array(coherence_timeseries)
        C0 = C[0]

        if C0 <= 0:
            return float('inf')

        log_ratio = np.log(np.maximum(C / C0, 1e-6))
        slope, _, r_value, _, _ = stats.linregress(t, log_ratio)

        if slope >= 0 or abs(r_value) < 0.5:
            return float('inf')

        tau = -1.0 / slope
        return float(np.clip(tau, 1.0, 1000.0))

    except Exception:
        return float('inf')
