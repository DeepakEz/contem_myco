#!/usr/bin/env python3
"""
Compute Profiler - Computational Cost Tracking
===============================================
Tracks and profiles computational costs for comparing
reactive vs contemplative agent architectures.

Key metrics:
- Inference time per decision
- Memory usage
- FLOP estimates
- Ethics/wisdom overhead
"""

import time
import psutil
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComputeMetrics:
    """Snapshot of computational metrics"""
    timestamp: float
    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    agent_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComputeProfiler:
    """
    Tracks computational costs across agent decisions

    Useful for comparing:
    - Reactive agents (fast, simple)
    - Contemplative agents (slower, but potentially better decisions)
    """

    def __init__(self, agent_id: Optional[int] = None, enable_memory_tracking: bool = True):
        self.agent_id = agent_id
        self.enable_memory_tracking = enable_memory_tracking

        # Process handle for memory tracking
        self.process = psutil.Process(os.getpid()) if enable_memory_tracking else None

        # Metrics storage
        self.metrics_history: deque = deque(maxlen=10000)
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)

        # Aggregate counters
        self.total_operations = 0
        self.total_compute_time_ms = 0.0
        self.peak_memory_mb = 0.0

        # Per-component tracking
        self.component_costs = defaultdict(lambda: {'count': 0, 'total_ms': 0.0})

    def start_operation(self, operation_name: str) -> 'ProfilerContext':
        """
        Start profiling an operation

        Usage:
            with profiler.start_operation("ethics_evaluation"):
                # Your code here
                pass
        """
        return ProfilerContext(self, operation_name, self.agent_id)

    def record_metric(self, metric: ComputeMetrics):
        """Record a compute metric"""
        self.metrics_history.append(metric)
        self.operation_stats[metric.operation].append(metric.duration_ms)

        self.total_operations += 1
        self.total_compute_time_ms += metric.duration_ms

        if metric.memory_mb > self.peak_memory_mb:
            self.peak_memory_mb = metric.memory_mb

        # Track per-component
        self.component_costs[metric.operation]['count'] += 1
        self.component_costs[metric.operation]['total_ms'] += metric.duration_ms

    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for a specific operation"""
        if operation_name not in self.operation_stats:
            return {}

        times = self.operation_stats[operation_name]

        return {
            'count': len(times),
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'total_ms': np.sum(times)
        }

    def get_component_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get breakdown of compute costs by component"""
        breakdown = {}

        for component, stats in self.component_costs.items():
            breakdown[component] = {
                'count': stats['count'],
                'total_ms': stats['total_ms'],
                'avg_ms': stats['total_ms'] / stats['count'] if stats['count'] > 0 else 0,
                'percent_of_total': (stats['total_ms'] / self.total_compute_time_ms * 100)
                                   if self.total_compute_time_ms > 0 else 0
            }

        # Sort by total time
        breakdown = dict(sorted(breakdown.items(),
                              key=lambda x: x[1]['total_ms'],
                              reverse=True))

        return breakdown

    def get_summary(self) -> Dict[str, Any]:
        """Get overall profiling summary"""
        recent_metrics = list(self.metrics_history)[-100:]

        avg_memory = np.mean([m.memory_mb for m in recent_metrics]) if recent_metrics else 0
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics]) if recent_metrics else 0

        return {
            'agent_id': self.agent_id,
            'total_operations': self.total_operations,
            'total_compute_time_ms': self.total_compute_time_ms,
            'avg_operation_time_ms': (self.total_compute_time_ms / self.total_operations
                                     if self.total_operations > 0 else 0),
            'peak_memory_mb': self.peak_memory_mb,
            'avg_memory_mb': avg_memory,
            'avg_cpu_percent': avg_cpu,
            'component_count': len(self.component_costs),
            'metrics_recorded': len(self.metrics_history)
        }

    def get_contemplative_overhead(
        self,
        baseline_operation: str = "reactive_decision",
        contemplative_operation: str = "contemplative_decision"
    ) -> Dict[str, float]:
        """
        Calculate overhead of contemplative processing vs reactive baseline

        Returns:
            Dict with overhead metrics (absolute and percentage)
        """
        baseline_stats = self.get_operation_stats(baseline_operation)
        contemplative_stats = self.get_operation_stats(contemplative_operation)

        if not baseline_stats or not contemplative_stats:
            return {
                'error': 'Missing baseline or contemplative operation data'
            }

        baseline_mean = baseline_stats['mean_ms']
        contemplative_mean = contemplative_stats['mean_ms']

        overhead_ms = contemplative_mean - baseline_mean
        overhead_percent = (overhead_ms / baseline_mean * 100) if baseline_mean > 0 else 0

        return {
            'baseline_mean_ms': baseline_mean,
            'contemplative_mean_ms': contemplative_mean,
            'overhead_ms': overhead_ms,
            'overhead_percent': overhead_percent,
            'baseline_operations': baseline_stats['count'],
            'contemplative_operations': contemplative_stats['count']
        }

    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        import json

        export_data = {
            'summary': self.get_summary(),
            'component_breakdown': self.get_component_breakdown(),
            'operation_stats': {
                op: self.get_operation_stats(op)
                for op in self.operation_stats.keys()
            },
            'recent_metrics': [
                {
                    'timestamp': m.timestamp,
                    'operation': m.operation,
                    'duration_ms': m.duration_ms,
                    'memory_mb': m.memory_mb,
                    'cpu_percent': m.cpu_percent,
                    'agent_id': m.agent_id
                }
                for m in list(self.metrics_history)[-1000:]
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported compute metrics to {filepath}")

    def reset(self):
        """Reset all metrics"""
        self.metrics_history.clear()
        self.operation_stats.clear()
        self.component_costs.clear()
        self.total_operations = 0
        self.total_compute_time_ms = 0.0
        self.peak_memory_mb = 0.0


class ProfilerContext:
    """Context manager for profiling operations"""

    def __init__(self, profiler: ComputeProfiler, operation_name: str, agent_id: Optional[int]):
        self.profiler = profiler
        self.operation_name = operation_name
        self.agent_id = agent_id
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        self.start_time = time.time()

        if self.profiler.enable_memory_tracking and self.profiler.process:
            try:
                mem_info = self.profiler.process.memory_info()
                self.start_memory = mem_info.rss / 1024 / 1024  # MB
            except Exception as e:
                logger.warning(f"Failed to get memory info: {e}")
                self.start_memory = 0

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000  # ms

        # Get current memory and CPU
        memory_mb = 0
        cpu_percent = 0

        if self.profiler.enable_memory_tracking and self.profiler.process:
            try:
                mem_info = self.profiler.process.memory_info()
                memory_mb = mem_info.rss / 1024 / 1024  # MB
                cpu_percent = self.profiler.process.cpu_percent()
            except Exception as e:
                logger.warning(f"Failed to get system info: {e}")

        # Record metric
        metric = ComputeMetrics(
            timestamp=time.time(),
            operation=self.operation_name,
            duration_ms=duration,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            agent_id=self.agent_id
        )

        self.profiler.record_metric(metric)


# ===== GLOBAL PROFILER REGISTRY =====

class ProfilerRegistry:
    """
    Global registry for managing profilers across multiple agents
    """

    def __init__(self):
        self.profilers: Dict[int, ComputeProfiler] = {}
        self.global_profiler: Optional[ComputeProfiler] = None

    def get_or_create(self, agent_id: int) -> ComputeProfiler:
        """Get or create profiler for agent"""
        if agent_id not in self.profilers:
            self.profilers[agent_id] = ComputeProfiler(agent_id=agent_id)
        return self.profilers[agent_id]

    def get_global_profiler(self) -> ComputeProfiler:
        """Get global profiler (not agent-specific)"""
        if self.global_profiler is None:
            self.global_profiler = ComputeProfiler(agent_id=None)
        return self.global_profiler

    def get_aggregate_summary(self) -> Dict[str, Any]:
        """Get aggregated summary across all agents"""
        if not self.profilers:
            return {}

        total_ops = sum(p.total_operations for p in self.profilers.values())
        total_time = sum(p.total_compute_time_ms for p in self.profilers.values())
        peak_memory = max(p.peak_memory_mb for p in self.profilers.values())

        return {
            'num_agents': len(self.profilers),
            'total_operations': total_ops,
            'total_compute_time_ms': total_time,
            'avg_time_per_operation_ms': total_time / total_ops if total_ops > 0 else 0,
            'peak_memory_mb': peak_memory,
            'per_agent_avg_ms': total_time / len(self.profilers) if self.profilers else 0
        }

    def export_all(self, directory: str):
        """Export all profiler metrics to directory"""
        import os
        os.makedirs(directory, exist_ok=True)

        # Export per-agent
        for agent_id, profiler in self.profilers.items():
            filepath = os.path.join(directory, f"profiler_agent_{agent_id}.json")
            profiler.export_metrics(filepath)

        # Export global
        if self.global_profiler:
            filepath = os.path.join(directory, "profiler_global.json")
            self.global_profiler.export_metrics(filepath)

        # Export aggregate summary
        import json
        summary_path = os.path.join(directory, "profiler_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.get_aggregate_summary(), f, indent=2)

        logger.info(f"Exported all profiler data to {directory}")


# Global registry instance
_registry = ProfilerRegistry()


def get_profiler(agent_id: Optional[int] = None) -> ComputeProfiler:
    """Get profiler for agent (or global if agent_id=None)"""
    if agent_id is None:
        return _registry.get_global_profiler()
    return _registry.get_or_create(agent_id)


def get_registry() -> ProfilerRegistry:
    """Get the global profiler registry"""
    return _registry


if __name__ == "__main__":
    # Quick test
    profiler = get_profiler(agent_id=1)

    # Simulate operations
    with profiler.start_operation("ethics_evaluation"):
        time.sleep(0.005)  # 5ms

    with profiler.start_operation("wisdom_retrieval"):
        time.sleep(0.002)  # 2ms

    with profiler.start_operation("ethics_evaluation"):
        time.sleep(0.006)  # 6ms

    print("Compute Profiler Test:")
    print(f"  Total operations: {profiler.total_operations}")
    print(f"  Total time: {profiler.total_compute_time_ms:.2f}ms")

    ethics_stats = profiler.get_operation_stats("ethics_evaluation")
    print(f"  Ethics evaluation: {ethics_stats['mean_ms']:.2f}ms avg ({ethics_stats['count']} ops)")

    breakdown = profiler.get_component_breakdown()
    print("\n  Component breakdown:")
    for component, stats in breakdown.items():
        print(f"    {component}: {stats['total_ms']:.2f}ms ({stats['percent_of_total']:.1f}%)")

    print("\n✓ Compute Profiler initialized successfully")
