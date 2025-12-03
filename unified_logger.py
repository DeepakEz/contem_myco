#!/usr/bin/env python3
"""
Unified Logger - Consolidated Experiment Logging
=================================================
Centralized logging system for all simulation data:
- Agent decisions and actions
- Environment state
- Ethical assessments
- Wisdom generation
- Compute metrics
- Experiment results
"""

import json
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


@dataclass
class LogEntry:
    """Generic log entry"""
    timestamp: float
    step: int
    category: str  # 'agent', 'environment', 'ethics', 'wisdom', 'compute', 'result'
    data: Dict[str, Any]


class UnifiedLogger:
    """
    Centralized logging system for experiments

    Features:
    - Multi-format output (JSON, CSV)
    - Structured logging by category
    - Real-time and batch export
    - Metric aggregation
    - Experiment metadata tracking
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "results",
        enable_console: bool = True,
        enable_file: bool = True
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.enable_console = enable_console
        self.enable_file = enable_file

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Log storage
        self.log_entries: List[LogEntry] = []
        self.categorized_logs: Dict[str, List[LogEntry]] = defaultdict(list)

        # Metric aggregation
        self.metrics = defaultdict(list)

        # Experiment metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': time.time(),
            'end_time': None,
            'total_steps': 0,
            'config': {}
        }

        # Setup standard logging
        self._setup_standard_logging()

    def _setup_standard_logging(self):
        """Setup standard Python logging"""
        self.logger = logging.getLogger(f"unified_logger.{self.experiment_name}")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers = []

        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if self.enable_file:
            log_file = self.output_dir / f"{self.experiment_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log(
        self,
        step: int,
        category: str,
        data: Dict[str, Any],
        console_msg: Optional[str] = None
    ):
        """
        Log an entry

        Args:
            step: Current simulation step
            category: Log category
            data: Data to log
            console_msg: Optional console message
        """
        entry = LogEntry(
            timestamp=time.time(),
            step=step,
            category=category,
            data=data
        )

        self.log_entries.append(entry)
        self.categorized_logs[category].append(entry)

        # Console output
        if console_msg and self.enable_console:
            self.logger.info(f"[{category}] {console_msg}")

    def log_agent_action(
        self,
        step: int,
        agent_id: int,
        action: str,
        ethical_score: float,
        mindfulness: float,
        **kwargs
    ):
        """Log agent action with ethical and mindfulness info"""
        data = {
            'agent_id': agent_id,
            'action': action,
            'ethical_score': ethical_score,
            'mindfulness': mindfulness,
            **kwargs
        }

        self.log(
            step=step,
            category='agent',
            data=data,
            console_msg=f"Agent {agent_id}: {action} (ethics={ethical_score:.2f}, mind={mindfulness:.2f})"
        )

    def log_environment_state(
        self,
        step: int,
        state_summary: Dict[str, Any]
    ):
        """Log environment state"""
        self.log(
            step=step,
            category='environment',
            data=state_summary
        )

    def log_ethical_assessment(
        self,
        step: int,
        agent_id: int,
        assessment: Dict[str, Any]
    ):
        """Log ethical assessment"""
        data = {
            'agent_id': agent_id,
            **assessment
        }

        self.log(
            step=step,
            category='ethics',
            data=data
        )

    def log_wisdom_insight(
        self,
        step: int,
        agent_id: int,
        wisdom_type: str,
        intensity: float,
        **kwargs
    ):
        """Log wisdom insight generation"""
        data = {
            'agent_id': agent_id,
            'wisdom_type': wisdom_type,
            'intensity': intensity,
            **kwargs
        }

        self.log(
            step=step,
            category='wisdom',
            data=data,
            console_msg=f"Agent {agent_id} generated {wisdom_type} (intensity={intensity:.2f})"
        )

    def log_compute_metrics(
        self,
        step: int,
        operation: str,
        duration_ms: float,
        **kwargs
    ):
        """Log computational metrics"""
        data = {
            'operation': operation,
            'duration_ms': duration_ms,
            **kwargs
        }

        self.log(
            step=step,
            category='compute',
            data=data
        )

    def log_result(
        self,
        step: int,
        metric_name: str,
        value: float,
        **kwargs
    ):
        """Log experiment result metric"""
        data = {
            'metric_name': metric_name,
            'value': value,
            **kwargs
        }

        self.log(
            step=step,
            category='result',
            data=data
        )

        # Also store in metrics
        self.metrics[metric_name].append({
            'step': step,
            'value': value,
            **kwargs
        })

    def set_metadata(self, key: str, value: Any):
        """Set experiment metadata"""
        self.metadata[key] = value

    def update_config(self, config: Dict[str, Any]):
        """Update experiment config"""
        self.metadata['config'].update(config)

    def get_metrics(self, metric_name: str) -> List[Dict[str, Any]]:
        """Get all values for a metric"""
        return self.metrics.get(metric_name, [])

    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        values = self.metrics.get(metric_name, [])

        if not values:
            return {}

        numeric_values = [v['value'] for v in values if isinstance(v['value'], (int, float))]

        if not numeric_values:
            return {}

        return {
            'count': len(numeric_values),
            'mean': np.mean(numeric_values),
            'std': np.std(numeric_values),
            'min': np.min(numeric_values),
            'max': np.max(numeric_values),
            'median': np.median(numeric_values)
        }

    def get_category_logs(self, category: str) -> List[LogEntry]:
        """Get all logs for a category"""
        return self.categorized_logs.get(category, [])

    def export_json(self, filename: Optional[str] = None):
        """Export logs to JSON"""
        if filename is None:
            filename = f"{self.experiment_name}_logs.json"

        filepath = self.output_dir / filename

        export_data = {
            'metadata': self.metadata,
            'logs': [asdict(entry) for entry in self.log_entries],
            'metrics_summary': {
                name: self.get_metric_summary(name)
                for name in self.metrics.keys()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Exported logs to {filepath}")

    def export_csv(self, category: str, filename: Optional[str] = None):
        """Export category logs to CSV"""
        if filename is None:
            filename = f"{self.experiment_name}_{category}.csv"

        filepath = self.output_dir / filename

        logs = self.categorized_logs.get(category, [])

        if not logs:
            self.logger.warning(f"No logs found for category: {category}")
            return

        # Extract all unique keys from data
        all_keys = set()
        for entry in logs:
            all_keys.update(entry.data.keys())

        fieldnames = ['timestamp', 'step'] + sorted(all_keys)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in logs:
                row = {
                    'timestamp': entry.timestamp,
                    'step': entry.step,
                    **entry.data
                }
                writer.writerow(row)

        self.logger.info(f"Exported {category} logs to {filepath}")

    def export_metrics_csv(self, filename: Optional[str] = None):
        """Export all metrics to CSV"""
        if filename is None:
            filename = f"{self.experiment_name}_metrics.csv"

        filepath = self.output_dir / filename

        # Build rows
        rows = []
        for metric_name, values in self.metrics.items():
            for entry in values:
                row = {
                    'metric_name': metric_name,
                    'step': entry.get('step', 0),
                    'value': entry.get('value', 0),
                    **{k: v for k, v in entry.items() if k not in ['step', 'value']}
                }
                rows.append(row)

        if not rows:
            self.logger.warning("No metrics to export")
            return

        # Get all keys
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())

        fieldnames = ['metric_name', 'step', 'value'] + sorted(all_keys - {'metric_name', 'step', 'value'})

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        self.logger.info(f"Exported metrics to {filepath}")

    def export_all(self):
        """Export all logs in all formats"""
        # Export main JSON
        self.export_json()

        # Export per-category CSVs
        for category in self.categorized_logs.keys():
            self.export_csv(category)

        # Export metrics CSV
        self.export_metrics_csv()

        # Export metadata
        metadata_file = self.output_dir / f"{self.experiment_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        self.logger.info(f"Exported all logs to {self.output_dir}")

    def finalize(self):
        """Finalize experiment and export all logs"""
        self.metadata['end_time'] = time.time()
        self.metadata['duration_seconds'] = self.metadata['end_time'] - self.metadata['start_time']
        self.metadata['total_entries'] = len(self.log_entries)

        self.export_all()

        self.logger.info(f"Experiment '{self.experiment_name}' finalized")
        self.logger.info(f"  Duration: {self.metadata['duration_seconds']:.2f}s")
        self.logger.info(f"  Total steps: {self.metadata['total_steps']}")
        self.logger.info(f"  Total entries: {self.metadata['total_entries']}")

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary"""
        return {
            'experiment_name': self.experiment_name,
            'total_entries': len(self.log_entries),
            'categories': {
                cat: len(logs) for cat, logs in self.categorized_logs.items()
            },
            'metrics_tracked': list(self.metrics.keys()),
            'duration_seconds': time.time() - self.metadata['start_time']
        }


# ===== LOGGER REGISTRY =====

class LoggerRegistry:
    """Registry for managing multiple experiment loggers"""

    def __init__(self):
        self.loggers: Dict[str, UnifiedLogger] = {}

    def get_or_create(
        self,
        experiment_name: str,
        output_dir: str = "results",
        **kwargs
    ) -> UnifiedLogger:
        """Get or create logger for experiment"""
        if experiment_name not in self.loggers:
            self.loggers[experiment_name] = UnifiedLogger(
                experiment_name=experiment_name,
                output_dir=output_dir,
                **kwargs
            )
        return self.loggers[experiment_name]

    def finalize_all(self):
        """Finalize all loggers"""
        for logger in self.loggers.values():
            logger.finalize()


# Global registry
_registry = LoggerRegistry()


def get_logger(experiment_name: str, **kwargs) -> UnifiedLogger:
    """Get logger for experiment"""
    return _registry.get_or_create(experiment_name, **kwargs)


def finalize_all_loggers():
    """Finalize all loggers"""
    _registry.finalize_all()


if __name__ == "__main__":
    # Quick test
    logger = get_logger("test_experiment", output_dir="test_results")

    # Log some data
    logger.log_agent_action(
        step=0,
        agent_id=1,
        action="help_other",
        ethical_score=0.85,
        mindfulness=0.72
    )

    logger.log_result(step=0, metric_name="casualties", value=5)
    logger.log_result(step=1, metric_name="casualties", value=3)
    logger.log_result(step=2, metric_name="casualties", value=2)

    # Get summary
    summary = logger.get_summary()
    print("Unified Logger Test:")
    print(f"  Total entries: {summary['total_entries']}")
    print(f"  Categories: {summary['categories']}")
    print(f"  Metrics: {summary['metrics_tracked']}")

    metrics_summary = logger.get_metric_summary("casualties")
    print(f"\n  Casualties metric:")
    print(f"    Mean: {metrics_summary['mean']:.2f}")
    print(f"    Min: {metrics_summary['min']:.0f}")
    print(f"    Max: {metrics_summary['max']:.0f}")

    # Export
    logger.export_all()

    print("\n✓ Unified Logger initialized successfully")
