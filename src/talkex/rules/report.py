"""Rule engine experiment reporting — structured output for rule benchmarks.

Provides data structures for per-rule metrics, per-configuration results,
and experiment reports with JSON/CSV export.

Reports compare different RuleEngineConfig policies (evaluation mode,
short-circuit policy, evidence policy) to help tune the rule engine
for cost, auditability, and quality.

Reports are deterministic and serializable for reproducibility.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuleMetrics:
    """Metrics for a single rule in a benchmark run.

    Args:
        rule_id: The evaluated rule identifier.
        rule_name: Human-readable rule name.
        matched: Whether the rule matched.
        score: Aggregate rule score.
        predicate_count: Number of predicates evaluated.
        predicates_matched: Number of predicates that matched.
        short_circuited: Whether evaluation was stopped early.
        execution_time_ms: Time taken to evaluate this rule.
    """

    rule_id: str
    rule_name: str
    matched: bool
    score: float
    predicate_count: int
    predicates_matched: int
    short_circuited: bool
    execution_time_ms: float


@dataclass(frozen=True)
class ConfigurationResult:
    """Evaluation results for a single engine configuration.

    Args:
        config_name: Human-readable name for the configuration.
        config_params: Configuration parameters used.
        rule_metrics: Per-rule detailed metrics.
        aggregated: Aggregated metrics across all rules.
        per_predicate_type: Predicate counts by type.
        total_rules: Number of rules evaluated.
        total_inputs: Number of inputs evaluated.
        total_ms: Total evaluation time in milliseconds.
    """

    config_name: str
    config_params: dict[str, str]
    rule_metrics: list[RuleMetrics]
    aggregated: dict[str, Any]
    per_predicate_type: dict[str, int]
    total_rules: int
    total_inputs: int
    total_ms: float


@dataclass(frozen=True)
class RuleExperimentReport:
    """Comparison report across multiple rule engine configurations.

    Args:
        experiment_name: Name of the benchmark experiment.
        experiment_version: Version for reproducibility.
        results: Per-configuration evaluation results.
    """

    experiment_name: str
    experiment_version: str
    results: list[ConfigurationResult] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize report to JSON string.

        Returns:
            Formatted JSON string with aggregated and per-predicate metrics.
        """
        data = {
            "experiment_name": self.experiment_name,
            "experiment_version": self.experiment_version,
            "results": [
                {
                    "config_name": r.config_name,
                    "config_params": r.config_params,
                    "total_rules": r.total_rules,
                    "total_inputs": r.total_inputs,
                    "total_ms": r.total_ms,
                    "aggregated": r.aggregated,
                    "per_predicate_type": r.per_predicate_type,
                }
                for r in self.results
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def save_json(self, path: str | Path) -> None:
        """Save report as JSON file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_json())

    def to_csv(self) -> str:
        """Serialize aggregated results to CSV string.

        One row per configuration with aggregated metrics.

        Returns:
            CSV string with one row per configuration.
        """
        if not self.results:
            return ""

        # Build header from first result's aggregated keys
        metric_keys = list(self.results[0].aggregated.keys())
        header = ["config_name", *metric_keys, "total_rules", "total_inputs", "total_ms"]

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header)

        for result in self.results:
            row = [result.config_name]
            for key in metric_keys:
                row.append(str(result.aggregated.get(key, "")))
            row.append(str(result.total_rules))
            row.append(str(result.total_inputs))
            row.append(str(result.total_ms))
            writer.writerow(row)

        return output.getvalue()

    def save_csv(self, path: str | Path) -> None:
        """Save aggregated results as CSV file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_csv())
