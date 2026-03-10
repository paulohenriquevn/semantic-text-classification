"""Classification experiment reporting — structured output for classifier benchmarks.

Provides data structures for per-example metrics, per-method results,
and experiment reports with JSON/CSV export.

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
class ExampleMetrics:
    """Metrics for a single classification example.

    Args:
        example_id: The evaluated example identifier.
        predicted_labels: Labels predicted by the classifier.
        ground_truth_labels: Known correct labels.
        precision: Fraction of predicted that are correct.
        recall: Fraction of ground truth that were predicted.
        f1: Harmonic mean of precision and recall.
    """

    example_id: str
    predicted_labels: list[str]
    ground_truth_labels: list[str]
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ClassificationMethodResult:
    """Evaluation results for a single classification method.

    Args:
        method_name: Human-readable method name.
        example_metrics: Per-example detailed metrics.
        aggregated: Aggregated metrics across all examples.
        per_label: Per-label precision, recall, F1, and support.
        total_examples: Number of examples evaluated.
        total_ms: Total evaluation time in milliseconds.
    """

    method_name: str
    example_metrics: list[ExampleMetrics]
    aggregated: dict[str, Any]
    per_label: dict[str, dict[str, float]]
    total_examples: int
    total_ms: float


@dataclass(frozen=True)
class ClassificationExperimentReport:
    """Comparison report across multiple classification methods.

    Args:
        dataset_name: Name of the evaluation dataset.
        dataset_version: Version of the evaluation dataset.
        label_names: Labels evaluated.
        results: Per-method evaluation results.
    """

    dataset_name: str
    dataset_version: str
    label_names: list[str]
    results: list[ClassificationMethodResult] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize report to JSON string.

        Returns:
            Formatted JSON string with aggregated and per-label metrics.
        """
        data = {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "label_names": self.label_names,
            "results": [
                {
                    "method_name": r.method_name,
                    "total_examples": r.total_examples,
                    "total_ms": r.total_ms,
                    "aggregated": r.aggregated,
                    "per_label": r.per_label,
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

        One row per method with aggregated metrics.

        Returns:
            CSV string with one row per method.
        """
        if not self.results:
            return ""

        # Build header from first result's aggregated keys
        metric_keys = list(self.results[0].aggregated.keys())
        header = ["method", *metric_keys, "total_examples", "total_ms"]

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header)

        for result in self.results:
            row = [result.method_name]
            for key in metric_keys:
                row.append(str(result.aggregated.get(key, "")))
            row.append(str(result.total_examples))
            row.append(str(result.total_ms))
            writer.writerow(row)

        return output.getvalue()

    def save_csv(self, path: str | Path) -> None:
        """Save aggregated results as CSV file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_csv())
