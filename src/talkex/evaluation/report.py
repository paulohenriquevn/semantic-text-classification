"""Experiment reporting — structured output for retrieval benchmarks.

Provides data structures for per-query metrics, per-method results,
and experiment reports with JSON/CSV export.

Reports are deterministic and serializable for reproducibility.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryMetrics:
    """Metrics for a single evaluation query.

    Args:
        query_id: The evaluated query identifier.
        recall_at_k: Recall values per K.
        precision_at_k: Precision values per K.
        ndcg_at_k: nDCG values per K.
        reciprocal_rank: Reciprocal rank of first relevant doc.
        retrieved_count: Number of documents retrieved.
        relevant_count: Number of known relevant documents.
    """

    query_id: str
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    reciprocal_rank: float
    retrieved_count: int
    relevant_count: int


@dataclass(frozen=True)
class MethodResult:
    """Evaluation results for a single retrieval method.

    Args:
        method_name: Human-readable method name.
        query_metrics: Per-query detailed metrics.
        aggregated: Aggregated (mean) metrics across all queries.
        total_queries: Number of queries evaluated.
        total_ms: Total evaluation time in milliseconds.
    """

    method_name: str
    query_metrics: list[QueryMetrics]
    aggregated: dict[str, Any]
    total_queries: int
    total_ms: float


@dataclass(frozen=True)
class ExperimentReport:
    """Comparison report across multiple retrieval methods.

    Args:
        dataset_name: Name of the evaluation dataset.
        dataset_version: Version of the evaluation dataset.
        k_values: K values used in evaluation.
        results: Per-method evaluation results.
    """

    dataset_name: str
    dataset_version: str
    k_values: list[int]
    results: list[MethodResult]

    def to_json(self) -> str:
        """Serialize report to JSON string.

        Returns:
            Formatted JSON string.
        """
        data = {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "k_values": self.k_values,
            "results": [
                {
                    "method_name": r.method_name,
                    "total_queries": r.total_queries,
                    "total_ms": r.total_ms,
                    "aggregated": r.aggregated,
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

        Returns:
            CSV string with one row per method.
        """
        if not self.results:
            return ""

        # Build header from first result's aggregated keys
        metric_keys = list(self.results[0].aggregated.keys())
        header = ["method", *metric_keys, "total_queries", "total_ms"]

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header)

        for result in self.results:
            row = [result.method_name]
            for key in metric_keys:
                row.append(str(result.aggregated.get(key, "")))
            row.append(str(result.total_queries))
            row.append(str(result.total_ms))
            writer.writerow(row)

        return output.getvalue()

    def save_csv(self, path: str | Path) -> None:
        """Save aggregated results as CSV file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_csv())
