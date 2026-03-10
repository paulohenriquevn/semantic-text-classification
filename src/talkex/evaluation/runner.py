"""Benchmark runner — evaluates retrievers against evaluation datasets.

Executes a retriever against every example in an EvaluationDataset,
computes IR metrics per query, and aggregates into per-method results.

The runner is retriever-agnostic: any object satisfying the
HybridRetriever protocol (or exposing a `retrieve(RetrievalQuery)`
method) can be benchmarked.

Results are collected into an ExperimentReport for comparison and
serialization.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from talkex.evaluation.dataset import (
    EvaluationDataset,
)
from talkex.evaluation.metrics import (
    ndcg,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from talkex.evaluation.report import (
    ExperimentReport,
    MethodResult,
    QueryMetrics,
)
from talkex.retrieval.models import RetrievalQuery, RetrievalResult


class _Retriever(Protocol):
    """Structural type for any retriever with a retrieve method."""

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult: ...


@dataclass(frozen=True)
class RunConfig:
    """Configuration for a benchmark run.

    Args:
        k_values: List of K values to compute metrics at.
            Default: [1, 3, 5, 10].
    """

    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass
class BenchmarkRunner:
    """Runs retrieval evaluation benchmarks.

    Evaluates one or more retrievers against an EvaluationDataset,
    computing IR metrics at multiple K values.

    Args:
        dataset: The evaluation dataset to benchmark against.
        config: Benchmark configuration.
    """

    dataset: EvaluationDataset
    config: RunConfig = field(default_factory=RunConfig)

    def evaluate(
        self,
        retriever: _Retriever,
        method_name: str,
    ) -> MethodResult:
        """Evaluate a single retriever against the dataset.

        Args:
            retriever: The retriever to evaluate.
            method_name: Human-readable name for this method
                (e.g., "BM25", "HYBRID_RRF").

        Returns:
            MethodResult with per-query and aggregated metrics.
        """
        max_k = max(self.config.k_values)
        query_metrics_list: list[QueryMetrics] = []

        start = time.monotonic()

        for example in self.dataset.examples:
            query = RetrievalQuery(
                query_text=example.query_text,
                top_k=max_k,
            )
            result = retriever.retrieve(query)
            retrieved_ids = [hit.object_id for hit in result.hits]

            relevant = example.relevant_doc_ids
            rel_map = example.relevance_map

            per_k_recall: dict[int, float] = {}
            per_k_precision: dict[int, float] = {}
            per_k_ndcg: dict[int, float] = {}

            for k in self.config.k_values:
                per_k_recall[k] = recall_at_k(retrieved_ids, relevant, k)
                per_k_precision[k] = precision_at_k(retrieved_ids, relevant, k)
                per_k_ndcg[k] = ndcg(retrieved_ids, rel_map, k)

            rr = reciprocal_rank(retrieved_ids, relevant)

            query_metrics_list.append(
                QueryMetrics(
                    query_id=example.query_id,
                    recall_at_k=per_k_recall,
                    precision_at_k=per_k_precision,
                    ndcg_at_k=per_k_ndcg,
                    reciprocal_rank=rr,
                    retrieved_count=len(retrieved_ids),
                    relevant_count=len(relevant),
                )
            )

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        # Aggregate: mean across all queries
        aggregated = self._aggregate(query_metrics_list)

        return MethodResult(
            method_name=method_name,
            query_metrics=query_metrics_list,
            aggregated=aggregated,
            total_queries=len(self.dataset.examples),
            total_ms=elapsed_ms,
        )

    def compare(
        self,
        retrievers: Mapping[str, _Retriever],
    ) -> ExperimentReport:
        """Evaluate multiple retrievers and produce a comparison report.

        Args:
            retrievers: Mapping from method name to retriever instance.

        Returns:
            ExperimentReport comparing all methods.
        """
        results = [self.evaluate(retriever, method_name) for method_name, retriever in retrievers.items()]
        return ExperimentReport(
            dataset_name=self.dataset.name,
            dataset_version=self.dataset.version,
            k_values=list(self.config.k_values),
            results=results,
        )

    def _aggregate(
        self,
        query_metrics_list: list[QueryMetrics],
    ) -> dict[str, Any]:
        """Compute mean metrics across all queries.

        Args:
            query_metrics_list: Per-query metric results.

        Returns:
            Dict with aggregated metrics.
        """
        if not query_metrics_list:
            return {}

        n = len(query_metrics_list)
        agg: dict[str, Any] = {}

        # Mean reciprocal rank
        agg["mrr"] = round(sum(qm.reciprocal_rank for qm in query_metrics_list) / n, 4)

        # Mean per-K metrics
        for k in self.config.k_values:
            agg[f"recall@{k}"] = round(
                sum(qm.recall_at_k.get(k, 0.0) for qm in query_metrics_list) / n,
                4,
            )
            agg[f"precision@{k}"] = round(
                sum(qm.precision_at_k.get(k, 0.0) for qm in query_metrics_list) / n,
                4,
            )
            agg[f"ndcg@{k}"] = round(
                sum(qm.ndcg_at_k.get(k, 0.0) for qm in query_metrics_list) / n,
                4,
            )

        return agg
