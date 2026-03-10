"""Classification benchmark runner — evaluates classifiers against evaluation datasets.

Executes a classifier against every example in a ClassificationDataset,
computes classification metrics per example, and aggregates into per-method results.

The runner is classifier-agnostic: any object satisfying the Classifier protocol
(or exposing a `classify(list[ClassificationInput]) -> list[ClassificationResult]`
method) can be benchmarked.

Results are collected into a ClassificationExperimentReport for comparison
and serialization.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from semantic_conversation_engine.classification.models import (
    ClassificationInput,
    ClassificationResult,
)
from semantic_conversation_engine.classification_eval.dataset import (
    ClassificationDataset,
)
from semantic_conversation_engine.classification_eval.metrics import (
    f1_score,
    macro_f1,
    micro_f1,
    per_label_metrics,
    precision,
    recall,
)
from semantic_conversation_engine.classification_eval.report import (
    ClassificationExperimentReport,
    ClassificationMethodResult,
    ExampleMetrics,
)


class _Classifier(Protocol):
    """Structural type for any classifier with a classify method."""

    def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]: ...


@dataclass(frozen=True)
class ClassificationRunConfig:
    """Configuration for a classification benchmark run.

    Args:
        threshold_override: Optional threshold to override classifier defaults.
            None means use the threshold from each LabelScore as-is.
    """

    threshold_override: float | None = None


@dataclass
class ClassificationBenchmarkRunner:
    """Runs classification evaluation benchmarks.

    Evaluates one or more classifiers against a ClassificationDataset,
    computing per-example and aggregated classification metrics.

    Args:
        dataset: The evaluation dataset to benchmark against.
        config: Benchmark configuration.
    """

    dataset: ClassificationDataset
    config: ClassificationRunConfig = field(default_factory=ClassificationRunConfig)

    def evaluate(
        self,
        classifier: _Classifier,
        method_name: str,
    ) -> ClassificationMethodResult:
        """Evaluate a single classifier against the dataset.

        Args:
            classifier: The classifier to evaluate.
            method_name: Human-readable name for this method
                (e.g., "similarity", "logistic").

        Returns:
            ClassificationMethodResult with per-example and aggregated metrics.
        """
        start = time.monotonic()

        # Build classification inputs from dataset examples
        inputs = [
            ClassificationInput(
                source_id=ex.example_id,
                source_type=ex.source_type,
                text=ex.text,
                embedding=ex.embedding,
                features=ex.features,
                metadata=ex.metadata,
            )
            for ex in self.dataset.examples
        ]

        # Run classifier
        results = classifier.classify(inputs)

        # Compute per-example metrics
        example_metrics_list: list[ExampleMetrics] = []
        all_predictions: list[set[str]] = []
        all_ground_truths: list[set[str]] = []

        for example, result in zip(self.dataset.examples, results, strict=True):
            gt = example.label_set
            pred = self._extract_predicted(result)

            all_predictions.append(pred)
            all_ground_truths.append(gt)

            example_metrics_list.append(
                ExampleMetrics(
                    example_id=example.example_id,
                    predicted_labels=sorted(pred),
                    ground_truth_labels=sorted(gt),
                    precision=precision(pred, gt),
                    recall=recall(pred, gt),
                    f1=f1_score(pred, gt),
                )
            )

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        # Aggregated metrics
        aggregated = self._aggregate(all_predictions, all_ground_truths)

        # Per-label metrics
        label_metrics = per_label_metrics(
            all_predictions,
            all_ground_truths,
            self.dataset.label_names,
        )

        return ClassificationMethodResult(
            method_name=method_name,
            example_metrics=example_metrics_list,
            aggregated=aggregated,
            per_label=label_metrics,
            total_examples=len(self.dataset.examples),
            total_ms=elapsed_ms,
        )

    def compare(
        self,
        classifiers: Mapping[str, _Classifier],
    ) -> ClassificationExperimentReport:
        """Evaluate multiple classifiers and produce a comparison report.

        Args:
            classifiers: Mapping from method name to classifier instance.

        Returns:
            ClassificationExperimentReport comparing all methods.
        """
        results = [self.evaluate(classifier, method_name) for method_name, classifier in classifiers.items()]
        return ClassificationExperimentReport(
            dataset_name=self.dataset.name,
            dataset_version=self.dataset.version,
            label_names=list(self.dataset.label_names),
            results=results,
        )

    def _extract_predicted(self, result: ClassificationResult) -> set[str]:
        """Extract predicted label set from a ClassificationResult.

        If threshold_override is set, uses that threshold instead of
        each LabelScore's own threshold.

        Args:
            result: Classification result with label scores.

        Returns:
            Set of predicted label names.
        """
        threshold = self.config.threshold_override
        if threshold is not None:
            return {ls.label for ls in result.label_scores if ls.score >= threshold}
        return set(result.predicted_labels)

    def _aggregate(
        self,
        predictions: list[set[str]],
        ground_truths: list[set[str]],
    ) -> dict[str, float]:
        """Compute aggregated metrics across all examples.

        Args:
            predictions: List of predicted label sets.
            ground_truths: List of ground truth label sets.

        Returns:
            Dict with aggregated metrics.
        """
        if not predictions:
            return {}

        n = len(predictions)
        label_names = self.dataset.label_names

        # Mean per-example metrics
        mean_precision = round(sum(precision(p, g) for p, g in zip(predictions, ground_truths, strict=True)) / n, 4)
        mean_recall = round(sum(recall(p, g) for p, g in zip(predictions, ground_truths, strict=True)) / n, 4)
        mean_f1 = round(sum(f1_score(p, g) for p, g in zip(predictions, ground_truths, strict=True)) / n, 4)

        # Global metrics
        mi_f1 = round(micro_f1(predictions, ground_truths, label_names), 4)
        ma_f1 = round(macro_f1(predictions, ground_truths, label_names), 4)

        return {
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1": mean_f1,
            "micro_f1": mi_f1,
            "macro_f1": ma_f1,
        }
