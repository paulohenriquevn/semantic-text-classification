"""Classification evaluation framework.

Provides classification metrics, evaluation datasets, benchmark runners, and
experiment reporting for comparing classifiers.

Metrics: Precision, Recall, F1, Micro-F1, Macro-F1, per-label support.
"""

from talkex.classification_eval.dataset import (
    ClassificationDataset,
    ClassificationExample,
    GroundTruthLabel,
)
from talkex.classification_eval.metrics import (
    f1_score,
    macro_f1,
    micro_f1,
    per_label_metrics,
    precision,
    recall,
)
from talkex.classification_eval.report import (
    ClassificationExperimentReport,
    ClassificationMethodResult,
    ExampleMetrics,
)
from talkex.classification_eval.runner import (
    ClassificationBenchmarkRunner,
    ClassificationRunConfig,
)

__all__ = [
    "ClassificationBenchmarkRunner",
    "ClassificationDataset",
    "ClassificationExample",
    "ClassificationExperimentReport",
    "ClassificationMethodResult",
    "ClassificationRunConfig",
    "ExampleMetrics",
    "GroundTruthLabel",
    "f1_score",
    "macro_f1",
    "micro_f1",
    "per_label_metrics",
    "precision",
    "recall",
]
