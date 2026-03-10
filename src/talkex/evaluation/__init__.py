"""Retrieval evaluation framework.

Provides IR metrics, evaluation datasets, benchmark runners, and
experiment reporting for comparing retrieval strategies.

Metrics: Recall@K, Precision@K, MRR, nDCG.
"""

from talkex.evaluation.dataset import (
    EvaluationDataset,
    EvaluationExample,
    RelevanceJudgment,
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
)
from talkex.evaluation.runner import (
    BenchmarkRunner,
    RunConfig,
)

__all__ = [
    "BenchmarkRunner",
    "EvaluationDataset",
    "EvaluationExample",
    "ExperimentReport",
    "MethodResult",
    "RelevanceJudgment",
    "RunConfig",
    "ndcg",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
