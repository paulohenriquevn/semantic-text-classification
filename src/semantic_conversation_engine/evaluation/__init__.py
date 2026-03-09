"""Retrieval evaluation framework.

Provides IR metrics, evaluation datasets, benchmark runners, and
experiment reporting for comparing retrieval strategies.

Metrics: Recall@K, Precision@K, MRR, nDCG.
"""

from semantic_conversation_engine.evaluation.dataset import (
    EvaluationDataset,
    EvaluationExample,
    RelevanceJudgment,
)
from semantic_conversation_engine.evaluation.metrics import (
    ndcg,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from semantic_conversation_engine.evaluation.report import (
    ExperimentReport,
    MethodResult,
)
from semantic_conversation_engine.evaluation.runner import (
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
