"""Pipeline orchestration for the TalkEx — Conversation Intelligence Engine.

Coordinates the execution of pipeline stages (segmentation, context building,
embedding generation, retrieval, classification, rule evaluation, analytics)
into composable, auditable processing flows.
"""

from talkex.pipeline.benchmark import (
    ScenarioResult,
    SystemBenchmarkConfig,
    SystemBenchmarkReport,
    SystemBenchmarkRunner,
)
from talkex.pipeline.config import PipelineConfig
from talkex.pipeline.manifest import (
    PipelineRunManifest,
    compute_config_fingerprint,
)
from talkex.pipeline.metrics import compute_pipeline_metrics
from talkex.pipeline.pipeline import TextProcessingPipeline
from talkex.pipeline.protocols import (
    AnalyticsAggregator,
    AnalyticsReporter,
    Classifier,
    ContextBuilder,
    EmbeddingGenerator,
    HybridRetriever,
    LexicalIndex,
    Reranker,
    RuleCompiler,
    RuleEvaluator,
    Segmenter,
    VectorIndex,
)
from talkex.pipeline.result import PipelineResult, PipelineWarning
from talkex.pipeline.runner import PipelineRunner, RunSummary
from talkex.pipeline.system_pipeline import (
    StageResult,
    SystemPipeline,
    SystemPipelineResult,
)

__all__ = [
    "AnalyticsAggregator",
    "AnalyticsReporter",
    "Classifier",
    "ContextBuilder",
    "EmbeddingGenerator",
    "HybridRetriever",
    "LexicalIndex",
    "PipelineConfig",
    "PipelineResult",
    "PipelineRunManifest",
    "PipelineRunner",
    "PipelineWarning",
    "Reranker",
    "RuleCompiler",
    "RuleEvaluator",
    "RunSummary",
    "ScenarioResult",
    "Segmenter",
    "StageResult",
    "SystemBenchmarkConfig",
    "SystemBenchmarkReport",
    "SystemBenchmarkRunner",
    "SystemPipeline",
    "SystemPipelineResult",
    "TextProcessingPipeline",
    "VectorIndex",
    "compute_config_fingerprint",
    "compute_pipeline_metrics",
]
