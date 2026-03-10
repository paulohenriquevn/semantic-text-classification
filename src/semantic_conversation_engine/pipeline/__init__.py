"""Pipeline orchestration for the Semantic Conversation Intelligence Engine.

Coordinates the execution of pipeline stages (segmentation, context building,
embedding generation, retrieval, classification, rule evaluation, analytics)
into composable, auditable processing flows.
"""

from semantic_conversation_engine.pipeline.benchmark import (
    ScenarioResult,
    SystemBenchmarkConfig,
    SystemBenchmarkReport,
    SystemBenchmarkRunner,
)
from semantic_conversation_engine.pipeline.config import PipelineConfig
from semantic_conversation_engine.pipeline.manifest import (
    PipelineRunManifest,
    compute_config_fingerprint,
)
from semantic_conversation_engine.pipeline.metrics import compute_pipeline_metrics
from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
from semantic_conversation_engine.pipeline.protocols import (
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
from semantic_conversation_engine.pipeline.result import PipelineResult, PipelineWarning
from semantic_conversation_engine.pipeline.runner import PipelineRunner, RunSummary
from semantic_conversation_engine.pipeline.system_pipeline import (
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
