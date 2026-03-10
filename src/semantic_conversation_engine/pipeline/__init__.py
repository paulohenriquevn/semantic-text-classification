"""Pipeline orchestration for the Semantic Conversation Intelligence Engine.

Coordinates the execution of pipeline stages (segmentation, context building,
embedding generation, retrieval, classification, rule evaluation) into
composable, auditable processing flows.
"""

from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
from semantic_conversation_engine.pipeline.protocols import (
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

__all__ = [
    "Classifier",
    "ContextBuilder",
    "EmbeddingGenerator",
    "HybridRetriever",
    "LexicalIndex",
    "PipelineResult",
    "PipelineWarning",
    "Reranker",
    "RuleCompiler",
    "RuleEvaluator",
    "Segmenter",
    "TextProcessingPipeline",
    "VectorIndex",
]
