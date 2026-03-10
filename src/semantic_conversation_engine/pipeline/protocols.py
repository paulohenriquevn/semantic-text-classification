"""Stage protocols for the processing pipeline.

These Protocol classes define the contracts that pipeline stage implementations
must fulfill. They enable dependency inversion — the pipeline orchestrator
depends on protocols, not on concrete implementations.

Protocols defined:
    Segmenter         — raw transcript → Turn objects
    ContextBuilder    — turns → ContextWindow objects
    EmbeddingGenerator — text → EmbeddingRecord vectors
    VectorIndex       — vector storage and similarity search
    LexicalIndex      — BM25 text indexing and search
    HybridRetriever   — combined lexical + semantic retrieval
    Reranker          — optional cross-encoder reranking
    Classifier        — feature vectors → classification results
    RuleCompiler      — DSL text → RuleDefinition (compiled AST)
    RuleEvaluator     — rules + input → RuleResult outcomes

Usage:
    Concrete implementations inherit from or structurally match these protocols.
    The pipeline orchestrator accepts any object that satisfies the protocol.
"""

from typing import Protocol

from semantic_conversation_engine.classification.models import (
    ClassificationInput,
    ClassificationResult,
)
from semantic_conversation_engine.context.config import ContextWindowConfig
from semantic_conversation_engine.embeddings.inputs import EmbeddingBatch
from semantic_conversation_engine.ingestion.inputs import TranscriptInput
from semantic_conversation_engine.models.context_window import ContextWindow
from semantic_conversation_engine.models.conversation import Conversation
from semantic_conversation_engine.models.embedding_record import EmbeddingRecord
from semantic_conversation_engine.models.turn import Turn
from semantic_conversation_engine.retrieval.models import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
)
from semantic_conversation_engine.rules.config import RuleEngineConfig
from semantic_conversation_engine.rules.models import (
    RuleDefinition,
    RuleEvaluationInput,
    RuleResult,
)
from semantic_conversation_engine.segmentation.config import SegmentationConfig


class Segmenter(Protocol):
    """Protocol for turn segmentation implementations.

    A Segmenter takes raw transcript input and produces a list of
    validated Turn objects with speaker attribution and offsets.
    """

    def segment(self, transcript: TranscriptInput, config: SegmentationConfig) -> list[Turn]:
        """Segment a transcript into turns.

        Args:
            transcript: Raw transcript input from the ingestion boundary.
            config: Segmentation configuration controlling parsing behavior.

        Returns:
            Ordered list of Turn objects with valid offsets and speaker attribution.

        Raises:
            PipelineError: If segmentation fails due to unrecoverable input issues.
        """
        ...


class ContextBuilder(Protocol):
    """Protocol for context window construction implementations.

    A ContextBuilder takes a conversation and its turns, then produces
    a list of ContextWindow objects using a sliding window strategy.
    """

    def build(
        self,
        conversation: Conversation,
        turns: list[Turn],
        config: ContextWindowConfig,
    ) -> list[ContextWindow]:
        """Build context windows from a conversation's turns.

        Args:
            conversation: The parent conversation (provides conversation_id).
            turns: Ordered list of turns to window over.
            config: Window configuration controlling size, stride, and rendering.

        Returns:
            Ordered list of ContextWindow objects covering the conversation.

        Raises:
            PipelineError: If window construction fails.
        """
        ...


class EmbeddingGenerator(Protocol):
    """Protocol for embedding generation implementations.

    An EmbeddingGenerator takes a batch of text inputs and produces
    a list of EmbeddingRecord objects with versioned vectors.
    """

    def generate(self, batch: EmbeddingBatch) -> list[EmbeddingRecord]:
        """Generate embeddings for a batch of inputs.

        Args:
            batch: Batch of embedding inputs with text and source identity.

        Returns:
            List of EmbeddingRecord objects, one per input. Order matches
            the input batch order.

        Raises:
            PipelineError: If embedding generation fails irrecoverably.
            ModelError: If the embedding model cannot be loaded or run.
        """
        ...


class VectorIndex(Protocol):
    """Protocol for vector index implementations.

    A VectorIndex stores embedding vectors and supports similarity
    search by vector. Implementations may use FAISS, Qdrant, Milvus,
    or other backends.
    """

    def upsert(self, records: list[EmbeddingRecord]) -> None:
        """Insert or update embedding records in the index.

        Args:
            records: Embedding records with vectors to index.

        Raises:
            PipelineError: If indexing fails.
        """
        ...

    def search_by_vector(self, vector: list[float], top_k: int = 10) -> list[RetrievalHit]:
        """Search the index by vector similarity.

        Args:
            vector: Query vector. Must match the index dimensionality.
            top_k: Maximum number of results.

        Returns:
            Ordered list of hits by descending similarity score.

        Raises:
            PipelineError: If search fails.
        """
        ...


class LexicalIndex(Protocol):
    """Protocol for lexical index implementations.

    A LexicalIndex stores documents and supports BM25 text search.
    Implementations may use rank-bm25, Elasticsearch, or other backends.
    """

    def index(self, documents: list[dict[str, object]]) -> None:
        """Index documents for lexical search.

        Each document dict must contain at minimum 'doc_id' and 'text'.
        Additional fields are stored as filterable metadata.

        Args:
            documents: Documents to index.

        Raises:
            PipelineError: If indexing fails.
        """
        ...

    def search(self, query_text: str, top_k: int = 10) -> list[RetrievalHit]:
        """Search the index by text (BM25).

        Args:
            query_text: The query text.
            top_k: Maximum number of results.

        Returns:
            Ordered list of hits by descending BM25 score.

        Raises:
            PipelineError: If search fails.
        """
        ...


class HybridRetriever(Protocol):
    """Protocol for hybrid retrieval implementations.

    A HybridRetriever combines lexical and semantic search results
    using score fusion, optional reranking, and structural filters.
    """

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Execute a hybrid retrieval query.

        Combines BM25 and ANN results, applies fusion, optional
        reranking, and structural filters. Degrades gracefully
        when an index is unavailable.

        Args:
            query: Retrieval query with text, filters, and parameters.

        Returns:
            RetrievalResult with ordered hits and execution metadata.

        Raises:
            PipelineError: If retrieval fails irrecoverably.
        """
        ...


class Reranker(Protocol):
    """Protocol for optional cross-encoder reranking.

    A Reranker rescores candidate hits using a more expensive model
    (typically a cross-encoder). This is an optional post-fusion step.
    """

    def rerank(self, query_text: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Rerank candidate hits using a cross-encoder.

        Args:
            query_text: The original query text.
            hits: Candidate hits from fusion.

        Returns:
            Reranked hits with updated scores and ranks.

        Raises:
            ModelError: If the reranker model fails.
        """
        ...


class Classifier(Protocol):
    """Protocol for classification implementations.

    A Classifier takes classification inputs (text with features) and
    produces classification results with label scores and evidence.
    Supports both single-label and multi-label modes.
    """

    def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
        """Classify a batch of inputs.

        Args:
            inputs: List of classification inputs with text and features.

        Returns:
            List of ClassificationResult objects, one per input.
            Order matches the input list order.

        Raises:
            PipelineError: If classification fails irrecoverably.
            ModelError: If the classification model cannot be loaded or run.
        """
        ...


class RuleCompiler(Protocol):
    """Protocol for rule compilation implementations.

    A RuleCompiler takes a DSL text representation and compiles it
    into a RuleDefinition with a typed AST ready for evaluation.
    """

    def compile(self, dsl_text: str, rule_id: str, rule_name: str) -> RuleDefinition:
        """Compile a DSL text into a rule definition.

        Args:
            dsl_text: Rule text in the DSL format.
            rule_id: Unique identifier for the compiled rule.
            rule_name: Human-readable name for the compiled rule.

        Returns:
            RuleDefinition with compiled AST and identity.

        Raises:
            RuleError: If the DSL text is syntactically or semantically invalid.
        """
        ...


class RuleEvaluator(Protocol):
    """Protocol for rule evaluation implementations.

    A RuleEvaluator takes a set of compiled rules and an evaluation input,
    then produces rule results with match outcomes and evidence.
    """

    def evaluate(
        self,
        rules: list[RuleDefinition],
        evaluation_input: RuleEvaluationInput,
        config: RuleEngineConfig,
    ) -> list[RuleResult]:
        """Evaluate rules against an input.

        Args:
            rules: List of compiled rule definitions to evaluate.
            evaluation_input: The object to evaluate rules against.
            config: Evaluation configuration (mode, evidence policy, etc.).

        Returns:
            List of RuleResult objects, one per rule evaluated.
            Order may differ from input if short-circuit policy reorders.

        Raises:
            RuleError: If rule evaluation fails irrecoverably.
        """
        ...
