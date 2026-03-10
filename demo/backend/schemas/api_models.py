"""API request/response models for the demo backend.

Boundary objects — these are the external contract. Internal TalkEx
domain models are mapped to/from these at the service layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Search API
# ---------------------------------------------------------------------------


class SearchFilters(BaseModel):
    """Optional filters for search queries."""

    speaker: str | None = None
    domain: str | None = None
    topic: str | None = None


class SearchRequest(BaseModel):
    """POST /search request body."""

    query: str = Field(..., min_length=1, max_length=500)
    filters: SearchFilters | None = None
    top_k: int = Field(default=20, ge=1, le=100)


class SearchHit(BaseModel):
    """Single search result."""

    window_id: str
    conversation_id: str
    text: str
    lexical_score: float | None = None
    semantic_score: float | None = None
    score: float
    rank: int
    domain: str = ""
    topic: str = ""


class SearchResponse(BaseModel):
    """POST /search response body."""

    results: list[SearchHit]
    total_candidates: int = 0
    query: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Conversation API
# ---------------------------------------------------------------------------


class TurnResponse(BaseModel):
    """Single turn in a conversation."""

    turn_id: str
    speaker: str
    raw_text: str
    normalized_text: str | None = None


class ConversationResponse(BaseModel):
    """GET /conversation/{id} response body."""

    conversation_id: str
    domain: str
    topic: str
    asr_confidence: float = 0.0
    audio_duration_seconds: int = 0
    turn_count: int = 0
    window_count: int = 0
    text_preview: str = ""
    turns: list[TurnResponse] = []


# ---------------------------------------------------------------------------
# Filters API
# ---------------------------------------------------------------------------


class FiltersResponse(BaseModel):
    """GET /filters response body."""

    domains: list[str]
    topics: list[str]
    speakers: list[str]


# ---------------------------------------------------------------------------
# Analytics API
# ---------------------------------------------------------------------------


class DomainCount(BaseModel):
    """Count of conversations per domain."""

    domain: str
    count: int


class AnalyticsSummary(BaseModel):
    """GET /analytics/summary response body."""

    total_conversations: int
    total_windows: int
    total_embeddings: int
    avg_asr_confidence: float
    avg_turns_per_conversation: float
    domains: list[DomainCount]
    embedding_model: str = ""
    index_dimensions: int = 0


# ---------------------------------------------------------------------------
# Categories API
# ---------------------------------------------------------------------------


class CreateCategoryRequest(BaseModel):
    """POST /categories request body."""

    name: str = Field(..., min_length=1, max_length=200)
    dsl_expression: str = Field(..., min_length=1, max_length=2000)
    description: str = ""


class CategoryMatchResponse(BaseModel):
    """A single window match within a category."""

    window_id: str
    conversation_id: str
    score: float
    matched_text: str | None = None


class CategoryResponse(BaseModel):
    """Single category response."""

    category_id: str
    name: str
    dsl_expression: str
    description: str = ""
    match_count: int = 0
    conversation_count: int = 0
    applied: bool = False
    apply_time_ms: float = 0.0
    created_at: str = ""


class CategoryDetailResponse(CategoryResponse):
    """Category with match details."""

    matches: list[CategoryMatchResponse] = []


class CategoryListResponse(BaseModel):
    """GET /categories response body."""

    categories: list[CategoryResponse]
    total: int


class ValidateDSLRequest(BaseModel):
    """POST /categories/validate request body."""

    dsl_expression: str = Field(..., min_length=1, max_length=2000)


class ValidateDSLResponse(BaseModel):
    """POST /categories/validate response body."""

    valid: bool
    error: str | None = None


class PreviewDSLRequest(BaseModel):
    """POST /categories/preview request body."""

    dsl_expression: str = Field(..., min_length=1, max_length=2000)


class PredicateEvidenceResponse(BaseModel):
    """Evidence from a single matched predicate."""

    predicate_type: str
    field_name: str
    operator: str
    score: float = 0.0
    threshold: float = 0.0
    matched_text: str | None = None


class PreviewMatchResponse(BaseModel):
    """A single window match in a preview — includes full text and per-predicate evidence."""

    window_id: str
    conversation_id: str
    score: float
    window_text: str
    evidence: list[PredicateEvidenceResponse] = []


class ScoreDistribution(BaseModel):
    """Statistical summary of match scores."""

    min: float
    max: float
    mean: float
    median: float
    p90: float


class PreExecutionAnalysis(BaseModel):
    """Static analysis of the DSL query before execution."""

    predicate_families: list[str]
    missing_families: list[str]
    predicate_count: int
    complexity: str
    threshold_warnings: list[str] = []
    pitfalls: list[str] = []


class PostExecutionAnalysis(BaseModel):
    """Analysis of query results after execution."""

    score_distribution: ScoreDistribution | None = None
    window_coverage_pct: float = 0.0
    conversation_coverage_pct: float = 0.0
    concentration_ratio: float = 0.0
    signal_warnings: list[str] = []
    quality_score: int = 0


class QueryEvaluation(BaseModel):
    """Combined pre- and post-execution query evaluation."""

    pre_execution: PreExecutionAnalysis
    post_execution: PostExecutionAnalysis


class PreviewDSLResponse(BaseModel):
    """POST /categories/preview response body — dry-run results without persisting."""

    valid: bool
    error: str | None = None
    match_count: int = 0
    conversation_count: int = 0
    sample_matches: list[PreviewMatchResponse] = []
    latency_ms: float = 0.0
    evaluation: QueryEvaluation | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = "ok"
    conversations: int = 0
    windows: int = 0
    embeddings: int = 0
    version: str = ""
