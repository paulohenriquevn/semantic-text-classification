"""Search value objects for the M5 QA hybrid-search surface (blueprint D1/D2).

`SearchQuery` is the QA request (text + optional criterion filters + window); `SearchHit` is one
fused, evidence-carrying result window. Frozen/strict so the boundary is immutable.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Criterion(BaseModel):
    """A single QA filter predicate (compliance/script/sentiment/intent), compiled to bound SQL.

    `field` names a whitelisted, indexable column/JSONB path; `value` is always passed as a bound
    parameter (never interpolated) — see `TurnSearchPort` implementations. M5 Phase 2 populates this.
    """

    model_config = ConfigDict(frozen=True)

    field: str
    value: str


class SearchQuery(BaseModel):
    """A QA hybrid-search request over the retention window."""

    model_config = ConfigDict(frozen=True)

    query_text: str
    top_k: int = Field(default=10, gt=0, le=100)
    window_days: int = Field(default=30, gt=0)
    criteria: tuple[Criterion, ...] = ()


class SearchHit(BaseModel):
    """One fused result window with the evidence QA needs to open + judge it."""

    model_config = ConfigDict(frozen=True)

    turn_id: str
    conversation_id: str
    raw_text: str
    created_at: datetime
    score: float
    lexical_score: float | None = None
    semantic_score: float | None = None


class Label(BaseModel):
    """A QA audit/label action on a window, destined for retraining (blueprint D2)."""

    model_config = ConfigDict(frozen=True)

    label_id: str
    turn_id: str
    conversation_id: str
    label: str
    labeled_by: str = "qa"
