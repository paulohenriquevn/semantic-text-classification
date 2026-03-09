"""RuleExecution model — an auditable record of semantic rule evaluation.

A RuleExecution captures the result of evaluating a single rule from the
semantic rule engine against a text object. It carries whether the rule
matched, the computed score, execution time, and structured evidence
enabling full auditability and traceability.

See PRD §11 for the data model specification.
See ADR-002 for the frozen/strict design decision.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypedDict

from semantic_conversation_engine.models.enums import ObjectType
from semantic_conversation_engine.models.types import RuleId


class EvidenceItem(TypedDict, total=False):
    """Structured evidence produced by a rule predicate evaluation.

    All fields are optional (total=False) because different predicate types
    produce different evidence. A lexical predicate produces matched_text;
    a semantic predicate produces score and threshold; a structural predicate
    may produce only predicate_type.

    Fields:
        predicate_type: Type of predicate that produced this evidence
            (e.g. 'lexical', 'semantic', 'structural', 'contextual').
        matched_text: Text fragment that triggered the predicate match.
        score: Numeric score produced by the predicate evaluation.
        threshold: Threshold used for the predicate decision.
        model_name: Name of the model used in evaluation (for semantic predicates).
        model_version: Version of the model used in evaluation.
        metadata: Additional predicate-specific evidence data.
    """

    predicate_type: str
    matched_text: str
    score: float
    threshold: float
    model_name: str
    model_version: str
    metadata: dict[str, Any]


class RuleExecution(BaseModel):
    """An auditable record of a semantic rule evaluation.

    Args:
        rule_id: Unique identifier of the rule. Format: rule_<uuid4>.
        rule_name: Human-readable name of the rule.
        source_id: ID of the evaluated object (turn, window, or conversation).
        source_type: Granularity level of the evaluated object.
        matched: Whether the rule matched (all predicates satisfied).
        score: Aggregate score of the rule evaluation in [0.0, 1.0].
        execution_time_ms: Time taken to evaluate the rule, in milliseconds.
        evidence: Structured evidence from predicate evaluations.
        metadata: Additional execution metadata (rule version, tags, etc.).
    """

    model_config = ConfigDict(frozen=True, strict=True)

    rule_id: RuleId
    rule_name: str
    source_id: str
    source_type: ObjectType
    matched: bool
    score: float
    execution_time_ms: float
    evidence: list[EvidenceItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rule_id")
    @classmethod
    def rule_id_must_not_be_empty(cls, v: RuleId) -> RuleId:
        """Reject empty or whitespace-only rule IDs.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("rule_id must not be empty or whitespace-only")
        return v

    @field_validator("rule_name")
    @classmethod
    def rule_name_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only rule names.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("rule_name must not be empty or whitespace-only")
        return v

    @field_validator("source_id")
    @classmethod
    def source_id_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only source IDs.

        Validates only — does NOT normalize. The original value is preserved.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("source_id must not be empty or whitespace-only")
        return v

    @field_validator("score")
    @classmethod
    def score_must_be_in_unit_range(cls, v: float) -> float:
        """Score must be in [0.0, 1.0]."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"score must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("execution_time_ms")
    @classmethod
    def execution_time_must_be_non_negative(cls, v: float) -> float:
        """Execution time cannot be negative."""
        if v < 0.0:
            raise ValueError(f"execution_time_ms must be non-negative, got {v}")
        return v

    @model_validator(mode="after")
    def matched_requires_evidence(self) -> "RuleExecution":
        """A matched rule must produce at least one evidence item.

        A match assertion without evidence violates auditability — it is
        a bug in the rule executor, not a valid state. Non-matched rules
        may have empty evidence (no obligation to document why something
        did not match).
        """
        if self.matched and not self.evidence:
            raise ValueError("matched=True requires at least one evidence item")
        return self
