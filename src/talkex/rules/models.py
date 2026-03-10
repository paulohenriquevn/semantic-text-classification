"""Rule engine internal models — rule definitions, inputs, and results.

Pipeline-internal data objects for the rule engine stage.
These are NOT domain entities — they are payloads exchanged between
rule engine components. The domain entity is RuleExecution (in models/).

RuleDefinition wraps a compiled AST with identity and priority.
RuleEvaluationInput packages the text object to evaluate.
PredicateResult captures individual predicate outcomes.
RuleResult captures the full outcome of a single rule evaluation.

Boundary mapping:
    RuleResult → RuleExecution (domain entity) happens at the orchestrator level,
    parallel to how ClassificationResult → Prediction works in classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from talkex.rules.ast import ASTNode
from talkex.rules.config import PredicateType


@dataclass(frozen=True)
class RuleDefinition:
    """A named, versioned rule with its compiled AST.

    Args:
        rule_id: Unique identifier for this rule.
        rule_name: Human-readable rule name.
        rule_version: Version string for reproducibility.
        description: Human-readable description of what the rule detects.
        ast: Compiled AST root node.
        priority: Evaluation priority (higher = evaluated first in PRIORITY mode).
        tags: Categorical tags for filtering and grouping (e.g., "compliance", "fraud").
        metadata: Additional rule context (author, created_at, etc.).
    """

    rule_id: str
    rule_name: str
    rule_version: str
    description: str
    ast: ASTNode
    priority: int = 0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleEvaluationInput:
    """Input to the rule evaluator — the object to evaluate rules against.

    Args:
        source_id: ID of the object being evaluated (turn_id, window_id, etc.).
        source_type: Granularity level (turn, context_window, conversation).
        text: Text content to evaluate predicates against.
        embedding: Pre-computed embedding vector. None if not available.
        features: Pre-computed features (lexical, structural, etc.).
        speaker_role: Speaker role for structural predicates. None if N/A.
        metadata: Additional context for predicate evaluation.
    """

    source_id: str
    source_type: str
    text: str
    embedding: list[float] | None = None
    features: dict[str, float] = field(default_factory=dict)
    speaker_role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredicateResult:
    """Outcome of a single predicate evaluation.

    Args:
        predicate_type: Signal family of the evaluated predicate.
        field_name: The field or signal that was checked.
        operator: The comparison operator used.
        matched: Whether the predicate matched.
        score: Numeric score produced by the predicate (0.0 if boolean).
        threshold: Threshold used for the decision.
        matched_text: Text fragment that triggered the match. None if no match.
        execution_time_ms: Time taken to evaluate this predicate.
        metadata: Additional predicate-specific evidence data.
    """

    predicate_type: PredicateType
    field_name: str
    operator: str
    matched: bool
    score: float = 0.0
    threshold: float = 0.0
    matched_text: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_evidence_item(self) -> dict[str, Any]:
        """Convert to EvidenceItem-compatible dict for RuleExecution mapping.

        Returns:
            Dict compatible with the EvidenceItem TypedDict structure.
        """
        evidence: dict[str, Any] = {
            "predicate_type": self.predicate_type.value,
            "score": self.score,
            "threshold": self.threshold,
        }
        if self.matched_text is not None:
            evidence["matched_text"] = self.matched_text
        if self.metadata:
            evidence["metadata"] = self.metadata
        return evidence


@dataclass(frozen=True)
class RuleResult:
    """Outcome of evaluating a single rule against an input.

    Args:
        rule_id: ID of the evaluated rule.
        rule_name: Name of the evaluated rule.
        rule_version: Version of the evaluated rule.
        source_id: ID of the evaluated object.
        source_type: Granularity level of the evaluated object.
        matched: Whether all predicates in the rule matched.
        score: Aggregate score of the rule evaluation in [0.0, 1.0].
        predicate_results: Individual predicate outcomes.
        execution_time_ms: Total rule evaluation time in milliseconds.
        short_circuited: Whether evaluation was stopped early.
        metadata: Additional execution context.
    """

    rule_id: str
    rule_name: str
    rule_version: str
    source_id: str
    source_type: str
    matched: bool
    score: float
    predicate_results: list[PredicateResult]
    execution_time_ms: float = 0.0
    short_circuited: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
