"""Rule engine configuration — evaluation mode, evidence policy, and cost ordering.

Defines configuration objects for rule engine behavior. All configs are
frozen pydantic models for immutability and validation.

Evaluation modes:
    ALL:           evaluate all rules, collect all results
    SHORT_CIRCUIT: stop on first matched rule (ordered by priority)

Evidence policies:
    ALWAYS:     collect evidence for both matched and unmatched predicates
    MATCH_ONLY: collect evidence only for matched predicates (cheaper)

Short-circuit policies:
    COST_ASCENDING:  evaluate cheapest predicates first (lexical before semantic)
    PRIORITY:        evaluate by rule priority (highest first)
    DECLARATION:     evaluate in declaration order
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RuleEvaluationMode(StrEnum):
    """How rules are evaluated against an input.

    ALL: evaluate every rule, collect all results.
    SHORT_CIRCUIT: stop after first matched rule.
    """

    ALL = "all"
    SHORT_CIRCUIT = "short_circuit"


class EvidencePolicy(StrEnum):
    """How much evidence to collect during evaluation.

    ALWAYS: collect evidence for matched AND unmatched predicates.
    MATCH_ONLY: collect evidence only for matched predicates.
    """

    ALWAYS = "always"
    MATCH_ONLY = "match_only"


class ShortCircuitPolicy(StrEnum):
    """Predicate evaluation order within a single rule.

    COST_ASCENDING: cheapest predicates first (lexical < structural < semantic).
    PRIORITY: highest priority predicates first.
    DECLARATION: evaluate in the order predicates are declared.
    """

    COST_ASCENDING = "cost_ascending"
    PRIORITY = "priority"
    DECLARATION = "declaration"


class PredicateType(StrEnum):
    """The four signal families for rule predicates.

    LEXICAL: keyword matching, regex, BM25 scoring.
    SEMANTIC: embedding similarity, intent scores.
    STRUCTURAL: speaker role, channel, turn count, duration.
    CONTEXTUAL: cross-turn patterns, repeated mentions, occurrence sequences.
    """

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    CONTEXTUAL = "contextual"


class RuleEngineConfig(BaseModel):
    """Configuration for the semantic rule engine.

    Args:
        evaluation_mode: How rules are evaluated (all vs short-circuit).
        evidence_policy: How much evidence to collect.
        short_circuit_policy: Predicate evaluation order within a rule.
        max_rules_per_evaluation: Maximum rules to evaluate per input.
            0 means unlimited.
        default_score_threshold: Default threshold for predicate scores.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    evaluation_mode: RuleEvaluationMode = RuleEvaluationMode.ALL
    evidence_policy: EvidencePolicy = EvidencePolicy.MATCH_ONLY
    short_circuit_policy: ShortCircuitPolicy = ShortCircuitPolicy.COST_ASCENDING
    max_rules_per_evaluation: int = Field(default=0, ge=0)
    default_score_threshold: float = 0.5

    @field_validator("default_score_threshold")
    @classmethod
    def threshold_must_be_in_unit_range(cls, v: float) -> float:
        """Threshold must be in [0.0, 1.0]."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"default_score_threshold must be in [0.0, 1.0], got {v}")
        return v
