"""Rule engine AST — typed Abstract Syntax Tree nodes for rule composition.

The AST represents compiled rules as a tree of composable nodes.
Leaf nodes are predicates (concrete checks). Internal nodes are
logical combinators (AND, OR, NOT).

Node types:
    PredicateNode: leaf — a concrete check against the input
    AndNode:       all children must match
    OrNode:        at least one child must match
    NotNode:       child must NOT match

Predicate families (from PredicateType):
    LEXICAL:     keyword matching, regex, BM25 scoring
    SEMANTIC:    embedding similarity, intent scores
    STRUCTURAL:  speaker role, channel, turn count
    CONTEXTUAL:  cross-turn patterns, occurrence sequences

Design decisions:
    - Frozen dataclasses: lightweight, composable, immutable
    - No ABC inheritance: nodes are discriminated by type, not by method
    - PredicateNode carries operator + value for the evaluator to interpret
    - Cost hint enables cost-ascending short-circuit optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from talkex.rules.config import PredicateType

# ---------------------------------------------------------------------------
# Leaf node: predicate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredicateNode:
    """A leaf AST node representing a concrete predicate check.

    Args:
        predicate_type: Signal family (lexical, semantic, structural, contextual).
        field_name: The field or signal to check (e.g., "text", "speaker_role",
            "intent_score", "embedding_similarity").
        operator: Comparison operator (e.g., "contains", "regex", "gte", "eq",
            "similarity_above", "occurs_after").
        value: Reference value for comparison. Type depends on operator.
        threshold: Optional score threshold for numeric predicates.
        cost_hint: Relative evaluation cost (lower = cheaper). Used by
            COST_ASCENDING short-circuit policy. Default costs:
            lexical=1, structural=2, contextual=3, semantic=4.
        metadata: Additional predicate parameters (e.g., model_name for
            semantic predicates, window_size for contextual).
    """

    predicate_type: PredicateType
    field_name: str
    operator: str
    value: Any
    threshold: float | None = None
    cost_hint: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Composite nodes: logical combinators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AndNode:
    """All children must match for this node to match.

    Short-circuit: if any child fails, remaining children are skipped.

    Args:
        children: Ordered list of child nodes (predicate or composite).
    """

    children: list[PredicateNode | AndNode | OrNode | NotNode]


@dataclass(frozen=True)
class OrNode:
    """At least one child must match for this node to match.

    Short-circuit: if any child matches, remaining children are skipped.

    Args:
        children: Ordered list of child nodes (predicate or composite).
    """

    children: list[PredicateNode | AndNode | OrNode | NotNode]


@dataclass(frozen=True)
class NotNode:
    """Child must NOT match for this node to match.

    Inverts the result of the child node.

    Args:
        child: Single child node to negate.
    """

    child: PredicateNode | AndNode | OrNode | NotNode


# Type alias for any AST node
ASTNode = PredicateNode | AndNode | OrNode | NotNode
"""Union type for all AST node types."""
