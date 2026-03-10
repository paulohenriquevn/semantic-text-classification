"""Rule compiler — compiles DSL text into RuleDefinition with validated AST.

The compiler is the concrete implementation of the RuleCompiler protocol.
It delegates parsing to ``parser.parse_dsl()`` and adds semantic validation
to ensure the resulting AST is well-formed and safe to evaluate.

Pipeline:

    DSL text → tokenize → parse → AST → semantic validation → RuleDefinition

Semantic validation checks:
    - No empty AND/OR nodes (degenerate trees)
    - No deeply nested trees beyond max_depth (protection against abuse)
    - Threshold values in valid range [0.0, 1.0] where present
    - Cost hints are positive integers
"""

from __future__ import annotations

from semantic_conversation_engine.exceptions import RuleError
from semantic_conversation_engine.rules.ast import (
    AndNode,
    ASTNode,
    NotNode,
    OrNode,
    PredicateNode,
)
from semantic_conversation_engine.rules.models import RuleDefinition
from semantic_conversation_engine.rules.parser import parse_dsl

# Maximum AST depth to prevent abuse or stack overflow during evaluation
_MAX_AST_DEPTH = 20


# ---------------------------------------------------------------------------
# Semantic validation
# ---------------------------------------------------------------------------


def _validate_ast(node: ASTNode, depth: int = 0) -> None:
    """Recursively validate semantic constraints on an AST.

    Args:
        node: The AST node to validate.
        depth: Current recursion depth.

    Raises:
        RuleError: If the AST violates semantic constraints.
    """
    if depth > _MAX_AST_DEPTH:
        raise RuleError(f"AST exceeds maximum depth of {_MAX_AST_DEPTH}. Simplify the rule expression.")

    if isinstance(node, PredicateNode):
        _validate_predicate(node)
    elif isinstance(node, AndNode | OrNode):
        node_type = "AND" if isinstance(node, AndNode) else "OR"
        if not node.children:
            raise RuleError(f"Empty {node_type} node — must have at least one child")
        for child in node.children:
            _validate_ast(child, depth + 1)
    elif isinstance(node, NotNode):
        _validate_ast(node.child, depth + 1)
    else:
        raise RuleError(f"Unknown AST node type: {type(node).__name__}")  # pragma: no cover


def _validate_predicate(node: PredicateNode) -> None:
    """Validate semantic constraints on a PredicateNode.

    Args:
        node: The predicate to validate.

    Raises:
        RuleError: If the predicate has invalid values.
    """
    if node.threshold is not None and (node.threshold < 0.0 or node.threshold > 1.0):
        raise RuleError(f"Threshold must be in [0.0, 1.0], got {node.threshold} for predicate on '{node.field_name}'")
    if node.cost_hint < 1:
        raise RuleError(f"cost_hint must be >= 1, got {node.cost_hint} for predicate on '{node.field_name}'")


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


class SimpleRuleCompiler:
    """Concrete RuleCompiler — compiles DSL text into RuleDefinition.

    Satisfies the RuleCompiler protocol via structural subtyping.

    Usage::

        compiler = SimpleRuleCompiler()
        rule = compiler.compile(
            dsl_text='keyword("text", "billing") AND speaker("customer")',
            rule_id="rule_001",
            rule_name="billing_detection",
        )
    """

    def compile(
        self,
        dsl_text: str,
        rule_id: str,
        rule_name: str,
        *,
        rule_version: str = "1.0",
        description: str = "",
        priority: int = 0,
        tags: list[str] | None = None,
    ) -> RuleDefinition:
        """Compile a DSL text into a RuleDefinition.

        Args:
            dsl_text: Rule expression in DSL syntax.
            rule_id: Unique identifier for the compiled rule.
            rule_name: Human-readable name for the compiled rule.
            rule_version: Version string. Defaults to "1.0".
            description: Human-readable description. If empty, auto-generated
                from DSL text.
            priority: Evaluation priority (higher = first in PRIORITY mode).
            tags: Categorical tags for filtering.

        Returns:
            A RuleDefinition with a validated AST.

        Raises:
            RuleError: If the DSL text is syntactically or semantically invalid.
        """
        ast = parse_dsl(dsl_text)
        _validate_ast(ast)

        final_description = description if description else f"Compiled from: {dsl_text.strip()}"

        return RuleDefinition(
            rule_id=rule_id,
            rule_name=rule_name,
            rule_version=rule_version,
            description=final_description,
            ast=ast,
            priority=priority,
            tags=tags if tags is not None else [],
        )
