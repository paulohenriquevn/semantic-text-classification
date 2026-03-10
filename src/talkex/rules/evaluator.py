"""Rule evaluator — executes compiled AST rules against evaluation inputs.

The evaluator is the concrete implementation of the RuleEvaluator protocol.
It walks AST nodes, dispatches predicate evaluation to per-family handlers,
and produces RuleResult objects with traceable evidence.

Execution flow:

    RuleDefinition + RuleEvaluationInput + RuleEngineConfig
        ↓
    for each rule:
        ↓
        walk AST (AND/OR/NOT/Predicate)
        ↓
        dispatch predicate to handler by PredicateType
        ↓
        collect PredicateResult evidence
        ↓
        produce RuleResult
    ↓
    list[RuleResult]

Short-circuit:
    - AND: stops on first unmatched child (saves cost)
    - OR: stops on first matched child (saves cost)
    - COST_ASCENDING: reorders children by cost_hint before evaluation

Predicate families:
    - LEXICAL: keyword containment, regex matching
    - STRUCTURAL: field equality and numeric comparisons
    - CONTEXTUAL: repeated mentions, sequential occurrence
    - SEMANTIC: placeholder for embedding-based predicates (requires external scorer)
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from talkex.rules.ast import (
    AndNode,
    ASTNode,
    NotNode,
    OrNode,
    PredicateNode,
)
from talkex.rules.config import (
    EvidencePolicy,
    PredicateType,
    RuleEvaluationMode,
    ShortCircuitPolicy,
)
from talkex.rules.models import (
    PredicateResult,
    RuleDefinition,
    RuleEvaluationInput,
    RuleResult,
)

if TYPE_CHECKING:
    from talkex.models.rule_execution import RuleExecution
    from talkex.rules.config import RuleEngineConfig


# ---------------------------------------------------------------------------
# Predicate handlers
# ---------------------------------------------------------------------------


def _evaluate_predicate(
    node: PredicateNode,
    evaluation_input: RuleEvaluationInput,
) -> PredicateResult:
    """Dispatch predicate evaluation to the correct handler by type.

    Args:
        node: The predicate AST node.
        evaluation_input: The input to evaluate against.

    Returns:
        A PredicateResult with match outcome and evidence.
    """
    start = time.perf_counter()

    if node.predicate_type == PredicateType.LEXICAL:
        result = _evaluate_lexical(node, evaluation_input)
    elif node.predicate_type == PredicateType.STRUCTURAL:
        result = _evaluate_structural(node, evaluation_input)
    elif node.predicate_type == PredicateType.CONTEXTUAL:
        result = _evaluate_contextual(node, evaluation_input)
    elif node.predicate_type == PredicateType.SEMANTIC:
        result = _evaluate_semantic(node, evaluation_input)
    else:
        # Unknown type — fail open with no match
        result = PredicateResult(
            predicate_type=node.predicate_type,
            field_name=node.field_name,
            operator=node.operator,
            matched=False,
            score=0.0,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Attach execution time
    return PredicateResult(
        predicate_type=result.predicate_type,
        field_name=result.field_name,
        operator=result.operator,
        matched=result.matched,
        score=result.score,
        threshold=result.threshold,
        matched_text=result.matched_text,
        execution_time_ms=elapsed_ms,
        metadata=result.metadata,
    )


def _get_field_value(
    field_name: str,
    evaluation_input: RuleEvaluationInput,
) -> str | float | None:
    """Resolve a field name to its value from the evaluation input.

    Checks known fields first, then falls back to features and metadata.

    Args:
        field_name: The field name to resolve.
        evaluation_input: The input carrying values.

    Returns:
        The field value, or None if not found.
    """
    # Direct fields
    if field_name == "text":
        return evaluation_input.text
    if field_name == "speaker_role":
        return evaluation_input.speaker_role
    if field_name == "source_type":
        return evaluation_input.source_type
    if field_name == "source_id":
        return evaluation_input.source_id

    # Features (numeric)
    if field_name in evaluation_input.features:
        return evaluation_input.features[field_name]

    # Metadata (any)
    if field_name in evaluation_input.metadata:
        val = evaluation_input.metadata[field_name]
        if isinstance(val, str | int | float):
            return val

    return None


# -- Lexical predicates --


def _evaluate_lexical(
    node: PredicateNode,
    evaluation_input: RuleEvaluationInput,
) -> PredicateResult:
    """Evaluate a lexical predicate (contains, regex).

    Args:
        node: Lexical predicate node.
        evaluation_input: Input to evaluate.

    Returns:
        PredicateResult with match info and matched_text if found.
    """
    text_value = _get_field_value(node.field_name, evaluation_input)
    if text_value is None:
        text_value = ""
    text_str = str(text_value).lower()
    target = str(node.value).lower()

    if node.operator == "contains":
        matched = target in text_str
        return PredicateResult(
            predicate_type=node.predicate_type,
            field_name=node.field_name,
            operator=node.operator,
            matched=matched,
            score=1.0 if matched else 0.0,
            matched_text=str(node.value) if matched else None,
        )

    if node.operator == "contains_any":
        # Check if text contains any word from the list
        word_list = node.value if isinstance(node.value, list) else [str(node.value)]
        matched_words = [w for w in word_list if w.lower() in text_str]
        matched = len(matched_words) > 0
        return PredicateResult(
            predicate_type=node.predicate_type,
            field_name=node.field_name,
            operator=node.operator,
            matched=matched,
            score=len(matched_words) / len(word_list) if word_list else 0.0,
            matched_text=", ".join(matched_words) if matched else None,
            metadata={"matched_words": matched_words, "total_words": len(word_list)},
        )

    if node.operator == "regex":
        pattern = str(node.value)
        match = re.search(pattern, str(_get_field_value(node.field_name, evaluation_input) or ""))
        matched = match is not None
        return PredicateResult(
            predicate_type=node.predicate_type,
            field_name=node.field_name,
            operator=node.operator,
            matched=matched,
            score=1.0 if matched else 0.0,
            matched_text=match.group(0) if match else None,
        )

    # Fallback for unknown lexical operators
    return PredicateResult(
        predicate_type=node.predicate_type,
        field_name=node.field_name,
        operator=node.operator,
        matched=False,
        score=0.0,
    )


# -- Structural predicates --


def _evaluate_structural(
    node: PredicateNode,
    evaluation_input: RuleEvaluationInput,
) -> PredicateResult:
    """Evaluate a structural predicate (eq, gte, lte).

    Args:
        node: Structural predicate node.
        evaluation_input: Input to evaluate.

    Returns:
        PredicateResult with match outcome.
    """
    actual = _get_field_value(node.field_name, evaluation_input)

    if node.operator == "eq":
        matched = str(actual).lower() == str(node.value).lower() if actual is not None else False
        return PredicateResult(
            predicate_type=node.predicate_type,
            field_name=node.field_name,
            operator=node.operator,
            matched=matched,
            score=1.0 if matched else 0.0,
            matched_text=str(actual) if matched else None,
        )

    if node.operator == "neq":
        matched = str(actual).lower() != str(node.value).lower() if actual is not None else True
        return PredicateResult(
            predicate_type=node.predicate_type,
            field_name=node.field_name,
            operator=node.operator,
            matched=matched,
            score=1.0 if matched else 0.0,
            matched_text=str(actual) if matched else None,
        )

    if node.operator in ("gte", "lte", "gt", "lt"):
        if actual is None:
            return PredicateResult(
                predicate_type=node.predicate_type,
                field_name=node.field_name,
                operator=node.operator,
                matched=False,
                score=0.0,
            )
        try:
            actual_num = float(actual)
            target_num = float(node.value)
        except (ValueError, TypeError):
            return PredicateResult(
                predicate_type=node.predicate_type,
                field_name=node.field_name,
                operator=node.operator,
                matched=False,
                score=0.0,
            )

        if node.operator == "gte":
            matched = actual_num >= target_num
        elif node.operator == "lte":
            matched = actual_num <= target_num
        elif node.operator == "gt":
            matched = actual_num > target_num
        else:  # lt
            matched = actual_num < target_num

        return PredicateResult(
            predicate_type=node.predicate_type,
            field_name=node.field_name,
            operator=node.operator,
            matched=matched,
            score=1.0 if matched else 0.0,
            matched_text=str(actual_num) if matched else None,
        )

    # Fallback
    return PredicateResult(
        predicate_type=node.predicate_type,
        field_name=node.field_name,
        operator=node.operator,
        matched=False,
        score=0.0,
    )


# -- Contextual predicates --


def _evaluate_contextual(
    node: PredicateNode,
    evaluation_input: RuleEvaluationInput,
) -> PredicateResult:
    """Evaluate a contextual predicate (repeated_in_window, occurs_after).

    Args:
        node: Contextual predicate node.
        evaluation_input: Input to evaluate.

    Returns:
        PredicateResult with match outcome.
    """
    text = evaluation_input.text.lower()

    if node.operator == "repeated_in_window":
        target = str(node.value).lower()
        required_count = int(node.metadata.get("count", 1))
        actual_count = text.count(target)
        matched = actual_count >= required_count
        return PredicateResult(
            predicate_type=node.predicate_type,
            field_name=node.field_name,
            operator=node.operator,
            matched=matched,
            score=min(actual_count / required_count, 1.0) if required_count > 0 else 0.0,
            matched_text=f"{target} (x{actual_count})" if matched else None,
            metadata={"actual_count": actual_count, "required_count": required_count},
        )

    if node.operator == "occurs_after":
        first = str(node.value).lower()
        second = str(node.metadata.get("second", "")).lower()
        first_pos = text.find(first)
        second_pos = text.find(second) if second else -1
        matched = first_pos >= 0 and second_pos > first_pos
        return PredicateResult(
            predicate_type=node.predicate_type,
            field_name=node.field_name,
            operator=node.operator,
            matched=matched,
            score=1.0 if matched else 0.0,
            matched_text=f"{first} → {second}" if matched else None,
        )

    # Fallback
    return PredicateResult(
        predicate_type=node.predicate_type,
        field_name=node.field_name,
        operator=node.operator,
        matched=False,
        score=0.0,
    )


# -- Semantic predicates --


def _evaluate_semantic(
    node: PredicateNode,
    evaluation_input: RuleEvaluationInput,
) -> PredicateResult:
    """Evaluate a semantic predicate (intent score, embedding similarity).

    Semantic predicates check against pre-computed features in the evaluation
    input. Actual embedding similarity computation is NOT done here — that
    belongs in the embedding pipeline. The evaluator only reads scores that
    were already computed and attached to the input.

    Args:
        node: Semantic predicate node.
        evaluation_input: Input with pre-computed features.

    Returns:
        PredicateResult with score and threshold comparison.
    """
    # Look for the score in features
    score_value = evaluation_input.features.get(node.field_name, 0.0)
    threshold = node.threshold if node.threshold is not None else 0.5

    if node.operator in ("gte", "similarity_above"):
        matched = score_value >= threshold
    elif node.operator == "gt":
        matched = score_value > threshold
    elif node.operator == "lte":
        matched = score_value <= threshold
    elif node.operator == "lt":
        matched = score_value < threshold
    else:
        matched = False

    return PredicateResult(
        predicate_type=node.predicate_type,
        field_name=node.field_name,
        operator=node.operator,
        matched=matched,
        score=score_value,
        threshold=threshold,
        metadata={"label": str(node.value)} if node.value else {},
    )


# ---------------------------------------------------------------------------
# AST walker
# ---------------------------------------------------------------------------


def _collect_predicate_costs(node: ASTNode) -> list[tuple[int, int]]:
    """Collect (cost_hint, original_index) for direct children of AND/OR.

    Only collects from PredicateNode children. Composite children get
    a synthetic cost equal to the sum of their predicate costs.

    Args:
        node: An AND or OR node.

    Returns:
        List of (cost, index) tuples for sorting.
    """
    if not isinstance(node, AndNode | OrNode):
        return []

    costs: list[tuple[int, int]] = []
    for i, child in enumerate(node.children):
        if isinstance(child, PredicateNode):
            costs.append((child.cost_hint, i))
        else:
            # Composite nodes get a high cost to evaluate last
            costs.append((_subtree_max_cost(child), i))
    return costs


def _subtree_max_cost(node: ASTNode) -> int:
    """Compute the maximum cost_hint in a subtree."""
    if isinstance(node, PredicateNode):
        return node.cost_hint
    if isinstance(node, NotNode):
        return _subtree_max_cost(node.child)
    if isinstance(node, AndNode | OrNode):
        if not node.children:
            return 0
        return max(_subtree_max_cost(c) for c in node.children)
    return 0  # pragma: no cover


def _evaluate_ast(
    node: ASTNode,
    evaluation_input: RuleEvaluationInput,
    config: RuleEngineConfig,
    predicate_results: list[PredicateResult],
) -> tuple[bool, float]:
    """Recursively evaluate an AST node.

    Args:
        node: The AST node to evaluate.
        evaluation_input: Input to evaluate against.
        config: Engine configuration.
        predicate_results: Mutable list to collect predicate results.

    Returns:
        Tuple of (matched, score).
    """
    if isinstance(node, PredicateNode):
        result = _evaluate_predicate(node, evaluation_input)
        # Always collect evaluated predicates for tracking/observability.
        # Evidence policy is applied at the boundary (map_to_rule_execution).
        predicate_results.append(result)
        return result.matched, result.score

    if isinstance(node, AndNode):
        return _evaluate_and(node, evaluation_input, config, predicate_results)

    if isinstance(node, OrNode):
        return _evaluate_or(node, evaluation_input, config, predicate_results)

    if isinstance(node, NotNode):
        matched, score = _evaluate_ast(node.child, evaluation_input, config, predicate_results)
        return not matched, 1.0 - score

    return False, 0.0  # pragma: no cover


def _get_evaluation_order(
    node: AndNode | OrNode,
    config: RuleEngineConfig,
) -> list[int]:
    """Determine child evaluation order based on short-circuit policy.

    Args:
        node: AND or OR node.
        config: Engine configuration with policy.

    Returns:
        List of child indices in evaluation order.
    """
    n = len(node.children)
    if config.short_circuit_policy == ShortCircuitPolicy.DECLARATION:
        return list(range(n))

    if config.short_circuit_policy == ShortCircuitPolicy.COST_ASCENDING:
        costs = _collect_predicate_costs(node)
        costs.sort(key=lambda x: x[0])
        return [idx for _, idx in costs]

    # PRIORITY — for now, same as declaration (priority is a rule-level concept)
    return list(range(n))


def _evaluate_and(
    node: AndNode,
    evaluation_input: RuleEvaluationInput,
    config: RuleEngineConfig,
    predicate_results: list[PredicateResult],
) -> tuple[bool, float]:
    """Evaluate an AND node with short-circuit support.

    All children must match. Stops on first failure.

    Returns:
        Tuple of (all_matched, min_score).
    """
    order = _get_evaluation_order(node, config)
    scores: list[float] = []
    all_matched = True

    for idx in order:
        child = node.children[idx]
        matched, score = _evaluate_ast(child, evaluation_input, config, predicate_results)
        scores.append(score)
        if not matched:
            all_matched = False
            break  # short-circuit AND

    return all_matched, min(scores) if scores else 0.0


def _evaluate_or(
    node: OrNode,
    evaluation_input: RuleEvaluationInput,
    config: RuleEngineConfig,
    predicate_results: list[PredicateResult],
) -> tuple[bool, float]:
    """Evaluate an OR node with short-circuit support.

    At least one child must match. Stops on first success.

    Returns:
        Tuple of (any_matched, max_score).
    """
    order = _get_evaluation_order(node, config)
    scores: list[float] = []
    any_matched = False

    for idx in order:
        child = node.children[idx]
        matched, score = _evaluate_ast(child, evaluation_input, config, predicate_results)
        scores.append(score)
        if matched:
            any_matched = True
            break  # short-circuit OR

    return any_matched, max(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Rule evaluator
# ---------------------------------------------------------------------------


class SimpleRuleEvaluator:
    """Concrete RuleEvaluator — evaluates compiled rules against inputs.

    Satisfies the RuleEvaluator protocol via structural subtyping.

    Usage::

        evaluator = SimpleRuleEvaluator()
        results = evaluator.evaluate(
            rules=[rule_definition],
            evaluation_input=input_obj,
            config=RuleEngineConfig(),
        )
    """

    def evaluate(
        self,
        rules: list[RuleDefinition],
        evaluation_input: RuleEvaluationInput,
        config: RuleEngineConfig,
    ) -> list[RuleResult]:
        """Evaluate rules against an input.

        Args:
            rules: Compiled rule definitions.
            evaluation_input: The object to evaluate rules against.
            config: Evaluation configuration.

        Returns:
            List of RuleResult objects, one per rule evaluated.
        """
        results: list[RuleResult] = []

        for rule in rules:
            start = time.perf_counter()
            predicate_results: list[PredicateResult] = []

            matched, score = _evaluate_ast(rule.ast, evaluation_input, config, predicate_results)

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # Clamp score to [0, 1]
            score = max(0.0, min(1.0, score))

            total_evaluated = len(predicate_results)

            # Apply evidence policy to filter predicate results
            if config.evidence_policy == EvidencePolicy.MATCH_ONLY:
                predicate_results = [pr for pr in predicate_results if pr.matched]

            results.append(
                RuleResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    rule_version=rule.rule_version,
                    source_id=evaluation_input.source_id,
                    source_type=evaluation_input.source_type,
                    matched=matched,
                    score=score,
                    predicate_results=predicate_results,
                    execution_time_ms=elapsed_ms,
                    short_circuited=total_evaluated < _count_predicates(rule.ast),
                )
            )

            # Short-circuit at rule level: stop after first match in SHORT_CIRCUIT mode
            if config.evaluation_mode == RuleEvaluationMode.SHORT_CIRCUIT and matched:
                break

        return results


def _count_predicates(node: ASTNode) -> int:
    """Count total predicate nodes in an AST."""
    if isinstance(node, PredicateNode):
        return 1
    if isinstance(node, NotNode):
        return _count_predicates(node.child)
    if isinstance(node, AndNode | OrNode):
        return sum(_count_predicates(c) for c in node.children)
    return 0  # pragma: no cover


# ---------------------------------------------------------------------------
# Boundary mapping: RuleResult → RuleExecution
# ---------------------------------------------------------------------------


def map_to_rule_execution(
    result: RuleResult,
) -> RuleExecution:
    """Map an internal RuleResult to a domain RuleExecution entity.

    This is the boundary crossing from the rule engine subsystem to the
    domain model, parallel to ClassificationResult → Prediction.

    Args:
        result: Internal rule result from the evaluator.

    Returns:
        A domain RuleExecution entity with evidence.

    Raises:
        ValueError: If the RuleExecution fails domain validation.
    """
    from talkex.models.enums import ObjectType
    from talkex.models.rule_execution import (
        RuleExecution,
    )
    from talkex.models.types import RuleId

    # Map source_type string to ObjectType
    source_type_map = {
        "turn": ObjectType.TURN,
        "context_window": ObjectType.CONTEXT_WINDOW,
        "conversation": ObjectType.CONVERSATION,
    }
    object_type = source_type_map.get(result.source_type, ObjectType.CONTEXT_WINDOW)

    # Build evidence items from predicate results
    evidence = [pr.to_evidence_item() for pr in result.predicate_results if pr.matched]

    return RuleExecution(
        rule_id=RuleId(result.rule_id),
        rule_name=result.rule_name,
        source_id=result.source_id,
        source_type=object_type,
        matched=result.matched,
        score=result.score,
        execution_time_ms=result.execution_time_ms,
        evidence=evidence,
        metadata={
            "rule_version": result.rule_version,
            "short_circuited": result.short_circuited,
            "predicate_count": len(result.predicate_results),
        },
    )
