"""Rule engine operational metrics — pure functions over rule evaluation results.

All functions take a list of RuleResult objects and return aggregated
operational metrics. These measure engine behavior, NOT classification
quality (which is handled by classification_eval/).

Supported metrics:
    match_rate:             fraction of rules that matched
    short_circuit_rate:     fraction of rules that short-circuited
    predicate_eval_rate:    fraction of predicates actually evaluated (vs total)
    avg_execution_time_ms:  mean rule evaluation latency
    predicate_type_distribution: count of predicates by family

All functions are deterministic and stateless.
"""

from __future__ import annotations

from talkex.rules.config import PredicateType
from talkex.rules.models import RuleResult


def match_rate(results: list[RuleResult]) -> float:
    """Fraction of rules that matched.

    Args:
        results: Rule evaluation results.

    Returns:
        Match rate in [0.0, 1.0]. Returns 0.0 if results is empty.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.matched) / len(results)


def short_circuit_rate(results: list[RuleResult]) -> float:
    """Fraction of rules that short-circuited evaluation.

    Args:
        results: Rule evaluation results.

    Returns:
        Short-circuit rate in [0.0, 1.0]. Returns 0.0 if results is empty.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.short_circuited) / len(results)


def avg_execution_time_ms(results: list[RuleResult]) -> float:
    """Mean rule evaluation latency in milliseconds.

    Args:
        results: Rule evaluation results.

    Returns:
        Average execution time. Returns 0.0 if results is empty.
    """
    if not results:
        return 0.0
    return sum(r.execution_time_ms for r in results) / len(results)


def total_predicates_evaluated(results: list[RuleResult]) -> int:
    """Total count of predicate evaluations across all rules.

    Args:
        results: Rule evaluation results.

    Returns:
        Sum of predicate_results lengths.
    """
    return sum(len(r.predicate_results) for r in results)


def total_predicates_matched(results: list[RuleResult]) -> int:
    """Total count of matched predicates across all rules.

    Args:
        results: Rule evaluation results.

    Returns:
        Count of matched predicates.
    """
    return sum(1 for r in results for pr in r.predicate_results if pr.matched)


def predicate_type_distribution(
    results: list[RuleResult],
) -> dict[str, int]:
    """Count of evaluated predicates by predicate type (family).

    Args:
        results: Rule evaluation results.

    Returns:
        Mapping from PredicateType value to count.
    """
    dist: dict[str, int] = {pt.value: 0 for pt in PredicateType}
    for r in results:
        for pr in r.predicate_results:
            dist[pr.predicate_type.value] = dist.get(pr.predicate_type.value, 0) + 1
    return dist


def compute_rule_metrics(results: list[RuleResult]) -> dict[str, object]:
    """Compute all operational metrics for a batch of rule results.

    Convenience function that calls all individual metric functions
    and returns a single aggregated dictionary.

    Args:
        results: Rule evaluation results.

    Returns:
        Dictionary with all operational metrics.
    """
    return {
        "rule_count": len(results),
        "rules_matched": sum(1 for r in results if r.matched),
        "match_rate": round(match_rate(results), 4),
        "short_circuit_count": sum(1 for r in results if r.short_circuited),
        "short_circuit_rate": round(short_circuit_rate(results), 4),
        "predicates_evaluated": total_predicates_evaluated(results),
        "predicates_matched": total_predicates_matched(results),
        "avg_execution_time_ms": round(avg_execution_time_ms(results), 4),
        "total_execution_time_ms": round(sum(r.execution_time_ms for r in results), 4),
        "predicate_type_distribution": predicate_type_distribution(results),
    }
