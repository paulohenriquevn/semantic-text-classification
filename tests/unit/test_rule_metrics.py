"""Unit tests for rule engine operational metrics.

Tests cover: match_rate, short_circuit_rate, avg_execution_time_ms,
total_predicates_evaluated/matched, predicate_type_distribution,
compute_rule_metrics, and edge cases (empty input).
"""

from semantic_conversation_engine.rules.config import PredicateType
from semantic_conversation_engine.rules.metrics import (
    avg_execution_time_ms,
    compute_rule_metrics,
    match_rate,
    predicate_type_distribution,
    short_circuit_rate,
    total_predicates_evaluated,
    total_predicates_matched,
)
from semantic_conversation_engine.rules.models import PredicateResult, RuleResult


def _pr(
    *,
    matched: bool = True,
    predicate_type: PredicateType = PredicateType.LEXICAL,
    score: float = 1.0,
) -> PredicateResult:
    return PredicateResult(
        predicate_type=predicate_type,
        field_name="text",
        operator="contains",
        matched=matched,
        score=score,
    )


def _rr(
    *,
    matched: bool = True,
    short_circuited: bool = False,
    execution_time_ms: float = 1.0,
    predicate_results: list[PredicateResult] | None = None,
    rule_id: str = "r1",
) -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        rule_name=f"rule_{rule_id}",
        rule_version="1.0",
        source_id="src_001",
        source_type="context_window",
        matched=matched,
        score=1.0 if matched else 0.0,
        predicate_results=predicate_results or [],
        execution_time_ms=execution_time_ms,
        short_circuited=short_circuited,
    )


# ---------------------------------------------------------------------------
# match_rate
# ---------------------------------------------------------------------------


class TestMatchRate:
    def test_all_matched(self) -> None:
        results = [_rr(matched=True), _rr(matched=True)]
        assert match_rate(results) == 1.0

    def test_none_matched(self) -> None:
        results = [_rr(matched=False), _rr(matched=False)]
        assert match_rate(results) == 0.0

    def test_partial_match(self) -> None:
        results = [_rr(matched=True), _rr(matched=False), _rr(matched=True)]
        assert abs(match_rate(results) - 2.0 / 3.0) < 1e-9

    def test_empty(self) -> None:
        assert match_rate([]) == 0.0


# ---------------------------------------------------------------------------
# short_circuit_rate
# ---------------------------------------------------------------------------


class TestShortCircuitRate:
    def test_all_short_circuited(self) -> None:
        results = [_rr(short_circuited=True), _rr(short_circuited=True)]
        assert short_circuit_rate(results) == 1.0

    def test_none_short_circuited(self) -> None:
        results = [_rr(short_circuited=False), _rr(short_circuited=False)]
        assert short_circuit_rate(results) == 0.0

    def test_empty(self) -> None:
        assert short_circuit_rate([]) == 0.0


# ---------------------------------------------------------------------------
# avg_execution_time_ms
# ---------------------------------------------------------------------------


class TestAvgExecutionTime:
    def test_average(self) -> None:
        results = [_rr(execution_time_ms=2.0), _rr(execution_time_ms=4.0)]
        assert avg_execution_time_ms(results) == 3.0

    def test_empty(self) -> None:
        assert avg_execution_time_ms([]) == 0.0


# ---------------------------------------------------------------------------
# total_predicates_evaluated / matched
# ---------------------------------------------------------------------------


class TestPredicateCounts:
    def test_total_evaluated(self) -> None:
        results = [
            _rr(predicate_results=[_pr(), _pr()]),
            _rr(predicate_results=[_pr()]),
        ]
        assert total_predicates_evaluated(results) == 3

    def test_total_matched(self) -> None:
        results = [
            _rr(predicate_results=[_pr(matched=True), _pr(matched=False)]),
            _rr(predicate_results=[_pr(matched=True)]),
        ]
        assert total_predicates_matched(results) == 2

    def test_empty(self) -> None:
        assert total_predicates_evaluated([]) == 0
        assert total_predicates_matched([]) == 0


# ---------------------------------------------------------------------------
# predicate_type_distribution
# ---------------------------------------------------------------------------


class TestPredicateTypeDistribution:
    def test_distribution(self) -> None:
        results = [
            _rr(
                predicate_results=[
                    _pr(predicate_type=PredicateType.LEXICAL),
                    _pr(predicate_type=PredicateType.SEMANTIC),
                    _pr(predicate_type=PredicateType.LEXICAL),
                ]
            ),
        ]
        dist = predicate_type_distribution(results)
        assert dist["lexical"] == 2
        assert dist["semantic"] == 1
        assert dist["structural"] == 0
        assert dist["contextual"] == 0

    def test_empty(self) -> None:
        dist = predicate_type_distribution([])
        assert all(v == 0 for v in dist.values())


# ---------------------------------------------------------------------------
# compute_rule_metrics
# ---------------------------------------------------------------------------


class TestComputeRuleMetrics:
    def test_aggregated_metrics(self) -> None:
        results = [
            _rr(
                matched=True,
                short_circuited=False,
                execution_time_ms=2.0,
                predicate_results=[
                    _pr(matched=True, predicate_type=PredicateType.LEXICAL),
                    _pr(matched=True, predicate_type=PredicateType.STRUCTURAL),
                ],
            ),
            _rr(
                matched=False,
                short_circuited=True,
                execution_time_ms=1.0,
                predicate_results=[
                    _pr(matched=False, predicate_type=PredicateType.LEXICAL),
                ],
            ),
        ]
        m = compute_rule_metrics(results)
        assert m["rule_count"] == 2
        assert m["rules_matched"] == 1
        assert m["match_rate"] == 0.5
        assert m["short_circuit_count"] == 1
        assert m["short_circuit_rate"] == 0.5
        assert m["predicates_evaluated"] == 3
        assert m["predicates_matched"] == 2
        assert m["avg_execution_time_ms"] == 1.5
        assert isinstance(m["predicate_type_distribution"], dict)

    def test_empty(self) -> None:
        m = compute_rule_metrics([])
        assert m["rule_count"] == 0
        assert m["match_rate"] == 0.0


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestMetricsReexport:
    def test_importable_from_rules_package(self) -> None:
        from semantic_conversation_engine.rules import compute_rule_metrics as fn

        assert callable(fn)
