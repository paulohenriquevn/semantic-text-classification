"""Unit tests for rule evaluator — AST execution, predicate handlers, short-circuit.

Tests cover: lexical predicates (contains, regex), structural predicates (eq, gte, lte),
contextual predicates (repeated, occurs_after), semantic predicates, AND/OR/NOT evaluation,
short-circuit behavior, cost-based reordering, evidence policy, RuleResult → RuleExecution
mapping, and SimpleRuleEvaluator protocol compliance.
"""

from talkex.models.enums import ObjectType
from talkex.rules.compiler import SimpleRuleCompiler
from talkex.rules.config import (
    EvidencePolicy,
    PredicateType,
    RuleEngineConfig,
    RuleEvaluationMode,
    ShortCircuitPolicy,
)
from talkex.rules.evaluator import (
    SimpleRuleEvaluator,
    map_to_rule_execution,
)
from talkex.rules.models import (
    RuleDefinition,
    RuleEvaluationInput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(**overrides: object) -> RuleEvaluationInput:
    defaults: dict[str, object] = {
        "source_id": "win_001",
        "source_type": "context_window",
        "text": "I have a billing issue with my credit card",
    }
    defaults.update(overrides)
    return RuleEvaluationInput(**defaults)  # type: ignore[arg-type]


def _make_rule(dsl_text: str, **overrides: object) -> RuleDefinition:
    compiler = SimpleRuleCompiler()
    kwargs: dict[str, object] = {
        "rule_id": "rule_001",
        "rule_name": "test_rule",
    }
    kwargs.update(overrides)
    return compiler.compile(dsl_text, **kwargs)  # type: ignore[arg-type]


def _config(**overrides: object) -> RuleEngineConfig:
    return RuleEngineConfig(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Lexical predicates
# ---------------------------------------------------------------------------


class TestLexicalPredicates:
    def test_keyword_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert len(results) == 1
        assert results[0].matched is True
        assert results[0].score == 1.0

    def test_keyword_no_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("shipping")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is False
        assert results[0].score == 0.0

    def test_keyword_case_insensitive(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("BILLING")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is True

    def test_keyword_custom_field(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("speaker_role", "customer")')
        inp = _make_input(speaker_role="customer")
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is True

    def test_regex_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('regex("bill(ing|ed)")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is True
        pr = results[0].predicate_results[0]
        assert pr.matched_text == "billing"

    def test_regex_no_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('regex("cancel(led|lation)")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is False

    def test_keyword_matched_text(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        pr = results[0].predicate_results[0]
        assert pr.matched_text == "billing"


# ---------------------------------------------------------------------------
# Structural predicates
# ---------------------------------------------------------------------------


class TestStructuralPredicates:
    def test_speaker_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('speaker("customer")')
        inp = _make_input(speaker_role="customer")
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is True

    def test_speaker_no_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('speaker("agent")')
        inp = _make_input(speaker_role="customer")
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is False

    def test_speaker_none_no_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('speaker("customer")')
        inp = _make_input()  # no speaker_role
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is False

    def test_field_gte_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('field_gte("word_count", 5)')
        inp = _make_input(features={"word_count": 10.0})
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is True

    def test_field_gte_no_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('field_gte("word_count", 20)')
        inp = _make_input(features={"word_count": 10.0})
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is False

    def test_field_lte_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('field_lte("word_count", 15)')
        inp = _make_input(features={"word_count": 10.0})
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is True

    def test_field_eq_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('field_eq("channel", "voice")')
        inp = _make_input(metadata={"channel": "voice"})
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is True

    def test_field_missing_no_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('field_gte("nonexistent", 5)')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is False


# ---------------------------------------------------------------------------
# Contextual predicates
# ---------------------------------------------------------------------------


class TestContextualPredicates:
    def test_repeated_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('repeated("text", "billing", 2)')
        inp = _make_input(text="billing issue and another billing complaint")
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is True
        pr = results[0].predicate_results[0]
        assert pr.metadata["actual_count"] == 2

    def test_repeated_no_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('repeated("text", "billing", 5)')
        inp = _make_input(text="billing issue")
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is False

    def test_occurs_after_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('occurs_after("text", "complaint", "resolution")')
        inp = _make_input(text="First a complaint, then a resolution")
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is True

    def test_occurs_after_wrong_order(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('occurs_after("text", "complaint", "resolution")')
        inp = _make_input(text="resolution before complaint")
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is False

    def test_occurs_after_missing_word(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('occurs_after("text", "complaint", "resolution")')
        inp = _make_input(text="just a complaint")
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is False


# ---------------------------------------------------------------------------
# Semantic predicates
# ---------------------------------------------------------------------------


class TestSemanticPredicates:
    def test_intent_match_with_feature(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('intent("cancel", 0.7)')
        inp = _make_input(features={"intent_score": 0.85})
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is True
        pr = results[0].predicate_results[0]
        assert pr.score == 0.85
        assert pr.threshold == 0.7

    def test_intent_no_match_below_threshold(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('intent("cancel", 0.8)')
        inp = _make_input(features={"intent_score": 0.5})
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is False

    def test_intent_default_threshold(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('intent("cancel")')
        inp = _make_input(features={"intent_score": 0.6})
        # Default threshold is 0.5
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is True

    def test_intent_missing_feature_no_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('intent("cancel", 0.5)')
        inp = _make_input()  # no features
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].matched is False


# ---------------------------------------------------------------------------
# Logical operators (AND, OR, NOT)
# ---------------------------------------------------------------------------


class TestLogicalOperators:
    def test_and_all_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing") AND keyword("credit")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is True

    def test_and_partial_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing") AND keyword("shipping")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is False

    def test_or_first_matches(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing") OR keyword("shipping")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is True

    def test_or_second_matches(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("shipping") OR keyword("billing")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is True

    def test_or_none_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("shipping") OR keyword("returns")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is False

    def test_not_inverts_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('NOT keyword("shipping")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is True

    def test_not_inverts_no_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('NOT keyword("billing")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is False

    def test_complex_composition(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('(keyword("billing") OR keyword("invoice")) AND NOT keyword("shipping")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].matched is True

    def test_and_score_is_min(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing") AND keyword("credit")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        # Both match with 1.0, min is 1.0
        assert results[0].score == 1.0

    def test_or_score_is_max(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing") OR keyword("shipping")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].score == 1.0


# ---------------------------------------------------------------------------
# Short-circuit behavior
# ---------------------------------------------------------------------------


class TestShortCircuit:
    def test_and_stops_on_first_failure(self) -> None:
        evaluator = SimpleRuleEvaluator()
        # "shipping" fails, so "billing" should be skipped in cost-ascending
        # But since both have same cost_hint, declaration order applies
        rule = _make_rule('keyword("shipping") AND keyword("billing")')
        config = _config(
            short_circuit_policy=ShortCircuitPolicy.DECLARATION,
            evidence_policy=EvidencePolicy.ALWAYS,
        )
        results = evaluator.evaluate([rule], _make_input(), config)
        assert results[0].matched is False
        assert results[0].short_circuited is True
        # Only one predicate should have been evaluated (visible with ALWAYS policy)
        assert len(results[0].predicate_results) == 1

    def test_or_stops_on_first_match(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing") OR keyword("shipping")')
        config = _config(
            short_circuit_policy=ShortCircuitPolicy.DECLARATION,
            evidence_policy=EvidencePolicy.ALWAYS,
        )
        results = evaluator.evaluate([rule], _make_input(), config)
        assert results[0].matched is True
        assert results[0].short_circuited is True
        assert len(results[0].predicate_results) == 1

    def test_rule_level_short_circuit(self) -> None:
        evaluator = SimpleRuleEvaluator()
        r1 = _make_rule('keyword("billing")', rule_id="r1", rule_name="r1")
        r2 = _make_rule('keyword("credit")', rule_id="r2", rule_name="r2")
        config = _config(evaluation_mode=RuleEvaluationMode.SHORT_CIRCUIT)
        results = evaluator.evaluate([r1, r2], _make_input(), config)
        # Should stop after first match
        assert len(results) == 1
        assert results[0].rule_id == "r1"

    def test_all_mode_evaluates_all_rules(self) -> None:
        evaluator = SimpleRuleEvaluator()
        r1 = _make_rule('keyword("billing")', rule_id="r1", rule_name="r1")
        r2 = _make_rule('keyword("credit")', rule_id="r2", rule_name="r2")
        config = _config(evaluation_mode=RuleEvaluationMode.ALL)
        results = evaluator.evaluate([r1, r2], _make_input(), config)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Cost-based reordering
# ---------------------------------------------------------------------------


class TestCostReordering:
    def test_cost_ascending_evaluates_cheap_first(self) -> None:
        """With COST_ASCENDING, lexical (cost 1) should be evaluated before semantic (cost 4)."""
        evaluator = SimpleRuleEvaluator()
        # intent has cost 4, keyword has cost 1
        # In an AND where keyword fails, intent should be skipped
        rule = _make_rule('intent("cancel", 0.8) AND keyword("shipping")')
        inp = _make_input(features={"intent_score": 0.9})
        config = _config(
            short_circuit_policy=ShortCircuitPolicy.COST_ASCENDING,
            evidence_policy=EvidencePolicy.ALWAYS,
        )
        results = evaluator.evaluate([rule], inp, config)
        assert results[0].matched is False
        # keyword (cheap) should have been evaluated first and failed
        assert len(results[0].predicate_results) == 1
        pr = results[0].predicate_results[0]
        assert pr.predicate_type == PredicateType.LEXICAL

    def test_declaration_preserves_order(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('intent("cancel", 0.8) AND keyword("shipping")')
        inp = _make_input(features={"intent_score": 0.5})
        config = _config(
            short_circuit_policy=ShortCircuitPolicy.DECLARATION,
            evidence_policy=EvidencePolicy.ALWAYS,
        )
        results = evaluator.evaluate([rule], inp, config)
        # intent evaluated first (declaration order), fails
        assert results[0].matched is False
        pr = results[0].predicate_results[0]
        assert pr.predicate_type == PredicateType.SEMANTIC


# ---------------------------------------------------------------------------
# Evidence policy
# ---------------------------------------------------------------------------


class TestEvidencePolicy:
    def test_match_only_collects_matched_predicates(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing") OR keyword("shipping")')
        config = _config(
            evidence_policy=EvidencePolicy.MATCH_ONLY,
            short_circuit_policy=ShortCircuitPolicy.DECLARATION,
        )
        results = evaluator.evaluate([rule], _make_input(), config)
        # OR short-circuits on first match, and MATCH_ONLY only collects matched
        assert all(pr.matched for pr in results[0].predicate_results)

    def test_always_collects_all_predicates(self) -> None:
        evaluator = SimpleRuleEvaluator()
        # In AND with DECLARATION, both get evaluated if first matches
        rule = _make_rule('keyword("billing") AND keyword("shipping")')
        config = _config(
            evidence_policy=EvidencePolicy.ALWAYS,
            short_circuit_policy=ShortCircuitPolicy.DECLARATION,
        )
        results = evaluator.evaluate([rule], _make_input(), config)
        # billing matches, shipping doesn't — AND short-circuits after shipping fails
        # With ALWAYS, both should be in evidence
        assert len(results[0].predicate_results) == 2


# ---------------------------------------------------------------------------
# Execution metadata
# ---------------------------------------------------------------------------


class TestExecutionMetadata:
    def test_result_has_execution_time(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].execution_time_ms >= 0.0

    def test_result_has_rule_identity(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")', rule_id="r42", rule_name="billing_rule")
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert results[0].rule_id == "r42"
        assert results[0].rule_name == "billing_rule"

    def test_result_has_source_identity(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")')
        inp = _make_input(source_id="win_042", source_type="conversation")
        results = evaluator.evaluate([rule], inp, _config())
        assert results[0].source_id == "win_042"
        assert results[0].source_type == "conversation"

    def test_predicate_result_has_execution_time(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        pr = results[0].predicate_results[0]
        assert pr.execution_time_ms >= 0.0

    def test_score_clamped_to_unit(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert 0.0 <= results[0].score <= 1.0


# ---------------------------------------------------------------------------
# Boundary mapping: RuleResult → RuleExecution
# ---------------------------------------------------------------------------


class TestMapToRuleExecution:
    def test_matched_rule_produces_execution(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        execution = map_to_rule_execution(results[0])
        assert execution.matched is True
        assert execution.rule_id == "rule_001"
        assert execution.rule_name == "test_rule"
        assert execution.source_type == ObjectType.CONTEXT_WINDOW

    def test_matched_rule_has_evidence(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        execution = map_to_rule_execution(results[0])
        assert len(execution.evidence) >= 1
        assert execution.evidence[0]["predicate_type"] == "lexical"

    def test_unmatched_rule_has_no_evidence(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("shipping")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        execution = map_to_rule_execution(results[0])
        assert execution.matched is False
        assert len(execution.evidence) == 0

    def test_execution_has_metadata(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        execution = map_to_rule_execution(results[0])
        assert "rule_version" in execution.metadata
        assert "short_circuited" in execution.metadata

    def test_source_type_mapping(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")')
        for src, expected in [
            ("turn", ObjectType.TURN),
            ("context_window", ObjectType.CONTEXT_WINDOW),
            ("conversation", ObjectType.CONVERSATION),
        ]:
            inp = _make_input(source_type=src)
            results = evaluator.evaluate([rule], inp, _config())
            execution = map_to_rule_execution(results[0])
            assert execution.source_type == expected

    def test_execution_score_in_unit_range(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        execution = map_to_rule_execution(results[0])
        assert 0.0 <= execution.score <= 1.0


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestEvaluatorProtocolCompliance:
    def test_satisfies_rule_evaluator_protocol(self) -> None:
        evaluator = SimpleRuleEvaluator()
        rule = _make_rule('keyword("billing")')
        results = evaluator.evaluate([rule], _make_input(), _config())
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestEvaluatorReexport:
    def test_simple_rule_evaluator_from_rules_package(self) -> None:
        from talkex.rules import SimpleRuleEvaluator as Imported

        assert Imported is SimpleRuleEvaluator

    def test_map_to_rule_execution_from_rules_package(self) -> None:
        from talkex.rules import map_to_rule_execution as imported

        assert imported is map_to_rule_execution
