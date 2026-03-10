"""Unit tests for rule engine internal models.

Tests cover: RuleDefinition, RuleEvaluationInput, PredicateResult,
RuleResult — construction, defaults, immutability, evidence mapping, reexport.
"""

from talkex.rules.ast import AndNode, PredicateNode
from talkex.rules.config import PredicateType
from talkex.rules.models import (
    PredicateResult,
    RuleDefinition,
    RuleEvaluationInput,
    RuleResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_predicate_node() -> PredicateNode:
    return PredicateNode(
        predicate_type=PredicateType.LEXICAL,
        field_name="text",
        operator="contains",
        value="billing",
    )


def _make_rule_definition(**overrides: object) -> RuleDefinition:
    defaults: dict[str, object] = {
        "rule_id": "rule_001",
        "rule_name": "billing_detection",
        "rule_version": "1.0",
        "description": "Detects billing-related conversations",
        "ast": _make_predicate_node(),
    }
    defaults.update(overrides)
    return RuleDefinition(**defaults)  # type: ignore[arg-type]


def _make_evaluation_input(**overrides: object) -> RuleEvaluationInput:
    defaults: dict[str, object] = {
        "source_id": "win_001",
        "source_type": "context_window",
        "text": "I have a billing issue with my credit card",
    }
    defaults.update(overrides)
    return RuleEvaluationInput(**defaults)  # type: ignore[arg-type]


def _make_predicate_result(**overrides: object) -> PredicateResult:
    defaults: dict[str, object] = {
        "predicate_type": PredicateType.LEXICAL,
        "field_name": "text",
        "operator": "contains",
        "matched": True,
        "score": 1.0,
        "threshold": 0.5,
        "matched_text": "billing",
    }
    defaults.update(overrides)
    return PredicateResult(**defaults)  # type: ignore[arg-type]


def _make_rule_result(**overrides: object) -> RuleResult:
    defaults: dict[str, object] = {
        "rule_id": "rule_001",
        "rule_name": "billing_detection",
        "rule_version": "1.0",
        "source_id": "win_001",
        "source_type": "context_window",
        "matched": True,
        "score": 0.95,
        "predicate_results": [_make_predicate_result()],
    }
    defaults.update(overrides)
    return RuleResult(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RuleDefinition
# ---------------------------------------------------------------------------


class TestRuleDefinition:
    def test_construction(self) -> None:
        rd = _make_rule_definition()
        assert rd.rule_id == "rule_001"
        assert rd.rule_name == "billing_detection"
        assert rd.rule_version == "1.0"
        assert rd.description == "Detects billing-related conversations"
        assert isinstance(rd.ast, PredicateNode)

    def test_default_priority(self) -> None:
        rd = _make_rule_definition()
        assert rd.priority == 0

    def test_custom_priority(self) -> None:
        rd = _make_rule_definition(priority=10)
        assert rd.priority == 10

    def test_default_tags_empty(self) -> None:
        rd = _make_rule_definition()
        assert rd.tags == []

    def test_custom_tags(self) -> None:
        rd = _make_rule_definition(tags=["compliance", "fraud"])
        assert "compliance" in rd.tags
        assert "fraud" in rd.tags

    def test_default_metadata_empty(self) -> None:
        rd = _make_rule_definition()
        assert rd.metadata == {}

    def test_composite_ast(self) -> None:
        p1 = _make_predicate_node()
        p2 = PredicateNode(
            predicate_type=PredicateType.SEMANTIC,
            field_name="intent_score",
            operator="gte",
            value=0.8,
        )
        ast = AndNode(children=[p1, p2])
        rd = _make_rule_definition(ast=ast)
        assert isinstance(rd.ast, AndNode)

    def test_frozen(self) -> None:
        rd = _make_rule_definition()
        try:
            rd.rule_name = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# RuleEvaluationInput
# ---------------------------------------------------------------------------


class TestRuleEvaluationInput:
    def test_construction(self) -> None:
        inp = _make_evaluation_input()
        assert inp.source_id == "win_001"
        assert inp.source_type == "context_window"
        assert "billing" in inp.text

    def test_default_embedding_none(self) -> None:
        inp = _make_evaluation_input()
        assert inp.embedding is None

    def test_with_embedding(self) -> None:
        inp = _make_evaluation_input(embedding=[0.1, 0.2, 0.3])
        assert inp.embedding == [0.1, 0.2, 0.3]

    def test_default_features_empty(self) -> None:
        inp = _make_evaluation_input()
        assert inp.features == {}

    def test_with_features(self) -> None:
        inp = _make_evaluation_input(features={"word_count": 10.0})
        assert inp.features["word_count"] == 10.0

    def test_default_speaker_role_none(self) -> None:
        inp = _make_evaluation_input()
        assert inp.speaker_role is None

    def test_with_speaker_role(self) -> None:
        inp = _make_evaluation_input(speaker_role="customer")
        assert inp.speaker_role == "customer"

    def test_default_metadata_empty(self) -> None:
        inp = _make_evaluation_input()
        assert inp.metadata == {}

    def test_frozen(self) -> None:
        inp = _make_evaluation_input()
        try:
            inp.text = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# PredicateResult
# ---------------------------------------------------------------------------


class TestPredicateResult:
    def test_construction(self) -> None:
        pr = _make_predicate_result()
        assert pr.predicate_type == PredicateType.LEXICAL
        assert pr.field_name == "text"
        assert pr.operator == "contains"
        assert pr.matched is True
        assert pr.score == 1.0
        assert pr.matched_text == "billing"

    def test_default_score_zero(self) -> None:
        pr = PredicateResult(
            predicate_type=PredicateType.STRUCTURAL,
            field_name="speaker_role",
            operator="eq",
            matched=True,
        )
        assert pr.score == 0.0

    def test_default_threshold_zero(self) -> None:
        pr = PredicateResult(
            predicate_type=PredicateType.STRUCTURAL,
            field_name="speaker_role",
            operator="eq",
            matched=True,
        )
        assert pr.threshold == 0.0

    def test_default_matched_text_none(self) -> None:
        pr = PredicateResult(
            predicate_type=PredicateType.STRUCTURAL,
            field_name="speaker_role",
            operator="eq",
            matched=True,
        )
        assert pr.matched_text is None

    def test_default_execution_time_zero(self) -> None:
        pr = PredicateResult(
            predicate_type=PredicateType.LEXICAL,
            field_name="text",
            operator="contains",
            matched=False,
        )
        assert pr.execution_time_ms == 0.0

    def test_frozen(self) -> None:
        pr = _make_predicate_result()
        try:
            pr.matched = False  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# PredicateResult.to_evidence_item
# ---------------------------------------------------------------------------


class TestPredicateResultEvidence:
    def test_to_evidence_item_with_matched_text(self) -> None:
        pr = _make_predicate_result()
        evidence = pr.to_evidence_item()
        assert evidence["predicate_type"] == "lexical"
        assert evidence["score"] == 1.0
        assert evidence["threshold"] == 0.5
        assert evidence["matched_text"] == "billing"

    def test_to_evidence_item_without_matched_text(self) -> None:
        pr = PredicateResult(
            predicate_type=PredicateType.STRUCTURAL,
            field_name="speaker_role",
            operator="eq",
            matched=True,
            score=1.0,
        )
        evidence = pr.to_evidence_item()
        assert "matched_text" not in evidence

    def test_to_evidence_item_with_metadata(self) -> None:
        pr = _make_predicate_result(metadata={"model_name": "e5-base"})
        evidence = pr.to_evidence_item()
        assert evidence["metadata"]["model_name"] == "e5-base"

    def test_to_evidence_item_without_metadata(self) -> None:
        pr = _make_predicate_result()
        evidence = pr.to_evidence_item()
        assert "metadata" not in evidence


# ---------------------------------------------------------------------------
# RuleResult
# ---------------------------------------------------------------------------


class TestRuleResult:
    def test_construction(self) -> None:
        rr = _make_rule_result()
        assert rr.rule_id == "rule_001"
        assert rr.rule_name == "billing_detection"
        assert rr.matched is True
        assert rr.score == 0.95
        assert len(rr.predicate_results) == 1

    def test_default_execution_time_zero(self) -> None:
        rr = _make_rule_result()
        assert rr.execution_time_ms == 0.0

    def test_default_short_circuited_false(self) -> None:
        rr = _make_rule_result()
        assert rr.short_circuited is False

    def test_short_circuited_true(self) -> None:
        rr = _make_rule_result(short_circuited=True)
        assert rr.short_circuited is True

    def test_default_metadata_empty(self) -> None:
        rr = _make_rule_result()
        assert rr.metadata == {}

    def test_unmatched_rule(self) -> None:
        pr = _make_predicate_result(matched=False, score=0.2)
        rr = _make_rule_result(matched=False, score=0.2, predicate_results=[pr])
        assert rr.matched is False
        assert rr.score == 0.2

    def test_multiple_predicate_results(self) -> None:
        pr1 = _make_predicate_result()
        pr2 = PredicateResult(
            predicate_type=PredicateType.SEMANTIC,
            field_name="intent_score",
            operator="gte",
            matched=True,
            score=0.85,
            threshold=0.7,
        )
        rr = _make_rule_result(predicate_results=[pr1, pr2])
        assert len(rr.predicate_results) == 2

    def test_frozen(self) -> None:
        rr = _make_rule_result()
        try:
            rr.matched = False  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestModelsReexport:
    def test_importable_from_rules_package(self) -> None:
        from talkex.rules import (
            PredicateResult as PR,
        )
        from talkex.rules import (
            RuleDefinition as RD,
        )
        from talkex.rules import (
            RuleEvaluationInput as REI,
        )
        from talkex.rules import (
            RuleResult as RR,
        )

        assert RD is RuleDefinition
        assert REI is RuleEvaluationInput
        assert PR is PredicateResult
        assert RR is RuleResult
