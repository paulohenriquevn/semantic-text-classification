"""Unit tests for rule compiler and semantic validation.

Tests cover: SimpleRuleCompiler construction, DSL → RuleDefinition compilation,
semantic validation (threshold, cost_hint, depth, empty nodes), identity
preservation, and error handling.
"""

import pytest

from semantic_conversation_engine.exceptions import RuleError
from semantic_conversation_engine.rules.ast import (
    AndNode,
    NotNode,
    OrNode,
    PredicateNode,
)
from semantic_conversation_engine.rules.compiler import SimpleRuleCompiler, _validate_ast
from semantic_conversation_engine.rules.config import PredicateType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_predicate(**overrides: object) -> PredicateNode:
    defaults: dict[str, object] = {
        "predicate_type": PredicateType.LEXICAL,
        "field_name": "text",
        "operator": "contains",
        "value": "billing",
    }
    defaults.update(overrides)
    return PredicateNode(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Semantic validation
# ---------------------------------------------------------------------------


class TestSemanticValidation:
    def test_valid_predicate_passes(self) -> None:
        node = _make_predicate()
        _validate_ast(node)  # should not raise

    def test_valid_and_node_passes(self) -> None:
        node = AndNode(children=[_make_predicate(), _make_predicate()])
        _validate_ast(node)  # should not raise

    def test_empty_and_node_rejected(self) -> None:
        node = AndNode(children=[])
        with pytest.raises(RuleError, match="Empty AND node"):
            _validate_ast(node)

    def test_empty_or_node_rejected(self) -> None:
        node = OrNode(children=[])
        with pytest.raises(RuleError, match="Empty OR node"):
            _validate_ast(node)

    def test_threshold_above_one_rejected(self) -> None:
        node = _make_predicate(threshold=1.5)
        with pytest.raises(RuleError, match="Threshold must be in"):
            _validate_ast(node)

    def test_threshold_below_zero_rejected(self) -> None:
        node = _make_predicate(threshold=-0.1)
        with pytest.raises(RuleError, match="Threshold must be in"):
            _validate_ast(node)

    def test_threshold_boundary_values_pass(self) -> None:
        _validate_ast(_make_predicate(threshold=0.0))
        _validate_ast(_make_predicate(threshold=1.0))

    def test_threshold_none_passes(self) -> None:
        _validate_ast(_make_predicate(threshold=None))

    def test_cost_hint_zero_rejected(self) -> None:
        node = _make_predicate(cost_hint=0)
        with pytest.raises(RuleError, match="cost_hint must be >= 1"):
            _validate_ast(node)

    def test_cost_hint_negative_rejected(self) -> None:
        node = _make_predicate(cost_hint=-1)
        with pytest.raises(RuleError, match="cost_hint must be >= 1"):
            _validate_ast(node)

    def test_excessive_depth_rejected(self) -> None:
        # Build a chain of 25 NOT nodes (exceeds max depth of 20)
        node: PredicateNode | AndNode | OrNode | NotNode = _make_predicate()
        for _ in range(25):
            node = NotNode(child=node)
        with pytest.raises(RuleError, match="exceeds maximum depth"):
            _validate_ast(node)

    def test_valid_deep_nesting_passes(self) -> None:
        # 10 levels deep — should be fine
        node: PredicateNode | AndNode | OrNode | NotNode = _make_predicate()
        for _ in range(10):
            node = NotNode(child=node)
        _validate_ast(node)  # should not raise

    def test_nested_or_validated(self) -> None:
        node = AndNode(
            children=[
                OrNode(children=[_make_predicate(), _make_predicate(threshold=2.0)]),
                _make_predicate(),
            ]
        )
        with pytest.raises(RuleError, match="Threshold must be in"):
            _validate_ast(node)


# ---------------------------------------------------------------------------
# SimpleRuleCompiler — basic compilation
# ---------------------------------------------------------------------------


class TestSimpleRuleCompiler:
    def test_compile_simple_keyword(self) -> None:
        compiler = SimpleRuleCompiler()
        rule = compiler.compile(
            dsl_text='keyword("billing")',
            rule_id="rule_001",
            rule_name="billing_detection",
        )
        assert rule.rule_id == "rule_001"
        assert rule.rule_name == "billing_detection"
        assert rule.rule_version == "1.0"
        assert isinstance(rule.ast, PredicateNode)
        assert rule.ast.value == "billing"

    def test_compile_and_expression(self) -> None:
        compiler = SimpleRuleCompiler()
        rule = compiler.compile(
            dsl_text='keyword("billing") AND speaker("customer")',
            rule_id="rule_002",
            rule_name="billing_customer",
        )
        assert isinstance(rule.ast, AndNode)
        assert len(rule.ast.children) == 2

    def test_compile_complex_expression(self) -> None:
        compiler = SimpleRuleCompiler()
        rule = compiler.compile(
            dsl_text='(keyword("billing") OR keyword("invoice")) AND NOT keyword("upgrade")',
            rule_id="rule_003",
            rule_name="billing_no_upgrade",
        )
        assert isinstance(rule.ast, AndNode)
        assert isinstance(rule.ast.children[0], OrNode)
        assert isinstance(rule.ast.children[1], NotNode)

    def test_custom_version(self) -> None:
        compiler = SimpleRuleCompiler()
        rule = compiler.compile(
            dsl_text='keyword("billing")',
            rule_id="r1",
            rule_name="billing",
            rule_version="2.3",
        )
        assert rule.rule_version == "2.3"

    def test_custom_description(self) -> None:
        compiler = SimpleRuleCompiler()
        rule = compiler.compile(
            dsl_text='keyword("billing")',
            rule_id="r1",
            rule_name="billing",
            description="Detects billing-related conversations",
        )
        assert rule.description == "Detects billing-related conversations"

    def test_auto_description(self) -> None:
        compiler = SimpleRuleCompiler()
        rule = compiler.compile(
            dsl_text='keyword("billing")',
            rule_id="r1",
            rule_name="billing",
        )
        assert 'keyword("billing")' in rule.description

    def test_custom_priority(self) -> None:
        compiler = SimpleRuleCompiler()
        rule = compiler.compile(
            dsl_text='keyword("billing")',
            rule_id="r1",
            rule_name="billing",
            priority=10,
        )
        assert rule.priority == 10

    def test_custom_tags(self) -> None:
        compiler = SimpleRuleCompiler()
        rule = compiler.compile(
            dsl_text='keyword("billing")',
            rule_id="r1",
            rule_name="billing",
            tags=["compliance", "fraud"],
        )
        assert "compliance" in rule.tags
        assert "fraud" in rule.tags

    def test_default_tags_empty(self) -> None:
        compiler = SimpleRuleCompiler()
        rule = compiler.compile(
            dsl_text='keyword("billing")',
            rule_id="r1",
            rule_name="billing",
        )
        assert rule.tags == []


# ---------------------------------------------------------------------------
# SimpleRuleCompiler — error propagation
# ---------------------------------------------------------------------------


class TestSimpleRuleCompilerErrors:
    def test_compile_empty_dsl_raises(self) -> None:
        compiler = SimpleRuleCompiler()
        with pytest.raises(RuleError, match="Empty DSL expression"):
            compiler.compile("", "r1", "test")

    def test_compile_invalid_syntax_raises(self) -> None:
        compiler = SimpleRuleCompiler()
        with pytest.raises(RuleError):
            compiler.compile("keyword AND", "r1", "test")

    def test_compile_unknown_predicate_raises(self) -> None:
        compiler = SimpleRuleCompiler()
        with pytest.raises(RuleError, match="Unknown predicate"):
            compiler.compile('unknown_func("x")', "r1", "test")


# ---------------------------------------------------------------------------
# SimpleRuleCompiler — protocol compliance
# ---------------------------------------------------------------------------


class TestCompilerProtocolCompliance:
    def test_satisfies_rule_compiler_protocol(self) -> None:
        """SimpleRuleCompiler structurally satisfies the RuleCompiler protocol."""
        compiler = SimpleRuleCompiler()
        # Protocol requires: compile(dsl_text, rule_id, rule_name) -> RuleDefinition
        rule = compiler.compile('keyword("billing")', "r1", "billing")
        assert rule.rule_id == "r1"

    def test_multiple_compilations_independent(self) -> None:
        compiler = SimpleRuleCompiler()
        r1 = compiler.compile('keyword("a")', "r1", "rule_a")
        r2 = compiler.compile('keyword("b")', "r2", "rule_b")
        assert r1.rule_id != r2.rule_id
        assert r1.ast != r2.ast  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestCompilerReexport:
    def test_simple_rule_compiler_from_rules_package(self) -> None:
        from semantic_conversation_engine.rules import SimpleRuleCompiler as Imported

        assert Imported is SimpleRuleCompiler
