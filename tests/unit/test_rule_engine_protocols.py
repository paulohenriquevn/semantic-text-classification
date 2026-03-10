"""Unit tests for rule engine protocol compliance.

Tests verify that stub implementations satisfying the RuleCompiler and
RuleEvaluator protocol signatures work correctly with the type system.
This validates DIP — concrete implementations need no inheritance from
Protocol classes.

Note: Protocol types are imported from ``rules`` (not ``pipeline.protocols``)
to avoid the circular import triggered by ``pipeline.__init__`` →
``pipeline.pipeline`` → ``pipeline.protocols`` → ``classification`` →
``classification.orchestrator`` → ``pipeline.protocols``.
"""

from semantic_conversation_engine.rules.ast import PredicateNode
from semantic_conversation_engine.rules.config import (
    PredicateType,
    RuleEngineConfig,
    RuleEvaluationMode,
)
from semantic_conversation_engine.rules.models import (
    PredicateResult,
    RuleDefinition,
    RuleEvaluationInput,
    RuleResult,
)

# ---------------------------------------------------------------------------
# Stubs (pure duck typing — no Protocol inheritance)
# ---------------------------------------------------------------------------


class StubRuleCompiler:
    """Stub compiler that produces a single lexical predicate AST."""

    def compile(self, dsl_text: str, rule_id: str, rule_name: str) -> RuleDefinition:
        return RuleDefinition(
            rule_id=rule_id,
            rule_name=rule_name,
            rule_version="1.0",
            description=f"Compiled from: {dsl_text[:40]}",
            ast=PredicateNode(
                predicate_type=PredicateType.LEXICAL,
                field_name="text",
                operator="contains",
                value="billing",
            ),
        )


class StubRuleEvaluator:
    """Stub evaluator that matches when text contains 'billing'."""

    def evaluate(
        self,
        rules: list[RuleDefinition],
        evaluation_input: RuleEvaluationInput,
        config: RuleEngineConfig,
    ) -> list[RuleResult]:
        results: list[RuleResult] = []
        text_lower = evaluation_input.text.lower()

        for rule in rules:
            matched = "billing" in text_lower
            pr = PredicateResult(
                predicate_type=PredicateType.LEXICAL,
                field_name="text",
                operator="contains",
                matched=matched,
                score=1.0 if matched else 0.0,
                threshold=0.5,
                matched_text="billing" if matched else None,
            )
            results.append(
                RuleResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    rule_version=rule.rule_version,
                    source_id=evaluation_input.source_id,
                    source_type=evaluation_input.source_type,
                    matched=matched,
                    score=pr.score,
                    predicate_results=[pr],
                )
            )
        return results


# ---------------------------------------------------------------------------
# RuleCompiler protocol compliance
# ---------------------------------------------------------------------------


class TestRuleCompilerProtocol:
    def test_stub_compiles_rule(self) -> None:
        compiler = StubRuleCompiler()
        rule = compiler.compile(
            dsl_text='WHEN text CONTAINS "billing" THEN tag("billing")',
            rule_id="rule_001",
            rule_name="billing_detection",
        )
        assert rule.rule_id == "rule_001"
        assert rule.rule_name == "billing_detection"
        assert isinstance(rule.ast, PredicateNode)

    def test_stub_preserves_rule_identity(self) -> None:
        compiler = StubRuleCompiler()
        r1 = compiler.compile("dsl_a", "id_a", "name_a")
        r2 = compiler.compile("dsl_b", "id_b", "name_b")
        assert r1.rule_id != r2.rule_id
        assert r1.rule_name != r2.rule_name

    def test_stub_includes_description(self) -> None:
        compiler = StubRuleCompiler()
        rule = compiler.compile("billing rule", "r1", "billing")
        assert "billing rule" in rule.description


# ---------------------------------------------------------------------------
# RuleEvaluator protocol compliance
# ---------------------------------------------------------------------------


class TestRuleEvaluatorProtocol:
    def test_stub_matches_billing_text(self) -> None:
        evaluator = StubRuleEvaluator()
        rule = RuleDefinition(
            rule_id="rule_001",
            rule_name="billing_detection",
            rule_version="1.0",
            description="Detect billing",
            ast=PredicateNode(
                predicate_type=PredicateType.LEXICAL,
                field_name="text",
                operator="contains",
                value="billing",
            ),
        )
        evaluation_input = RuleEvaluationInput(
            source_id="win_001",
            source_type="context_window",
            text="I have a billing issue",
        )
        results = evaluator.evaluate([rule], evaluation_input, RuleEngineConfig())
        assert len(results) == 1
        assert results[0].matched is True
        assert results[0].score == 1.0

    def test_stub_does_not_match_unrelated_text(self) -> None:
        evaluator = StubRuleEvaluator()
        rule = RuleDefinition(
            rule_id="rule_001",
            rule_name="billing_detection",
            rule_version="1.0",
            description="Detect billing",
            ast=PredicateNode(
                predicate_type=PredicateType.LEXICAL,
                field_name="text",
                operator="contains",
                value="billing",
            ),
        )
        evaluation_input = RuleEvaluationInput(
            source_id="win_002",
            source_type="context_window",
            text="I need help with shipping",
        )
        results = evaluator.evaluate([rule], evaluation_input, RuleEngineConfig())
        assert len(results) == 1
        assert results[0].matched is False
        assert results[0].score == 0.0

    def test_stub_returns_one_result_per_rule(self) -> None:
        evaluator = StubRuleEvaluator()
        rules = [
            RuleDefinition(
                rule_id=f"rule_{i}",
                rule_name=f"rule_{i}",
                rule_version="1.0",
                description="test",
                ast=PredicateNode(
                    predicate_type=PredicateType.LEXICAL,
                    field_name="text",
                    operator="contains",
                    value="billing",
                ),
            )
            for i in range(3)
        ]
        evaluation_input = RuleEvaluationInput(
            source_id="win_001",
            source_type="context_window",
            text="billing problem",
        )
        results = evaluator.evaluate(rules, evaluation_input, RuleEngineConfig())
        assert len(results) == 3

    def test_stub_propagates_source_identity(self) -> None:
        evaluator = StubRuleEvaluator()
        rule = RuleDefinition(
            rule_id="rule_001",
            rule_name="test",
            rule_version="1.0",
            description="test",
            ast=PredicateNode(
                predicate_type=PredicateType.LEXICAL,
                field_name="text",
                operator="contains",
                value="billing",
            ),
        )
        evaluation_input = RuleEvaluationInput(
            source_id="win_042",
            source_type="conversation",
            text="billing inquiry",
        )
        results = evaluator.evaluate([rule], evaluation_input, RuleEngineConfig())
        assert results[0].source_id == "win_042"
        assert results[0].source_type == "conversation"

    def test_stub_result_has_predicate_results(self) -> None:
        evaluator = StubRuleEvaluator()
        rule = RuleDefinition(
            rule_id="rule_001",
            rule_name="test",
            rule_version="1.0",
            description="test",
            ast=PredicateNode(
                predicate_type=PredicateType.LEXICAL,
                field_name="text",
                operator="contains",
                value="billing",
            ),
        )
        evaluation_input = RuleEvaluationInput(
            source_id="win_001",
            source_type="context_window",
            text="billing issue",
        )
        results = evaluator.evaluate([rule], evaluation_input, RuleEngineConfig())
        pr = results[0].predicate_results[0]
        assert pr.predicate_type == PredicateType.LEXICAL
        assert pr.matched is True
        assert pr.matched_text == "billing"

    def test_stub_accepts_short_circuit_config(self) -> None:
        evaluator = StubRuleEvaluator()
        rule = RuleDefinition(
            rule_id="rule_001",
            rule_name="test",
            rule_version="1.0",
            description="test",
            ast=PredicateNode(
                predicate_type=PredicateType.LEXICAL,
                field_name="text",
                operator="contains",
                value="billing",
            ),
        )
        config = RuleEngineConfig(evaluation_mode=RuleEvaluationMode.SHORT_CIRCUIT)
        evaluation_input = RuleEvaluationInput(
            source_id="win_001",
            source_type="context_window",
            text="billing",
        )
        results = evaluator.evaluate([rule], evaluation_input, config)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Reexport from pipeline package
# ---------------------------------------------------------------------------


class TestRuleProtocolReexport:
    def test_rule_compiler_from_pipeline(self) -> None:
        from semantic_conversation_engine.pipeline import RuleCompiler
        from semantic_conversation_engine.pipeline.protocols import (
            RuleCompiler as DirectImport,
        )

        assert RuleCompiler is DirectImport

    def test_rule_evaluator_from_pipeline(self) -> None:
        from semantic_conversation_engine.pipeline import RuleEvaluator
        from semantic_conversation_engine.pipeline.protocols import (
            RuleEvaluator as DirectImport,
        )

        assert RuleEvaluator is DirectImport
