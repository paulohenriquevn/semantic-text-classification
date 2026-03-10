"""Unit tests for rule engine benchmark runner.

Tests cover: single-configuration evaluation, multi-configuration comparison,
per-rule metrics collection, aggregated metrics, config params traceability,
protocol compliance, and report generation.
"""

from talkex.rules.benchmark import (
    RuleBenchmarkConfig,
    RuleBenchmarkRunner,
)
from talkex.rules.compiler import SimpleRuleCompiler
from talkex.rules.config import (
    EvidencePolicy,
    RuleEngineConfig,
    RuleEvaluationMode,
    ShortCircuitPolicy,
)
from talkex.rules.evaluator import SimpleRuleEvaluator
from talkex.rules.models import (
    RuleDefinition,
    RuleEvaluationInput,
)

_COMPILER = SimpleRuleCompiler()


def _make_rule(dsl: str, *, rule_id: str = "r1", rule_name: str = "test") -> RuleDefinition:
    return _COMPILER.compile(dsl, rule_id=rule_id, rule_name=rule_name)


def _make_input(
    *,
    text: str = "I need help with billing for my credit card",
    source_id: str = "win_001",
    features: dict[str, float] | None = None,
    speaker_role: str | None = None,
) -> RuleEvaluationInput:
    return RuleEvaluationInput(
        source_id=source_id,
        source_type="context_window",
        text=text,
        features=features or {},
        speaker_role=speaker_role,
    )


def _config(**overrides: object) -> RuleEngineConfig:
    return RuleEngineConfig(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Single configuration evaluation
# ---------------------------------------------------------------------------


class TestSingleEvaluation:
    def test_evaluate_returns_configuration_result(self) -> None:
        rules = [_make_rule('keyword("billing")')]
        inputs = [_make_input()]
        runner = RuleBenchmarkRunner(rules=rules, inputs=inputs)
        evaluator = SimpleRuleEvaluator()

        result = runner.evaluate(evaluator, _config(), "test_config")
        assert result.config_name == "test_config"
        assert result.total_rules == 1
        assert result.total_inputs == 1
        assert result.total_ms > 0

    def test_evaluate_collects_per_rule_metrics(self) -> None:
        rules = [
            _make_rule('keyword("billing")', rule_id="r1", rule_name="billing_rule"),
            _make_rule('keyword("shipping")', rule_id="r2", rule_name="shipping_rule"),
        ]
        inputs = [_make_input()]
        runner = RuleBenchmarkRunner(rules=rules, inputs=inputs)
        evaluator = SimpleRuleEvaluator()

        result = runner.evaluate(
            evaluator,
            _config(evidence_policy=EvidencePolicy.ALWAYS),
            "test",
        )
        assert len(result.rule_metrics) == 2
        # billing should match, shipping should not
        billing = next(rm for rm in result.rule_metrics if rm.rule_id == "r1")
        shipping = next(rm for rm in result.rule_metrics if rm.rule_id == "r2")
        assert billing.matched is True
        assert shipping.matched is False

    def test_evaluate_aggregated_metrics(self) -> None:
        rules = [_make_rule('keyword("billing")')]
        inputs = [_make_input(), _make_input(source_id="win_002")]
        runner = RuleBenchmarkRunner(rules=rules, inputs=inputs)
        evaluator = SimpleRuleEvaluator()

        result = runner.evaluate(evaluator, _config(), "test")
        agg = result.aggregated
        assert agg["rule_count"] == 2  # 1 rule x 2 inputs
        assert agg["match_rate"] == 1.0
        assert "avg_execution_time_ms" in agg

    def test_evaluate_config_params_traced(self) -> None:
        runner = RuleBenchmarkRunner(
            rules=[_make_rule('keyword("billing")')],
            inputs=[_make_input()],
        )
        evaluator = SimpleRuleEvaluator()
        cfg = _config(
            evaluation_mode=RuleEvaluationMode.SHORT_CIRCUIT,
            short_circuit_policy=ShortCircuitPolicy.DECLARATION,
            evidence_policy=EvidencePolicy.ALWAYS,
        )
        result = runner.evaluate(evaluator, cfg, "traced")
        assert result.config_params["evaluation_mode"] == "short_circuit"
        assert result.config_params["short_circuit_policy"] == "declaration"
        assert result.config_params["evidence_policy"] == "always"

    def test_evaluate_per_predicate_type(self) -> None:
        rules = [_make_rule('keyword("billing") AND speaker("customer")')]
        inputs = [_make_input(speaker_role="customer")]
        runner = RuleBenchmarkRunner(rules=rules, inputs=inputs)
        evaluator = SimpleRuleEvaluator()
        result = runner.evaluate(
            evaluator,
            _config(evidence_policy=EvidencePolicy.ALWAYS),
            "typed",
        )
        assert result.per_predicate_type.get("lexical", 0) >= 1
        assert result.per_predicate_type.get("structural", 0) >= 1


# ---------------------------------------------------------------------------
# Multi-configuration comparison
# ---------------------------------------------------------------------------


class TestComparison:
    def test_compare_produces_report(self) -> None:
        rules = [_make_rule('keyword("billing") AND keyword("credit")')]
        inputs = [_make_input()]
        runner = RuleBenchmarkRunner(
            rules=rules,
            inputs=inputs,
            config=RuleBenchmarkConfig(
                experiment_name="policy_compare",
                experiment_version="1.0",
            ),
        )
        evaluator = SimpleRuleEvaluator()

        configs = {
            "ALL_COST": _config(
                evaluation_mode=RuleEvaluationMode.ALL,
                short_circuit_policy=ShortCircuitPolicy.COST_ASCENDING,
                evidence_policy=EvidencePolicy.ALWAYS,
            ),
            "SC_DECL": _config(
                evaluation_mode=RuleEvaluationMode.ALL,
                short_circuit_policy=ShortCircuitPolicy.DECLARATION,
                evidence_policy=EvidencePolicy.ALWAYS,
            ),
        }
        report = runner.compare(evaluator, configs)
        assert report.experiment_name == "policy_compare"
        assert len(report.results) == 2
        names = {r.config_name for r in report.results}
        assert names == {"ALL_COST", "SC_DECL"}

    def test_compare_all_vs_short_circuit_mode(self) -> None:
        rules = [
            _make_rule('keyword("billing")', rule_id="r1", rule_name="r1"),
            _make_rule('keyword("shipping")', rule_id="r2", rule_name="r2"),
        ]
        inputs = [_make_input()]
        runner = RuleBenchmarkRunner(rules=rules, inputs=inputs)
        evaluator = SimpleRuleEvaluator()

        configs = {
            "ALL": _config(evaluation_mode=RuleEvaluationMode.ALL),
            "SHORT_CIRCUIT": _config(evaluation_mode=RuleEvaluationMode.SHORT_CIRCUIT),
        }
        report = runner.compare(evaluator, configs)

        all_result = next(r for r in report.results if r.config_name == "ALL")
        sc_result = next(r for r in report.results if r.config_name == "SHORT_CIRCUIT")

        # ALL evaluates both rules, SHORT_CIRCUIT stops after first match
        assert all_result.aggregated["rule_count"] == 2
        assert sc_result.aggregated["rule_count"] == 1

    def test_compare_evidence_policy(self) -> None:
        rules = [_make_rule('keyword("billing") OR keyword("shipping")')]
        inputs = [_make_input()]
        runner = RuleBenchmarkRunner(rules=rules, inputs=inputs)
        evaluator = SimpleRuleEvaluator()

        configs = {
            "ALWAYS": _config(
                evidence_policy=EvidencePolicy.ALWAYS,
                short_circuit_policy=ShortCircuitPolicy.DECLARATION,
            ),
            "MATCH_ONLY": _config(
                evidence_policy=EvidencePolicy.MATCH_ONLY,
                short_circuit_policy=ShortCircuitPolicy.DECLARATION,
            ),
        }
        report = runner.compare(evaluator, configs)

        always = next(r for r in report.results if r.config_name == "ALWAYS")
        match_only = next(r for r in report.results if r.config_name == "MATCH_ONLY")

        # ALWAYS keeps all evaluated predicates, MATCH_ONLY only matched
        assert always.aggregated["predicates_evaluated"] >= match_only.aggregated["predicates_evaluated"]

    def test_compare_serializes_to_json(self) -> None:
        rules = [_make_rule('keyword("billing")')]
        inputs = [_make_input()]
        runner = RuleBenchmarkRunner(rules=rules, inputs=inputs)
        evaluator = SimpleRuleEvaluator()

        configs = {"cfg1": _config(), "cfg2": _config()}
        report = runner.compare(evaluator, configs)

        import json

        data = json.loads(report.to_json())
        assert data["experiment_name"] == "rule_engine_benchmark"
        assert len(data["results"]) == 2

    def test_compare_serializes_to_csv(self) -> None:
        rules = [_make_rule('keyword("billing")')]
        inputs = [_make_input()]
        runner = RuleBenchmarkRunner(rules=rules, inputs=inputs)
        evaluator = SimpleRuleEvaluator()

        configs = {"cfg1": _config(), "cfg2": _config()}
        report = runner.compare(evaluator, configs)

        csv_text = report.to_csv()
        lines = csv_text.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows


# ---------------------------------------------------------------------------
# Multiple inputs
# ---------------------------------------------------------------------------


class TestMultipleInputs:
    def test_evaluate_across_multiple_inputs(self) -> None:
        rules = [_make_rule('keyword("billing")')]
        inputs = [
            _make_input(text="billing question", source_id="w1"),
            _make_input(text="shipping question", source_id="w2"),
            _make_input(text="billing issue again", source_id="w3"),
        ]
        runner = RuleBenchmarkRunner(rules=rules, inputs=inputs)
        evaluator = SimpleRuleEvaluator()

        result = runner.evaluate(evaluator, _config(), "multi")
        assert result.aggregated["rule_count"] == 3
        assert result.aggregated["rules_matched"] == 2  # w1 and w3 match


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestBenchmarkReexport:
    def test_importable_from_rules_package(self) -> None:
        from talkex.rules import (
            RuleBenchmarkConfig,
            RuleBenchmarkRunner,
        )

        assert RuleBenchmarkRunner is not None
        assert RuleBenchmarkConfig is not None
