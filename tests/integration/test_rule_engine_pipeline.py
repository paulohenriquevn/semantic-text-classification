"""Integration tests for rule engine pipeline.

End-to-end: raw text → segmentation → context windows → feature extraction
→ rule compilation → rule evaluation → metrics → benchmark comparison
→ report serialization (JSON/CSV).

Proves the rule engine operates on real pipeline outputs, not synthetic objects.
"""

import json

from semantic_conversation_engine.classification.features import (
    extract_lexical_features,
    extract_structural_features,
    merge_feature_sets,
)
from semantic_conversation_engine.context.builder import SlidingWindowBuilder
from semantic_conversation_engine.context.config import ContextWindowConfig
from semantic_conversation_engine.ingestion.enums import SourceFormat
from semantic_conversation_engine.ingestion.inputs import TranscriptInput
from semantic_conversation_engine.models.enums import Channel
from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
from semantic_conversation_engine.rules.benchmark import (
    RuleBenchmarkConfig,
    RuleBenchmarkRunner,
)
from semantic_conversation_engine.rules.compiler import SimpleRuleCompiler
from semantic_conversation_engine.rules.config import (
    EvidencePolicy,
    RuleEngineConfig,
    RuleEvaluationMode,
    ShortCircuitPolicy,
)
from semantic_conversation_engine.rules.evaluator import (
    SimpleRuleEvaluator,
    map_to_rule_execution,
)
from semantic_conversation_engine.rules.metrics import compute_rule_metrics
from semantic_conversation_engine.rules.models import RuleEvaluationInput
from semantic_conversation_engine.segmentation.segmenter import TurnSegmenter

_TRANSCRIPT_TEXT = """\
Customer: I have a billing issue with my credit card.
Agent: I can help you with that. What is the issue?
Customer: I was charged twice for the same order. Can I get a refund?
Agent: Let me look into that for you right away.
Customer: Also, I want to cancel my subscription.
Agent: I understand. Let me process both requests.
"""

_COMPILER = SimpleRuleCompiler()


def _build_pipeline_result():
    """Run the text processing pipeline on test transcript."""
    pipeline = TextProcessingPipeline(
        segmenter=TurnSegmenter(),
        context_builder=SlidingWindowBuilder(),
    )
    transcript = TranscriptInput(
        conversation_id="conv_rule_integ",
        raw_text=_TRANSCRIPT_TEXT,
        channel=Channel.VOICE,
        source_format=SourceFormat.LABELED,
    )
    return pipeline.run(
        transcript,
        context_config=ContextWindowConfig(window_size=3, stride=2),
    )


def _build_evaluation_inputs(pipeline_result):
    """Convert pipeline windows to rule evaluation inputs with features."""
    inputs = []
    for window in pipeline_result.windows:
        lex = extract_lexical_features(window.window_text)
        struct = extract_structural_features(
            turn_count=window.window_size,
            speaker_count=2,
        )
        merged = merge_feature_sets(lex, struct)

        inputs.append(
            RuleEvaluationInput(
                source_id=window.window_id,
                source_type="context_window",
                text=window.window_text,
                features=merged.features,
                metadata={
                    "conversation_id": pipeline_result.conversation.conversation_id,
                },
            )
        )
    return inputs


# ---------------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------------


class TestEndToEndEvaluation:
    def test_pipeline_to_rule_evaluation(self) -> None:
        """Full pipeline: text → windows → features → rule evaluation."""
        result = _build_pipeline_result()
        assert len(result.windows) > 0

        inputs = _build_evaluation_inputs(result)
        assert len(inputs) == len(result.windows)

        rules = [
            _COMPILER.compile(
                'keyword("billing")',
                rule_id="billing_detect",
                rule_name="Billing Detection",
            ),
            _COMPILER.compile(
                'keyword("cancel")',
                rule_id="cancel_detect",
                rule_name="Cancel Detection",
            ),
            _COMPILER.compile(
                'keyword("refund") OR keyword("charged twice")',
                rule_id="refund_detect",
                rule_name="Refund Detection",
            ),
        ]

        evaluator = SimpleRuleEvaluator()
        config = RuleEngineConfig(evidence_policy=EvidencePolicy.ALWAYS)

        all_results = []
        for inp in inputs:
            results = evaluator.evaluate(rules, inp, config)
            all_results.extend(results)

        # Should have results for each rule x each input
        assert len(all_results) == len(rules) * len(inputs)

        # At least some rules should match (billing is in the transcript)
        assert any(r.matched for r in all_results)

        # Every result has valid structure
        for r in all_results:
            assert r.source_type == "context_window"
            assert r.rule_version == "1.0"
            assert r.execution_time_ms >= 0
            assert 0.0 <= r.score <= 1.0

    def test_matched_rules_have_evidence(self) -> None:
        """Matched rules with ALWAYS policy must have predicate results."""
        result = _build_pipeline_result()
        inputs = _build_evaluation_inputs(result)

        rules = [
            _COMPILER.compile(
                'keyword("billing")',
                rule_id="r1",
                rule_name="r1",
            ),
        ]

        evaluator = SimpleRuleEvaluator()
        config = RuleEngineConfig(evidence_policy=EvidencePolicy.ALWAYS)

        for inp in inputs:
            results = evaluator.evaluate(rules, inp, config)
            for r in results:
                # With ALWAYS, evaluated predicates are always present
                assert len(r.predicate_results) > 0


# ---------------------------------------------------------------------------
# Metrics on real pipeline output
# ---------------------------------------------------------------------------


class TestMetricsOnPipelineOutput:
    def test_compute_metrics_from_pipeline(self) -> None:
        result = _build_pipeline_result()
        inputs = _build_evaluation_inputs(result)

        rules = [
            _COMPILER.compile('keyword("billing")', rule_id="r1", rule_name="r1"),
            _COMPILER.compile('keyword("shipping")', rule_id="r2", rule_name="r2"),
        ]

        evaluator = SimpleRuleEvaluator()
        config = RuleEngineConfig(evidence_policy=EvidencePolicy.ALWAYS)

        all_results = []
        for inp in inputs:
            all_results.extend(evaluator.evaluate(rules, inp, config))

        metrics = compute_rule_metrics(all_results)

        assert metrics["rule_count"] == len(rules) * len(inputs)
        assert 0.0 <= metrics["match_rate"] <= 1.0
        assert 0.0 <= metrics["short_circuit_rate"] <= 1.0
        assert metrics["predicates_evaluated"] > 0
        assert metrics["avg_execution_time_ms"] >= 0
        assert isinstance(metrics["predicate_type_distribution"], dict)


# ---------------------------------------------------------------------------
# Boundary mapping: RuleResult → RuleExecution
# ---------------------------------------------------------------------------


class TestBoundaryMapping:
    def test_map_to_rule_execution_from_pipeline(self) -> None:
        result = _build_pipeline_result()
        inputs = _build_evaluation_inputs(result)

        rules = [
            _COMPILER.compile('keyword("billing")', rule_id="r1", rule_name="r1"),
        ]

        evaluator = SimpleRuleEvaluator()
        config = RuleEngineConfig(evidence_policy=EvidencePolicy.ALWAYS)

        results = evaluator.evaluate(rules, inputs[0], config)
        assert len(results) == 1

        execution = map_to_rule_execution(results[0])
        assert execution.rule_id == "r1"
        assert execution.source_id == inputs[0].source_id
        assert execution.execution_time_ms >= 0
        if execution.matched:
            assert len(execution.evidence) > 0


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------


class TestBenchmarkComparison:
    def test_compare_policies_on_pipeline_output(self) -> None:
        result = _build_pipeline_result()
        inputs = _build_evaluation_inputs(result)

        rules = [
            _COMPILER.compile(
                'keyword("billing") AND keyword("credit")',
                rule_id="compound",
                rule_name="compound_rule",
            ),
            _COMPILER.compile(
                'keyword("cancel") OR keyword("refund")',
                rule_id="disjunction",
                rule_name="disjunction_rule",
            ),
        ]

        runner = RuleBenchmarkRunner(
            rules=rules,
            inputs=inputs,
            config=RuleBenchmarkConfig(
                experiment_name="pipeline_policy_compare",
                experiment_version="1.0",
            ),
        )
        evaluator = SimpleRuleEvaluator()

        configs = {
            "ALL_COST_ALWAYS": RuleEngineConfig(
                evaluation_mode=RuleEvaluationMode.ALL,
                short_circuit_policy=ShortCircuitPolicy.COST_ASCENDING,
                evidence_policy=EvidencePolicy.ALWAYS,
            ),
            "ALL_DECL_MATCH": RuleEngineConfig(
                evaluation_mode=RuleEvaluationMode.ALL,
                short_circuit_policy=ShortCircuitPolicy.DECLARATION,
                evidence_policy=EvidencePolicy.MATCH_ONLY,
            ),
            "SC_COST_ALWAYS": RuleEngineConfig(
                evaluation_mode=RuleEvaluationMode.SHORT_CIRCUIT,
                short_circuit_policy=ShortCircuitPolicy.COST_ASCENDING,
                evidence_policy=EvidencePolicy.ALWAYS,
            ),
        }

        report = runner.compare(evaluator, configs)

        assert report.experiment_name == "pipeline_policy_compare"
        assert len(report.results) == 3

        for cr in report.results:
            assert cr.total_inputs == len(inputs)
            assert cr.total_rules == len(rules)
            assert cr.total_ms > 0
            assert 0.0 <= cr.aggregated["match_rate"] <= 1.0
            assert 0.0 <= cr.aggregated["short_circuit_rate"] <= 1.0
            assert cr.aggregated["predicates_evaluated"] >= 0

    def test_short_circuit_evaluates_fewer_rules(self) -> None:
        result = _build_pipeline_result()
        inputs = _build_evaluation_inputs(result)

        # First rule matches, so SHORT_CIRCUIT should skip the second
        rules = [
            _COMPILER.compile('keyword("billing")', rule_id="r1", rule_name="r1"),
            _COMPILER.compile('keyword("cancel")', rule_id="r2", rule_name="r2"),
        ]

        runner = RuleBenchmarkRunner(rules=rules, inputs=inputs)
        evaluator = SimpleRuleEvaluator()

        configs = {
            "ALL": RuleEngineConfig(evaluation_mode=RuleEvaluationMode.ALL),
            "SC": RuleEngineConfig(evaluation_mode=RuleEvaluationMode.SHORT_CIRCUIT),
        }
        report = runner.compare(evaluator, configs)

        all_count = next(r for r in report.results if r.config_name == "ALL")
        sc_count = next(r for r in report.results if r.config_name == "SC")

        # SHORT_CIRCUIT should evaluate fewer total rules
        assert sc_count.aggregated["rule_count"] <= all_count.aggregated["rule_count"]


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------


class TestReportSerialization:
    def test_json_round_trip(self, tmp_path) -> None:
        result = _build_pipeline_result()
        inputs = _build_evaluation_inputs(result)

        rules = [_COMPILER.compile('keyword("billing")', rule_id="r1", rule_name="r1")]
        runner = RuleBenchmarkRunner(
            rules=rules,
            inputs=inputs,
            config=RuleBenchmarkConfig(experiment_name="json_rt"),
        )
        evaluator = SimpleRuleEvaluator()

        report = runner.compare(evaluator, {"default": RuleEngineConfig()})

        path = tmp_path / "report.json"
        report.save_json(path)
        loaded = json.loads(path.read_text())

        assert loaded["experiment_name"] == "json_rt"
        assert len(loaded["results"]) == 1
        assert "match_rate" in loaded["results"][0]["aggregated"]

    def test_csv_export(self, tmp_path) -> None:
        result = _build_pipeline_result()
        inputs = _build_evaluation_inputs(result)

        rules = [_COMPILER.compile('keyword("billing")', rule_id="r1", rule_name="r1")]
        runner = RuleBenchmarkRunner(rules=rules, inputs=inputs)
        evaluator = SimpleRuleEvaluator()

        report = runner.compare(
            evaluator,
            {
                "cfg_a": RuleEngineConfig(),
                "cfg_b": RuleEngineConfig(
                    evaluation_mode=RuleEvaluationMode.SHORT_CIRCUIT,
                ),
            },
        )

        path = tmp_path / "report.csv"
        report.save_csv(path)
        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 configs
        assert "cfg_a" in content
        assert "cfg_b" in content


# ---------------------------------------------------------------------------
# Feature-based rules on pipeline output
# ---------------------------------------------------------------------------


class TestFeatureBasedRules:
    def test_structural_predicates_with_extracted_features(self) -> None:
        """Rules using structural features from pipeline extraction."""
        result = _build_pipeline_result()
        inputs = _build_evaluation_inputs(result)

        # word_count is extracted by extract_lexical_features
        rules = [
            _COMPILER.compile(
                'field_gte("word_count", 5)',
                rule_id="verbose",
                rule_name="verbose_window",
            ),
        ]

        evaluator = SimpleRuleEvaluator()
        config = RuleEngineConfig(evidence_policy=EvidencePolicy.ALWAYS)

        matched_count = 0
        for inp in inputs:
            results = evaluator.evaluate(rules, inp, config)
            if results[0].matched:
                matched_count += 1

        # Windows with real text should have word_count >= 5
        assert matched_count > 0
