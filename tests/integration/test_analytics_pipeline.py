"""Integration tests for analytics pipeline.

End-to-end: raw text → segmentation → context windows → classification
→ rule evaluation → analytics events → engine → query runner → benchmark
→ report serialization (JSON/CSV).

Proves the analytics subsystem operates on real pipeline outputs, not synthetic objects.
"""

import json
from datetime import UTC, datetime

from semantic_conversation_engine.analytics.aggregators import SimpleAnalyticsEngine
from semantic_conversation_engine.analytics.benchmark import (
    AnalyticsBenchmarkRunner,
)
from semantic_conversation_engine.analytics.builders import (
    rule_execution_to_event,
)
from semantic_conversation_engine.analytics.config import (
    AggregationWindow,
    AnalyticsLevel,
    MetricType,
)
from semantic_conversation_engine.analytics.metrics import compute_analytics_metrics
from semantic_conversation_engine.analytics.models import AnalyticsQuery
from semantic_conversation_engine.analytics.query_runner import AnalyticsQueryRunner
from semantic_conversation_engine.classification.features import (
    extract_lexical_features,
)
from semantic_conversation_engine.context.builder import SlidingWindowBuilder
from semantic_conversation_engine.context.config import ContextWindowConfig
from semantic_conversation_engine.ingestion.enums import SourceFormat
from semantic_conversation_engine.ingestion.inputs import TranscriptInput
from semantic_conversation_engine.models.enums import Channel
from semantic_conversation_engine.pipeline.pipeline import TextProcessingPipeline
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
        conversation_id="conv_analytics_test",
        raw_text=_TRANSCRIPT_TEXT,
        source_format=SourceFormat.LABELED,
        channel=Channel.VOICE,
    )
    config = ContextWindowConfig(window_size=3, stride=2)
    return pipeline.run(transcript, context_config=config)


def _build_analytics_events():
    """Build analytics events from real pipeline outputs via rules."""
    result = _build_pipeline_result()
    windows = result.windows

    # Compile rules
    rules = [
        _COMPILER.compile('keyword("billing")', "rule_billing", "billing_issue"),
        _COMPILER.compile('keyword("cancel")', "rule_cancel", "cancel_intent"),
        _COMPILER.compile('keyword("refund")', "rule_refund", "refund_request"),
    ]

    evaluator = SimpleRuleEvaluator()
    config = RuleEngineConfig(
        evaluation_mode=RuleEvaluationMode.ALL,
        evidence_policy=EvidencePolicy.ALWAYS,
        short_circuit_policy=ShortCircuitPolicy.DECLARATION,
    )

    events = []
    event_counter = 0

    for window in windows:
        # Extract features for rule evaluation
        lexical = extract_lexical_features(window.window_text)

        rule_input = RuleEvaluationInput(
            source_id=window.window_id,
            source_type="context_window",
            text=window.window_text,
            features=lexical.features,
            metadata=window.metadata,
        )

        rule_results = evaluator.evaluate(rules, rule_input, config)

        for rule_result in rule_results:
            execution = map_to_rule_execution(rule_result)
            event_counter += 1
            event = rule_execution_to_event(
                execution,
                event_id=f"evt_{event_counter:03d}",
                timestamp=datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC),
            )
            events.append(event)

    return events


class TestAnalyticsPipelineIntegration:
    """End-to-end analytics pipeline integration tests."""

    def test_events_from_real_pipeline(self) -> None:
        """Verify events are produced from real pipeline outputs."""
        events = _build_analytics_events()
        assert len(events) > 0
        assert all(e.metric_type == MetricType.RULE for e in events)
        assert all(e.event_type == "rule_execution" for e in events)

    def test_engine_aggregation_on_real_events(self) -> None:
        """Verify engine can aggregate real pipeline events."""
        events = _build_analytics_events()
        engine = SimpleAnalyticsEngine(events=events)

        stats = engine.compute_stats()
        assert stats["total_events"] == len(events)
        assert "match_rate" in stats
        assert "metric_type_distribution" in stats

    def test_query_grouped_on_real_events(self) -> None:
        """Verify grouped query on real pipeline events."""
        events = _build_analytics_events()
        engine = SimpleAnalyticsEngine(events=events)
        runner = AnalyticsQueryRunner(engine)

        query = AnalyticsQuery(
            query_id="int_q1",
            level=AnalyticsLevel.SYSTEM,
            group_by="label",
        )
        report = runner.run_grouped(query, section_name="Rules by Name")

        assert len(report.sections) == 1
        # Should have groups for billing_issue, cancel_intent, refund_request
        group_values = {gm.group_value for gm in report.sections[0].grouped_metrics}
        assert "billing_issue" in group_values
        assert "cancel_intent" in group_values

    def test_query_trend_on_real_events(self) -> None:
        """Verify trend query on real pipeline events."""
        events = _build_analytics_events()
        engine = SimpleAnalyticsEngine(events=events)
        runner = AnalyticsQueryRunner(engine)

        query = AnalyticsQuery(
            query_id="int_q2",
            level=AnalyticsLevel.SYSTEM,
            window=AggregationWindow.DAILY,
        )
        report = runner.run_trend(query, "match_rate", section_name="Match Rate Trend")

        assert len(report.sections) == 1
        assert len(report.sections[0].trend_series) == 1
        # All events on same day → single trend point
        assert len(report.sections[0].trend_series[0].points) == 1

    def test_composite_report_on_real_events(self) -> None:
        """Verify composite report with grouped + trend on real events."""
        events = _build_analytics_events()
        engine = SimpleAnalyticsEngine(events=events)
        runner = AnalyticsQueryRunner(engine)

        report = runner.run_composite(
            [
                (
                    AnalyticsQuery(query_id="int_q3", level=AnalyticsLevel.SYSTEM, group_by="label"),
                    "By Rule Name",
                    "Rules grouped by name",
                ),
            ],
            trend_queries=[
                (
                    AnalyticsQuery(query_id="int_q4", level=AnalyticsLevel.SYSTEM, window=AggregationWindow.DAILY),
                    "match_rate",
                    "Match Rate Over Time",
                    "Daily match rate trend",
                ),
            ],
            report_name="Full Analytics Report",
        )

        assert report.report_name == "Full Analytics Report"
        assert len(report.sections) == 2
        assert report.metadata["grouped_sections"] == 1
        assert report.metadata["trend_sections"] == 1

    def test_benchmark_on_real_events(self) -> None:
        """Verify benchmark runner compares scenarios on real events."""
        events = _build_analytics_events()
        engine = SimpleAnalyticsEngine(events=events)
        runner = AnalyticsBenchmarkRunner(engine=engine)

        report = runner.compare(
            {
                "by_label": AnalyticsQuery(query_id="bq1", level=AnalyticsLevel.SYSTEM, group_by="label"),
                "by_source": AnalyticsQuery(query_id="bq2", level=AnalyticsLevel.SYSTEM, group_by="source_id"),
            },
            trend_scenarios={
                "daily_match": (
                    AnalyticsQuery(query_id="bq3", level=AnalyticsLevel.SYSTEM, window=AggregationWindow.DAILY),
                    "match_rate",
                ),
            },
        )

        assert len(report.results) == 3
        assert report.total_events == len(events)
        assert report.aggregated["report_count"] == 3

    def test_benchmark_json_serialization(self) -> None:
        """Verify benchmark report serializes to valid JSON."""
        events = _build_analytics_events()
        engine = SimpleAnalyticsEngine(events=events)
        runner = AnalyticsBenchmarkRunner(engine=engine)

        report = runner.compare(
            {
                "by_label": AnalyticsQuery(query_id="bq4", level=AnalyticsLevel.SYSTEM, group_by="label"),
            }
        )

        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["experiment_name"] == "analytics_benchmark"
        assert len(data["results"]) == 1

    def test_benchmark_csv_serialization(self) -> None:
        """Verify benchmark report serializes to valid CSV."""
        events = _build_analytics_events()
        engine = SimpleAnalyticsEngine(events=events)
        runner = AnalyticsBenchmarkRunner(engine=engine)

        report = runner.compare(
            {
                "by_label": AnalyticsQuery(query_id="bq5", level=AnalyticsLevel.SYSTEM, group_by="label"),
                "by_source": AnalyticsQuery(query_id="bq6", level=AnalyticsLevel.SYSTEM, group_by="source_id"),
            }
        )

        csv_str = report.to_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 3  # header + 2 scenarios
        assert "by_label" in lines[1] or "by_label" in lines[2]

    def test_operational_metrics_on_real_reports(self) -> None:
        """Verify compute_analytics_metrics on real reports."""
        events = _build_analytics_events()
        engine = SimpleAnalyticsEngine(events=events)
        runner = AnalyticsQueryRunner(engine)

        reports = [
            runner.run_grouped(
                AnalyticsQuery(query_id="mq1", level=AnalyticsLevel.SYSTEM, group_by="label"),
            ),
            runner.run_trend(
                AnalyticsQuery(query_id="mq2", level=AnalyticsLevel.SYSTEM, window=AggregationWindow.DAILY),
                "match_rate",
            ),
        ]

        metrics = compute_analytics_metrics(reports)
        assert metrics["report_count"] == 2
        assert metrics["empty_report_rate"] == 0.0
        assert metrics["total_events_considered"] == len(events) * 2  # same engine, counted twice
