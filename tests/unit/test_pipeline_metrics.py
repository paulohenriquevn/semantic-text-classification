"""Unit tests for system pipeline operational metrics.

Tests cover: avg_total_pipeline_ms, avg_stage_latency_ms, stage_skip_rate,
avg_outputs_per_stage, total_artifacts_produced, compute_pipeline_metrics,
and edge cases (empty inputs).
"""

from __future__ import annotations

from datetime import UTC, datetime

from talkex.classification.orchestrator import (
    ClassificationBatchResult,
)
from talkex.models.context_window import ContextWindow
from talkex.models.conversation import Conversation
from talkex.models.embedding_record import EmbeddingRecord
from talkex.models.enums import (
    Channel,
    ObjectType,
    PoolingStrategy,
    SpeakerRole,
)
from talkex.models.prediction import Prediction
from talkex.models.rule_execution import EvidenceItem, RuleExecution
from talkex.models.turn import Turn
from talkex.models.types import (
    EmbeddingId,
    PredictionId,
    RuleId,
)
from talkex.pipeline.metrics import (
    avg_outputs_per_stage,
    avg_stage_latency_ms,
    avg_total_pipeline_ms,
    compute_pipeline_metrics,
    stage_skip_rate,
    total_artifacts_produced,
)
from talkex.pipeline.result import PipelineResult
from talkex.pipeline.system_pipeline import (
    StageResult,
    SystemPipelineResult,
)


def _make_pipeline_result(
    *,
    turn_count: int = 2,
    window_count: int = 1,
) -> PipelineResult:
    """Build a minimal PipelineResult."""
    conv = Conversation(
        conversation_id="conv_test",
        channel=Channel.VOICE,
        start_time=datetime(2025, 1, 1, tzinfo=UTC),
    )
    turns = [
        Turn(
            turn_id=f"conv_test_turn_{i}",
            conversation_id="conv_test",
            speaker=SpeakerRole.CUSTOMER if i % 2 == 0 else SpeakerRole.AGENT,
            raw_text=f"Turn {i} text",
            start_offset=i * 20,
            end_offset=(i + 1) * 20,
        )
        for i in range(turn_count)
    ]
    windows = [
        ContextWindow(
            window_id=f"conv_test_win_{i}",
            conversation_id="conv_test",
            turn_ids=[f"conv_test_turn_{i}"],
            window_text=f"Window {i} text",
            start_index=i,
            end_index=i,
            window_size=1,
            stride=1,
        )
        for i in range(window_count)
    ]
    return PipelineResult(
        conversation=conv,
        turns=turns,
        windows=windows,
        coverage={"coverage_ratio": 1.0},
    )


def _make_system_result(
    *,
    total_ms: float = 10.0,
    stages: list[StageResult] | None = None,
    embedding_count: int = 0,
    prediction_count: int = 0,
    rule_count: int = 0,
    event_count: int = 0,
    turn_count: int = 2,
    window_count: int = 1,
) -> SystemPipelineResult:
    """Build a SystemPipelineResult with configurable outputs."""
    if stages is None:
        stages = [
            StageResult(name="text_processing", elapsed_ms=3.0, item_count=window_count),
            StageResult(name="embedding", elapsed_ms=2.0, item_count=embedding_count),
            StageResult(name="indexing", elapsed_ms=1.0, item_count=0, skipped=True),
            StageResult(name="classification", elapsed_ms=2.0, item_count=prediction_count),
            StageResult(name="rule_evaluation", elapsed_ms=1.5, item_count=rule_count),
            StageResult(name="analytics", elapsed_ms=0.5, item_count=event_count),
        ]

    embeddings = [
        EmbeddingRecord(
            embedding_id=EmbeddingId(f"emb_{i}"),
            source_id=f"win_{i}",
            source_type=ObjectType.CONTEXT_WINDOW,
            model_name="test",
            model_version="1.0",
            pooling_strategy=PoolingStrategy.MEAN,
            dimensions=3,
            vector=[0.1, 0.2, 0.3],
        )
        for i in range(embedding_count)
    ]

    predictions = [
        Prediction(
            prediction_id=PredictionId(f"pred_{i}"),
            source_id=f"win_{i}",
            source_type=ObjectType.CONTEXT_WINDOW,
            label="test_label",
            score=0.9,
            confidence=0.9,
            threshold=0.5,
            model_name="test",
            model_version="1.0",
        )
        for i in range(prediction_count)
    ]

    classification = (
        ClassificationBatchResult(
            predictions=predictions,
            classification_results=[],
        )
        if prediction_count > 0
        else None
    )

    rule_executions = [
        RuleExecution(
            rule_id=RuleId(f"rule_{i}"),
            rule_name=f"rule_{i}",
            source_id=f"win_{i}",
            source_type=ObjectType.CONTEXT_WINDOW,
            matched=True,
            score=1.0,
            execution_time_ms=0.5,
            evidence=[EvidenceItem(predicate_type="lexical", matched_text="test")],
        )
        for i in range(rule_count)
    ]

    return SystemPipelineResult(
        pipeline_result=_make_pipeline_result(turn_count=turn_count, window_count=window_count),
        embeddings=embeddings,
        classification=classification,
        rule_executions=rule_executions,
        analytics_events=[],  # Simplified — count tracked via event_count param
        stages=stages,
        stats={"total_pipeline_ms": total_ms},
    )


# ---------------------------------------------------------------------------
# avg_total_pipeline_ms
# ---------------------------------------------------------------------------


class TestAvgTotalPipelineMs:
    def test_empty(self) -> None:
        assert avg_total_pipeline_ms([]) == 0.0

    def test_single_result(self) -> None:
        r = _make_system_result(total_ms=15.5)
        assert avg_total_pipeline_ms([r]) == 15.5

    def test_multiple_results(self) -> None:
        r1 = _make_system_result(total_ms=10.0)
        r2 = _make_system_result(total_ms=20.0)
        assert avg_total_pipeline_ms([r1, r2]) == 15.0


# ---------------------------------------------------------------------------
# avg_stage_latency_ms
# ---------------------------------------------------------------------------


class TestAvgStageLatencyMs:
    def test_empty(self) -> None:
        assert avg_stage_latency_ms([]) == {}

    def test_single_result(self) -> None:
        r = _make_system_result()
        latencies = avg_stage_latency_ms([r])
        assert "text_processing" in latencies
        assert latencies["text_processing"] == 3.0

    def test_averages_across_runs(self) -> None:
        stages1 = [StageResult(name="text_processing", elapsed_ms=2.0, item_count=1)]
        stages2 = [StageResult(name="text_processing", elapsed_ms=4.0, item_count=1)]
        r1 = _make_system_result(stages=stages1)
        r2 = _make_system_result(stages=stages2)
        latencies = avg_stage_latency_ms([r1, r2])
        assert latencies["text_processing"] == 3.0


# ---------------------------------------------------------------------------
# stage_skip_rate
# ---------------------------------------------------------------------------


class TestStageSkipRate:
    def test_empty(self) -> None:
        assert stage_skip_rate([]) == {}

    def test_no_skips(self) -> None:
        stages = [
            StageResult(name="text_processing", elapsed_ms=1.0, item_count=1),
            StageResult(name="embedding", elapsed_ms=1.0, item_count=1),
        ]
        r = _make_system_result(stages=stages)
        rates = stage_skip_rate([r])
        assert rates["text_processing"] == 0.0
        assert rates["embedding"] == 0.0

    def test_with_skips(self) -> None:
        stages1 = [
            StageResult(name="embedding", elapsed_ms=1.0, item_count=1),
        ]
        stages2 = [
            StageResult(name="embedding", elapsed_ms=0.0, skipped=True),
        ]
        r1 = _make_system_result(stages=stages1)
        r2 = _make_system_result(stages=stages2)
        rates = stage_skip_rate([r1, r2])
        assert rates["embedding"] == 0.5


# ---------------------------------------------------------------------------
# avg_outputs_per_stage
# ---------------------------------------------------------------------------


class TestAvgOutputsPerStage:
    def test_empty(self) -> None:
        assert avg_outputs_per_stage([]) == {}

    def test_single_result(self) -> None:
        stages = [
            StageResult(name="text_processing", elapsed_ms=1.0, item_count=5),
            StageResult(name="embedding", elapsed_ms=1.0, item_count=3),
        ]
        r = _make_system_result(stages=stages)
        outputs = avg_outputs_per_stage([r])
        assert outputs["text_processing"] == 5.0
        assert outputs["embedding"] == 3.0

    def test_averages_across_runs(self) -> None:
        stages1 = [StageResult(name="embedding", elapsed_ms=1.0, item_count=2)]
        stages2 = [StageResult(name="embedding", elapsed_ms=1.0, item_count=4)]
        r1 = _make_system_result(stages=stages1)
        r2 = _make_system_result(stages=stages2)
        outputs = avg_outputs_per_stage([r1, r2])
        assert outputs["embedding"] == 3.0


# ---------------------------------------------------------------------------
# total_artifacts_produced
# ---------------------------------------------------------------------------


class TestTotalArtifactsProduced:
    def test_empty(self) -> None:
        totals = total_artifacts_produced([])
        assert all(v == 0 for v in totals.values())

    def test_counts_all_artifact_types(self) -> None:
        r = _make_system_result(
            embedding_count=3,
            prediction_count=2,
            rule_count=4,
            turn_count=5,
            window_count=3,
        )
        totals = total_artifacts_produced([r])
        assert totals["turns"] == 5
        assert totals["windows"] == 3
        assert totals["embeddings"] == 3
        assert totals["predictions"] == 2
        assert totals["rule_executions"] == 4

    def test_sums_across_runs(self) -> None:
        r1 = _make_system_result(embedding_count=2, turn_count=3, window_count=1)
        r2 = _make_system_result(embedding_count=3, turn_count=4, window_count=2)
        totals = total_artifacts_produced([r1, r2])
        assert totals["embeddings"] == 5
        assert totals["turns"] == 7
        assert totals["windows"] == 3


# ---------------------------------------------------------------------------
# compute_pipeline_metrics
# ---------------------------------------------------------------------------


class TestComputePipelineMetrics:
    def test_empty(self) -> None:
        m = compute_pipeline_metrics([])
        assert m["run_count"] == 0
        assert m["avg_total_pipeline_ms"] == 0.0

    def test_full_metrics(self) -> None:
        r = _make_system_result(
            total_ms=12.0,
            embedding_count=3,
            prediction_count=2,
            rule_count=1,
        )
        m = compute_pipeline_metrics([r])
        assert m["run_count"] == 1
        assert m["avg_total_pipeline_ms"] == 12.0
        assert "text_processing" in m["avg_stage_latency_ms"]
        assert "embedding" in m["stage_skip_rate"]
        assert "total_artifacts" in m
        assert m["total_artifacts"]["embeddings"] == 3
