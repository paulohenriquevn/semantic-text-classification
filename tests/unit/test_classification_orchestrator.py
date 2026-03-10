"""Unit tests for ClassificationOrchestrator.

Tests cover: window classification, input classification, Prediction mapping,
multi-label, stats, batch processing, and reexport.
"""

from talkex.classification.models import (
    ClassificationInput,
    ClassificationResult,
    LabelScore,
)
from talkex.classification.orchestrator import (
    ClassificationBatchResult,
    ClassificationOrchestrator,
)
from talkex.models.context_window import ContextWindow
from talkex.models.enums import ObjectType
from talkex.models.types import WindowId

# ---------------------------------------------------------------------------
# Stub classifier (satisfies Classifier protocol via duck typing)
# ---------------------------------------------------------------------------


class _StubClassifier:
    """Returns fixed label scores for testing the orchestrator."""

    def __init__(
        self,
        label_scores: list[LabelScore] | None = None,
        model_name: str = "stub",
        model_version: str = "1.0",
    ) -> None:
        self._label_scores = label_scores or [
            LabelScore(label="billing", score=0.9, confidence=0.9, threshold=0.5),
            LabelScore(label="cancel", score=0.3, confidence=0.3, threshold=0.5),
        ]
        self._model_name = model_name
        self._model_version = model_version

    def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
        return [
            ClassificationResult(
                source_id=inp.source_id,
                source_type=inp.source_type,
                label_scores=list(self._label_scores),
                model_name=self._model_name,
                model_version=self._model_version,
            )
            for inp in inputs
        ]


def _make_window(
    window_id: str = "win_001",
    conversation_id: str = "conv_001",
    window_text: str = "Customer: I have a billing issue. Agent: Let me help.",
    window_size: int = 2,
) -> ContextWindow:
    return ContextWindow(
        window_id=WindowId(window_id),
        conversation_id=conversation_id,
        turn_ids=["turn_0", "turn_1"][:window_size],
        window_text=window_text,
        start_index=0,
        end_index=window_size - 1,
        window_size=window_size,
        stride=1,
        metadata={
            "speakers": {
                "has_customer": True,
                "has_agent": True,
                "distribution": {"Customer": 1, "Agent": 1},
            },
        },
    )


# ---------------------------------------------------------------------------
# Window classification
# ---------------------------------------------------------------------------


class TestClassifyWindows:
    def test_produces_batch_result(self) -> None:
        orch = ClassificationOrchestrator(classifier=_StubClassifier())
        result = orch.classify_windows([_make_window()])
        assert isinstance(result, ClassificationBatchResult)

    def test_produces_predictions_for_positive_labels(self) -> None:
        orch = ClassificationOrchestrator(classifier=_StubClassifier())
        result = orch.classify_windows([_make_window()])
        # billing=0.9 exceeds 0.5, cancel=0.3 does not
        assert len(result.predictions) == 1
        assert result.predictions[0].label == "billing"

    def test_prediction_has_correct_source(self) -> None:
        orch = ClassificationOrchestrator(classifier=_StubClassifier())
        result = orch.classify_windows([_make_window(window_id="win_42")])
        assert result.predictions[0].source_id == "win_42"
        assert result.predictions[0].source_type == ObjectType.CONTEXT_WINDOW

    def test_prediction_has_model_info(self) -> None:
        orch = ClassificationOrchestrator(classifier=_StubClassifier(model_name="test-clf", model_version="2.0"))
        result = orch.classify_windows([_make_window()])
        assert result.predictions[0].model_name == "test-clf"
        assert result.predictions[0].model_version == "2.0"

    def test_prediction_has_score_and_threshold(self) -> None:
        orch = ClassificationOrchestrator(classifier=_StubClassifier())
        result = orch.classify_windows([_make_window()])
        pred = result.predictions[0]
        assert pred.score == 0.9
        assert pred.confidence == 0.9
        assert pred.threshold == 0.5
        assert pred.is_above_threshold is True

    def test_classification_results_preserved(self) -> None:
        orch = ClassificationOrchestrator(classifier=_StubClassifier())
        result = orch.classify_windows([_make_window()])
        assert len(result.classification_results) == 1
        assert result.classification_results[0].top_label == "billing"

    def test_batch_multiple_windows(self) -> None:
        orch = ClassificationOrchestrator(classifier=_StubClassifier())
        windows = [
            _make_window(window_id="win_0"),
            _make_window(window_id="win_1"),
            _make_window(window_id="win_2"),
        ]
        result = orch.classify_windows(windows)
        assert len(result.classification_results) == 3
        assert len(result.predictions) == 3  # 1 positive per window
        source_ids = {p.source_id for p in result.predictions}
        assert source_ids == {"win_0", "win_1", "win_2"}

    def test_embeddings_attached_when_provided(self) -> None:
        """Verify that embeddings dict is passed through to inputs."""
        captured_inputs: list[ClassificationInput] = []

        class _CapturingClassifier:
            def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
                captured_inputs.extend(inputs)
                return [
                    ClassificationResult(
                        source_id=inp.source_id,
                        source_type=inp.source_type,
                        label_scores=[],
                        model_name="test",
                        model_version="1.0",
                    )
                    for inp in inputs
                ]

        orch = ClassificationOrchestrator(classifier=_CapturingClassifier())
        w = _make_window(window_id="win_emb")
        orch.classify_windows([w], embeddings={"win_emb": [0.1, 0.2, 0.3]})
        assert captured_inputs[0].embedding == [0.1, 0.2, 0.3]

    def test_no_embedding_when_not_provided(self) -> None:
        captured_inputs: list[ClassificationInput] = []

        class _CapturingClassifier:
            def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
                captured_inputs.extend(inputs)
                return [
                    ClassificationResult(
                        source_id=inp.source_id,
                        source_type=inp.source_type,
                        label_scores=[],
                        model_name="test",
                        model_version="1.0",
                    )
                    for inp in inputs
                ]

        orch = ClassificationOrchestrator(classifier=_CapturingClassifier())
        orch.classify_windows([_make_window()])
        assert captured_inputs[0].embedding is None

    def test_features_extracted_from_window(self) -> None:
        captured_inputs: list[ClassificationInput] = []

        class _CapturingClassifier:
            def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
                captured_inputs.extend(inputs)
                return [
                    ClassificationResult(
                        source_id=inp.source_id,
                        source_type=inp.source_type,
                        label_scores=[],
                        model_name="test",
                        model_version="1.0",
                    )
                    for inp in inputs
                ]

        orch = ClassificationOrchestrator(classifier=_CapturingClassifier())
        orch.classify_windows([_make_window()])
        features = captured_inputs[0].features
        # Should have lexical + structural features
        assert "word_count" in features
        assert "char_count" in features
        assert "is_customer" in features
        assert "is_agent" in features
        assert "turn_count" in features

    def test_metadata_includes_conversation_id(self) -> None:
        captured_inputs: list[ClassificationInput] = []

        class _CapturingClassifier:
            def classify(self, inputs: list[ClassificationInput]) -> list[ClassificationResult]:
                captured_inputs.extend(inputs)
                return [
                    ClassificationResult(
                        source_id=inp.source_id,
                        source_type=inp.source_type,
                        label_scores=[],
                        model_name="test",
                        model_version="1.0",
                    )
                    for inp in inputs
                ]

        orch = ClassificationOrchestrator(classifier=_CapturingClassifier())
        orch.classify_windows([_make_window(conversation_id="conv_42")])
        assert captured_inputs[0].metadata["conversation_id"] == "conv_42"


# ---------------------------------------------------------------------------
# Multi-label predictions
# ---------------------------------------------------------------------------


class TestMultiLabelPredictions:
    def test_multiple_positive_labels_produce_multiple_predictions(self) -> None:
        scores = [
            LabelScore(label="billing", score=0.9, confidence=0.9, threshold=0.5),
            LabelScore(label="cancel", score=0.7, confidence=0.7, threshold=0.5),
            LabelScore(label="refund", score=0.2, confidence=0.2, threshold=0.5),
        ]
        orch = ClassificationOrchestrator(classifier=_StubClassifier(label_scores=scores))
        result = orch.classify_windows([_make_window()])
        labels = [p.label for p in result.predictions]
        assert "billing" in labels
        assert "cancel" in labels
        assert "refund" not in labels
        assert len(result.predictions) == 2

    def test_no_positive_labels_produce_no_predictions(self) -> None:
        scores = [
            LabelScore(label="billing", score=0.3, confidence=0.3, threshold=0.5),
            LabelScore(label="cancel", score=0.1, confidence=0.1, threshold=0.5),
        ]
        orch = ClassificationOrchestrator(classifier=_StubClassifier(label_scores=scores))
        result = orch.classify_windows([_make_window()])
        assert len(result.predictions) == 0


# ---------------------------------------------------------------------------
# Pre-built input classification
# ---------------------------------------------------------------------------


class TestClassifyInputs:
    def test_classifies_pre_built_inputs(self) -> None:
        orch = ClassificationOrchestrator(classifier=_StubClassifier())
        inputs = [
            ClassificationInput(
                source_id="turn_0",
                source_type="turn",
                text="I want to cancel",
            )
        ]
        result = orch.classify_inputs(inputs)
        assert len(result.classification_results) == 1
        assert result.predictions[0].source_type == ObjectType.TURN

    def test_conversation_level_source_type(self) -> None:
        orch = ClassificationOrchestrator(classifier=_StubClassifier())
        inputs = [
            ClassificationInput(
                source_id="conv_0",
                source_type="conversation",
                text="Full conversation text",
            )
        ]
        result = orch.classify_inputs(inputs)
        assert result.predictions[0].source_type == ObjectType.CONVERSATION


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestOrchestratorStats:
    def test_stats_populated(self) -> None:
        orch = ClassificationOrchestrator(classifier=_StubClassifier())
        result = orch.classify_windows([_make_window()])
        assert "classification_latency_ms" in result.stats
        assert result.stats["num_inputs"] == 1
        assert result.stats["num_results"] == 1
        assert result.stats["num_predictions"] == 1
        assert result.stats["classification_level"] == "context_window"

    def test_stats_reflect_batch_size(self) -> None:
        orch = ClassificationOrchestrator(classifier=_StubClassifier())
        windows = [_make_window(window_id=f"w_{i}") for i in range(5)]
        result = orch.classify_windows(windows)
        assert result.stats["num_inputs"] == 5


# ---------------------------------------------------------------------------
# Prediction validation
# ---------------------------------------------------------------------------


class TestPredictionValidation:
    def test_prediction_id_has_pred_prefix(self) -> None:
        orch = ClassificationOrchestrator(classifier=_StubClassifier())
        result = orch.classify_windows([_make_window()])
        assert result.predictions[0].prediction_id.startswith("pred_")

    def test_each_prediction_has_unique_id(self) -> None:
        scores = [
            LabelScore(label="billing", score=0.9, confidence=0.9, threshold=0.5),
            LabelScore(label="cancel", score=0.7, confidence=0.7, threshold=0.5),
        ]
        orch = ClassificationOrchestrator(classifier=_StubClassifier(label_scores=scores))
        result = orch.classify_windows([_make_window()])
        ids = [p.prediction_id for p in result.predictions]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestOrchestratorReexport:
    def test_orchestrator_importable(self) -> None:
        from talkex.classification import (
            ClassificationOrchestrator as CO,
        )

        assert CO is ClassificationOrchestrator

    def test_batch_result_importable(self) -> None:
        from talkex.classification import (
            ClassificationBatchResult as CBR,
        )

        assert CBR is ClassificationBatchResult
