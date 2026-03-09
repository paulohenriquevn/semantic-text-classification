"""Classification orchestrator — composes feature extraction and classification.

Transforms ContextWindows (or Turns) into ClassificationInputs, runs them
through a Classifier, and maps results to domain Prediction entities.

The orchestrator is a thin composition layer:

    ContextWindow / Turn
        ↓
    feature extraction (lexical + structural)
        ↓
    ClassificationInput assembly (text + features + embedding)
        ↓
    Classifier.classify()
        ↓
    ClassificationResult → Prediction mapping
        ↓
    ClassificationBatchResult

Dependencies are injected — the orchestrator depends on the Classifier
protocol, not on concrete implementations.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from semantic_conversation_engine.classification.features import (
    extract_lexical_features,
    extract_structural_features,
    merge_feature_sets,
)
from semantic_conversation_engine.classification.models import (
    ClassificationInput,
    ClassificationResult,
    LabelScore,
)
from semantic_conversation_engine.models.context_window import ContextWindow
from semantic_conversation_engine.models.enums import ObjectType
from semantic_conversation_engine.models.prediction import Prediction
from semantic_conversation_engine.models.types import PredictionId
from semantic_conversation_engine.pipeline.protocols import Classifier


@dataclass(frozen=True)
class ClassificationBatchResult:
    """Execution envelope for a classification batch.

    Carries all predictions from a batch classification run along
    with operational statistics and source classification results.

    Args:
        predictions: Domain Prediction entities, one per positive label score.
        classification_results: Raw ClassificationResults from the classifier.
        stats: Operational statistics (timing, counts, classifier identity).
    """

    predictions: list[Prediction]
    classification_results: list[ClassificationResult]
    stats: dict[str, Any] = field(default_factory=dict)


def _source_type_to_object_type(source_type: str) -> ObjectType:
    """Map a source_type string to an ObjectType enum.

    Args:
        source_type: Source type string from ClassificationInput.

    Returns:
        Corresponding ObjectType.
    """
    mapping = {
        "turn": ObjectType.TURN,
        "context_window": ObjectType.CONTEXT_WINDOW,
        "conversation": ObjectType.CONVERSATION,
    }
    return mapping.get(source_type, ObjectType.CONTEXT_WINDOW)


def _label_score_to_prediction(
    label_score: LabelScore,
    source_id: str,
    source_type: ObjectType,
    model_name: str,
    model_version: str,
) -> Prediction:
    """Map a LabelScore to a domain Prediction entity.

    Args:
        label_score: The label score to convert.
        source_id: ID of the classified object.
        source_type: ObjectType of the classified object.
        model_name: Classifier model name.
        model_version: Classifier model version.

    Returns:
        A validated Prediction domain entity.
    """
    return Prediction(
        prediction_id=PredictionId(f"pred_{uuid.uuid4().hex[:12]}"),
        source_id=source_id,
        source_type=source_type,
        label=label_score.label,
        score=label_score.score,
        confidence=label_score.confidence,
        threshold=label_score.threshold,
        model_name=model_name,
        model_version=model_version,
    )


class ClassificationOrchestrator:
    """Orchestrates feature extraction, classification, and prediction mapping.

    Composes the full classification flow:
    1. Build ClassificationInputs from ContextWindows (with features + embeddings)
    2. Run classifier
    3. Map positive label scores to domain Predictions

    Args:
        classifier: Any object implementing the Classifier protocol.
    """

    def __init__(self, classifier: Classifier) -> None:
        self._classifier = classifier

    def classify_windows(
        self,
        windows: list[ContextWindow],
        embeddings: dict[str, list[float]] | None = None,
    ) -> ClassificationBatchResult:
        """Classify a batch of context windows.

        Extracts features from each window, attaches embeddings if
        available, runs classification, and maps results to Predictions.

        Args:
            windows: Context windows to classify.
            embeddings: Optional mapping from window_id to embedding vector.

        Returns:
            ClassificationBatchResult with predictions and stats.
        """
        t0 = time.perf_counter()

        inputs = self._build_inputs_from_windows(windows, embeddings)
        classification_results = self._classifier.classify(inputs)
        predictions = self._map_to_predictions(classification_results)

        stats = self._build_stats(
            t0=t0,
            num_inputs=len(inputs),
            num_results=len(classification_results),
            num_predictions=len(predictions),
            source_type="context_window",
        )

        return ClassificationBatchResult(
            predictions=predictions,
            classification_results=classification_results,
            stats=stats,
        )

    def classify_inputs(
        self,
        inputs: list[ClassificationInput],
    ) -> ClassificationBatchResult:
        """Classify pre-built ClassificationInputs.

        Use this when inputs are already assembled (e.g., for turn-level
        or conversation-level classification).

        Args:
            inputs: Pre-built classification inputs.

        Returns:
            ClassificationBatchResult with predictions and stats.
        """
        t0 = time.perf_counter()

        classification_results = self._classifier.classify(inputs)
        predictions = self._map_to_predictions(classification_results)

        source_types = {inp.source_type for inp in inputs}
        source_type = source_types.pop() if len(source_types) == 1 else "mixed"

        stats = self._build_stats(
            t0=t0,
            num_inputs=len(inputs),
            num_results=len(classification_results),
            num_predictions=len(predictions),
            source_type=source_type,
        )

        return ClassificationBatchResult(
            predictions=predictions,
            classification_results=classification_results,
            stats=stats,
        )

    def _build_inputs_from_windows(
        self,
        windows: list[ContextWindow],
        embeddings: dict[str, list[float]] | None,
    ) -> list[ClassificationInput]:
        """Build ClassificationInputs from context windows."""
        inputs: list[ClassificationInput] = []
        for window in windows:
            lexical = extract_lexical_features(window.window_text)
            metadata = window.metadata or {}
            speakers = metadata.get("speakers", {})

            structural = extract_structural_features(
                is_customer=bool(speakers.get("has_customer", False)),
                is_agent=bool(speakers.get("has_agent", False)),
                turn_count=window.window_size,
                speaker_count=len(speakers.get("distribution", {})),
            )

            merged = merge_feature_sets(lexical, structural)
            embedding = (
                embeddings.get(window.window_id, None) if embeddings else None
            )

            inputs.append(
                ClassificationInput(
                    source_id=window.window_id,
                    source_type="context_window",
                    text=window.window_text,
                    embedding=embedding,
                    features=merged.features,
                    metadata={
                        "conversation_id": window.conversation_id,
                        "start_index": window.start_index,
                        "end_index": window.end_index,
                    },
                )
            )
        return inputs

    def _map_to_predictions(
        self,
        results: list[ClassificationResult],
    ) -> list[Prediction]:
        """Map positive label scores to domain Predictions."""
        predictions: list[Prediction] = []
        for result in results:
            object_type = _source_type_to_object_type(result.source_type)
            for ls in result.label_scores:
                if ls.is_positive:
                    predictions.append(
                        _label_score_to_prediction(
                            label_score=ls,
                            source_id=result.source_id,
                            source_type=object_type,
                            model_name=result.model_name,
                            model_version=result.model_version,
                        )
                    )
        return predictions

    def _build_stats(
        self,
        *,
        t0: float,
        num_inputs: int,
        num_results: int,
        num_predictions: int,
        source_type: str,
    ) -> dict[str, Any]:
        """Build operational statistics."""
        return {
            "classification_level": source_type,
            "num_inputs": num_inputs,
            "num_results": num_results,
            "num_predictions": num_predictions,
            "classification_latency_ms": round(
                (time.perf_counter() - t0) * 1000, 2
            ),
        }
