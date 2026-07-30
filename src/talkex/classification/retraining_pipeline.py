"""Retraining pipeline for the sentiment model (M7 D5/D6).

Consumes anonymized retraining samples (the Parquet cold sample, joined with QA labels) + a held-out
evaluation set, retrains a NEW `SentimentDetector` version (reusing the shipped detector — DRY), and
benchmarks its macro-F1 against the deployed model. Promotes the candidate ONLY on a strict gain — an
honest gate for the low-label-volume risk (ROADMAP risk 2): no gain → keep the deployed model, record it.
"""

from __future__ import annotations

from collections.abc import Callable

from talkex.classification.sentiment import SentimentDetector
from talkex.monitoring.domain.lifecycle import BenchmarkResult, RetrainingSample


class RetrainingPipeline:
    """Retrains + benchmarks a candidate sentiment model against the deployed one.

    The `detector_factory` (version → new detector) is injectable (DIP) so tests can supply a
    small-data-friendly pipeline; production defaults to the shipped TF-IDF word+char detector.
    """

    def __init__(self, detector_factory: Callable[[str], SentimentDetector] | None = None) -> None:
        self._make = detector_factory or (lambda version: SentimentDetector(model_version=version))

    def retrain_and_benchmark(
        self,
        samples: list[RetrainingSample],
        eval_texts: list[str],
        eval_labels: list[str],
        new_version: str,
        deployed: SentimentDetector | None = None,
    ) -> tuple[SentimentDetector, BenchmarkResult]:
        """Train a candidate on the labeled samples, benchmark vs `deployed`, and decide promotion.

        Only samples carrying a QA label are usable for supervised retraining; an unlabeled sample
        contributes text but no target, so it is skipped here (fail-fast if nothing is labeled).
        """
        labeled = [s for s in samples if s.label is not None]
        if not labeled:
            raise ValueError("no labeled samples to retrain on — QA labels are required")

        candidate = self._make(new_version)
        candidate.train([s.redacted_text for s in labeled], [str(s.label) for s in labeled])

        new_f1 = candidate.evaluate(eval_texts, eval_labels)
        deployed_f1 = deployed.evaluate(eval_texts, eval_labels) if deployed is not None else None
        # Promote when there is no incumbent (first model) or the candidate strictly improves.
        promoted = deployed_f1 is None or new_f1 > deployed_f1

        result = BenchmarkResult(
            new_model_version=new_version,
            new_f1=new_f1,
            deployed_f1=deployed_f1,
            promoted=promoted,
        )
        return candidate, result
