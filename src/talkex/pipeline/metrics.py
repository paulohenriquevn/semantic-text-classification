"""System pipeline operational metrics — pure functions over SystemPipelineResult.

All functions take SystemPipelineResult objects and return aggregated
operational metrics about the system pipeline execution. These measure
cross-theme pipeline behavior, NOT individual subsystem quality (which
is handled by each theme's own metrics module).

Supported metrics:
    avg_total_pipeline_ms:     mean total pipeline latency
    avg_stage_latency_ms:      mean per-stage latency distribution
    stage_skip_rate:           fraction of stages skipped per run
    avg_outputs_per_stage:     mean item_count per stage across runs
    total_artifacts_produced:  sum of all artifacts across all runs
    compute_pipeline_metrics:  convenience aggregator returning all metrics

All functions are deterministic and stateless.
"""

from __future__ import annotations

from talkex.pipeline.system_pipeline import (
    SystemPipelineResult,
)


def avg_total_pipeline_ms(results: list[SystemPipelineResult]) -> float:
    """Mean total pipeline execution time in milliseconds.

    Args:
        results: Pipeline results to measure.

    Returns:
        Average total pipeline time. Returns 0.0 if results is empty.
    """
    if not results:
        return 0.0
    times = [float(r.stats.get("total_pipeline_ms", 0.0)) for r in results]
    return sum(times) / len(times)


def avg_stage_latency_ms(results: list[SystemPipelineResult]) -> dict[str, float]:
    """Mean per-stage latency across all runs.

    Args:
        results: Pipeline results to measure.

    Returns:
        Dictionary mapping stage name to average elapsed_ms.
        Returns empty dict if results is empty.
    """
    if not results:
        return {}

    stage_times: dict[str, list[float]] = {}
    for result in results:
        for stage in result.stages:
            stage_times.setdefault(stage.name, []).append(stage.elapsed_ms)

    return {name: sum(times) / len(times) for name, times in stage_times.items()}


def stage_skip_rate(results: list[SystemPipelineResult]) -> dict[str, float]:
    """Fraction of runs where each stage was skipped.

    Args:
        results: Pipeline results to measure.

    Returns:
        Dictionary mapping stage name to skip rate in [0.0, 1.0].
        Returns empty dict if results is empty.
    """
    if not results:
        return {}

    stage_counts: dict[str, int] = {}
    stage_skips: dict[str, int] = {}

    for result in results:
        for stage in result.stages:
            stage_counts[stage.name] = stage_counts.get(stage.name, 0) + 1
            if stage.skipped:
                stage_skips[stage.name] = stage_skips.get(stage.name, 0) + 1

    return {name: stage_skips.get(name, 0) / count for name, count in stage_counts.items()}


def avg_outputs_per_stage(results: list[SystemPipelineResult]) -> dict[str, float]:
    """Mean number of items produced by each stage across runs.

    Args:
        results: Pipeline results to measure.

    Returns:
        Dictionary mapping stage name to average item_count.
        Returns empty dict if results is empty.
    """
    if not results:
        return {}

    stage_items: dict[str, list[int]] = {}
    for result in results:
        for stage in result.stages:
            stage_items.setdefault(stage.name, []).append(stage.item_count)

    return {name: sum(items) / len(items) for name, items in stage_items.items()}


def total_artifacts_produced(results: list[SystemPipelineResult]) -> dict[str, int]:
    """Total count of each artifact type produced across all runs.

    Args:
        results: Pipeline results to measure.

    Returns:
        Dictionary with artifact counts: turns, windows, embeddings,
        predictions, rule_executions, analytics_events.
    """
    totals: dict[str, int] = {
        "turns": 0,
        "windows": 0,
        "embeddings": 0,
        "predictions": 0,
        "rule_executions": 0,
        "analytics_events": 0,
    }
    for result in results:
        totals["turns"] += len(result.pipeline_result.turns)
        totals["windows"] += len(result.pipeline_result.windows)
        totals["embeddings"] += len(result.embeddings)
        if result.classification:
            totals["predictions"] += len(result.classification.predictions)
        totals["rule_executions"] += len(result.rule_executions)
        totals["analytics_events"] += len(result.analytics_events)
    return totals


def compute_pipeline_metrics(
    results: list[SystemPipelineResult],
) -> dict[str, object]:
    """Compute all operational metrics for a batch of system pipeline results.

    Convenience function that calls all individual metric functions
    and returns a single aggregated dictionary.

    Args:
        results: Pipeline results to measure.

    Returns:
        Dictionary with all operational metrics.
    """
    return {
        "run_count": len(results),
        "avg_total_pipeline_ms": round(avg_total_pipeline_ms(results), 4),
        "avg_stage_latency_ms": {k: round(v, 4) for k, v in avg_stage_latency_ms(results).items()},
        "stage_skip_rate": {k: round(v, 4) for k, v in stage_skip_rate(results).items()},
        "avg_outputs_per_stage": {k: round(v, 4) for k, v in avg_outputs_per_stage(results).items()},
        "total_artifacts": total_artifacts_produced(results),
    }
