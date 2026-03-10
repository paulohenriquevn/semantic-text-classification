"""System pipeline benchmark runner — compares pipeline configurations.

Executes the SystemPipeline under different scenarios (full vs partial,
with/without embeddings, with/without rules, etc.) and produces a
comparison report with per-stage and aggregated operational metrics.

The runner is pipeline-agnostic in the sense that it takes a callable
that produces SystemPipelineResult. This allows benchmarking different
pipeline configurations without coupling to construction details.

Results are collected into a SystemBenchmarkReport for comparison
and serialization (JSON/CSV).

Usage::

    runner = SystemBenchmarkRunner()
    report = runner.compare({
        "full": lambda: pipeline_full.run(transcript, ...),
        "no_rules": lambda: pipeline_no_rules.run(transcript, ...),
    })
    report.save_json("system_benchmark.json")
"""

from __future__ import annotations

import csv
import io
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from talkex.pipeline.metrics import compute_pipeline_metrics
from talkex.pipeline.system_pipeline import (
    SystemPipelineResult,
)


@dataclass(frozen=True)
class ScenarioResult:
    """Results for a single benchmark scenario.

    Args:
        scenario_name: Human-readable scenario name.
        scenario_params: Description of what this scenario tests.
        result: The SystemPipelineResult from execution.
        stage_latencies: Per-stage elapsed_ms for this run.
        stage_item_counts: Per-stage item_count for this run.
        stages_executed: Number of stages that actually ran.
        stages_skipped: Number of stages skipped.
        total_ms: Total execution time for this scenario.
    """

    scenario_name: str
    scenario_params: dict[str, str] = field(default_factory=dict)
    result: SystemPipelineResult | None = None
    stage_latencies: dict[str, float] = field(default_factory=dict)
    stage_item_counts: dict[str, int] = field(default_factory=dict)
    stages_executed: int = 0
    stages_skipped: int = 0
    total_ms: float = 0.0


@dataclass(frozen=True)
class SystemBenchmarkReport:
    """Comparison report across multiple pipeline benchmark scenarios.

    Args:
        experiment_name: Name of the benchmark experiment.
        experiment_version: Version for reproducibility.
        results: Per-scenario benchmark results.
        aggregated: Aggregated operational metrics across all scenarios.
        total_runs: Total number of pipeline executions.
        total_ms: Total benchmark execution time.
    """

    experiment_name: str
    experiment_version: str
    results: list[ScenarioResult] = field(default_factory=list)
    aggregated: dict[str, object] = field(default_factory=dict)
    total_runs: int = 0
    total_ms: float = 0.0

    def to_json(self) -> str:
        """Serialize benchmark report to JSON string.

        Returns:
            Formatted JSON string with per-scenario and aggregated metrics.
        """
        data: dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "experiment_version": self.experiment_version,
            "total_runs": self.total_runs,
            "total_ms": self.total_ms,
            "aggregated": self.aggregated,
            "results": [
                {
                    "scenario_name": r.scenario_name,
                    "scenario_params": r.scenario_params,
                    "stage_latencies": r.stage_latencies,
                    "stage_item_counts": r.stage_item_counts,
                    "stages_executed": r.stages_executed,
                    "stages_skipped": r.stages_skipped,
                    "total_ms": r.total_ms,
                }
                for r in self.results
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)

    def save_json(self, path: str | Path) -> None:
        """Save benchmark report as JSON file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_json())

    def to_csv(self) -> str:
        """Serialize per-scenario results to CSV string.

        One row per scenario with key metrics.

        Returns:
            CSV string with header and one row per scenario.
        """
        if not self.results:
            return ""

        # Collect all stage names across all results
        all_stages: list[str] = []
        for r in self.results:
            for name in r.stage_latencies:
                if name not in all_stages:
                    all_stages.append(name)

        header = [
            "scenario_name",
            "total_ms",
            "stages_executed",
            "stages_skipped",
        ]
        for stage in all_stages:
            header.append(f"{stage}_ms")
            header.append(f"{stage}_items")

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header)

        for r in self.results:
            row: list[str] = [
                r.scenario_name,
                str(r.total_ms),
                str(r.stages_executed),
                str(r.stages_skipped),
            ]
            for stage in all_stages:
                row.append(str(r.stage_latencies.get(stage, 0.0)))
                row.append(str(r.stage_item_counts.get(stage, 0)))
            writer.writerow(row)

        return output.getvalue()

    def save_csv(self, path: str | Path) -> None:
        """Save per-scenario results as CSV file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_csv())


@dataclass(frozen=True)
class SystemBenchmarkConfig:
    """Configuration for a system benchmark run.

    Args:
        experiment_name: Name for the benchmark experiment.
        experiment_version: Version string for reproducibility.
    """

    experiment_name: str = "system_pipeline_benchmark"
    experiment_version: str = "1.0"


@dataclass
class SystemBenchmarkRunner:
    """Runs system pipeline benchmark comparisons.

    Executes pipeline scenarios and produces comparison reports
    with per-stage and aggregated operational metrics.

    Args:
        config: Benchmark configuration.
    """

    config: SystemBenchmarkConfig = field(default_factory=SystemBenchmarkConfig)

    def run_scenario(
        self,
        scenario_fn: Callable[[], SystemPipelineResult],
        scenario_name: str,
        *,
        scenario_params: dict[str, str] | None = None,
    ) -> ScenarioResult:
        """Run a single benchmark scenario.

        Args:
            scenario_fn: Callable that executes the pipeline and returns
                a SystemPipelineResult.
            scenario_name: Human-readable name for this scenario.
            scenario_params: Optional description of scenario configuration.

        Returns:
            ScenarioResult with timing and output metrics.
        """
        start = time.monotonic()
        pipeline_result = scenario_fn()
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        stage_latencies = {s.name: s.elapsed_ms for s in pipeline_result.stages}
        stage_item_counts = {s.name: s.item_count for s in pipeline_result.stages}

        return ScenarioResult(
            scenario_name=scenario_name,
            scenario_params=scenario_params or {},
            result=pipeline_result,
            stage_latencies=stage_latencies,
            stage_item_counts=stage_item_counts,
            stages_executed=sum(1 for s in pipeline_result.stages if not s.skipped),
            stages_skipped=sum(1 for s in pipeline_result.stages if s.skipped),
            total_ms=elapsed_ms,
        )

    def compare(
        self,
        scenarios: dict[str, Callable[[], SystemPipelineResult]],
        *,
        scenario_params: dict[str, dict[str, str]] | None = None,
    ) -> SystemBenchmarkReport:
        """Run multiple scenarios and produce a comparison report.

        Args:
            scenarios: Mapping from scenario name to callable that
                executes the pipeline.
            scenario_params: Optional mapping from scenario name to
                parameter descriptions.

        Returns:
            SystemBenchmarkReport comparing all scenarios.
        """
        start = time.monotonic()
        results: list[ScenarioResult] = []
        params_map = scenario_params or {}

        for name, fn in scenarios.items():
            result = self.run_scenario(
                fn,
                name,
                scenario_params=params_map.get(name),
            )
            results.append(result)

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        # Aggregate metrics across all pipeline results
        pipeline_results = [r.result for r in results if r.result is not None]
        aggregated = compute_pipeline_metrics(pipeline_results)

        return SystemBenchmarkReport(
            experiment_name=self.config.experiment_name,
            experiment_version=self.config.experiment_version,
            results=results,
            aggregated=aggregated,
            total_runs=len(results),
            total_ms=elapsed_ms,
        )
