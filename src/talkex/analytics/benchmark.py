"""Analytics benchmark runner — compares query scenarios over analytics events.

Executes analytics queries under different configurations (group_by dimensions,
temporal windows, filters, composite vs simple) and produces a comparison
report with operational metrics per scenario.

The runner is engine-agnostic: any SimpleAnalyticsEngine (or compatible) can
be benchmarked. Results are collected into an AnalyticsBenchmarkReport for
comparison and serialization (JSON/CSV).

Usage::

    runner = AnalyticsBenchmarkRunner(engine=engine)
    report = runner.compare({
        "by_label": query_label,
        "by_channel": query_channel,
    })
    report.save_json("analytics_benchmark.json")
"""

from __future__ import annotations

import csv
import io
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from talkex.analytics.metrics import compute_analytics_metrics
from talkex.analytics.models import AnalyticsQuery
from talkex.analytics.query_runner import AnalyticsQueryRunner
from talkex.analytics.report import AnalyticsReport


@dataclass(frozen=True)
class QueryScenarioResult:
    """Results for a single analytics query scenario.

    Args:
        scenario_name: Human-readable name for this scenario.
        query_params: Query parameters used.
        report: Generated analytics report.
        group_count: Number of dimensional groups in the report.
        trend_point_count: Number of trend points in the report.
        generation_time_ms: Time to generate this report.
    """

    scenario_name: str
    query_params: dict[str, str]
    report: AnalyticsReport
    group_count: int
    trend_point_count: int
    generation_time_ms: float


@dataclass(frozen=True)
class AnalyticsBenchmarkReport:
    """Comparison report across multiple analytics query scenarios.

    Args:
        experiment_name: Name of the benchmark experiment.
        experiment_version: Version for reproducibility.
        results: Per-scenario query results.
        aggregated: Aggregated operational metrics across all scenarios.
        total_events: Total events in the engine.
        total_ms: Total benchmark execution time.
    """

    experiment_name: str
    experiment_version: str
    results: list[QueryScenarioResult] = field(default_factory=list)
    aggregated: dict[str, object] = field(default_factory=dict)
    total_events: int = 0
    total_ms: float = 0.0

    def to_json(self) -> str:
        """Serialize benchmark report to JSON string.

        Returns:
            Formatted JSON string with per-scenario and aggregated metrics.
        """
        data = {
            "experiment_name": self.experiment_name,
            "experiment_version": self.experiment_version,
            "total_events": self.total_events,
            "total_ms": self.total_ms,
            "aggregated": self.aggregated,
            "results": [
                {
                    "scenario_name": r.scenario_name,
                    "query_params": r.query_params,
                    "group_count": r.group_count,
                    "trend_point_count": r.trend_point_count,
                    "generation_time_ms": r.generation_time_ms,
                }
                for r in self.results
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

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
            CSV string with one row per scenario.
        """
        if not self.results:
            return ""

        header = [
            "scenario_name",
            "group_count",
            "trend_point_count",
            "generation_time_ms",
        ]

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header)

        for result in self.results:
            writer.writerow(
                [
                    result.scenario_name,
                    str(result.group_count),
                    str(result.trend_point_count),
                    str(result.generation_time_ms),
                ]
            )

        return output.getvalue()

    def save_csv(self, path: str | Path) -> None:
        """Save per-scenario results as CSV file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_csv())


@dataclass(frozen=True)
class AnalyticsBenchmarkConfig:
    """Configuration for an analytics benchmark run.

    Args:
        experiment_name: Name for the benchmark experiment.
        experiment_version: Version string for reproducibility.
    """

    experiment_name: str = "analytics_benchmark"
    experiment_version: str = "1.0"


@dataclass
class AnalyticsBenchmarkRunner:
    """Runs analytics query scenario benchmarks.

    Executes analytics queries under different configurations (group_by,
    window, filters) and produces comparison reports with operational metrics.

    Args:
        engine: Analytics engine containing events to query.
        config: Benchmark configuration.
    """

    engine: Any  # SimpleAnalyticsEngine — avoids circular import
    config: AnalyticsBenchmarkConfig = field(default_factory=AnalyticsBenchmarkConfig)

    def run_scenario(
        self,
        analytics_query: AnalyticsQuery,
        scenario_name: str,
        *,
        metric_name: str | None = None,
    ) -> QueryScenarioResult:
        """Run a single query scenario and measure it.

        If ``metric_name`` is provided, runs a trend query.
        Otherwise, runs a grouped query.

        Args:
            analytics_query: Query to execute.
            scenario_name: Human-readable name for this scenario.
            metric_name: If provided, runs a trend query for this metric.

        Returns:
            QueryScenarioResult with report and operational metrics.
        """
        runner = AnalyticsQueryRunner(self.engine)
        start = time.monotonic()

        if metric_name is not None:
            report = runner.run_trend(
                analytics_query,
                metric_name,
                section_name=scenario_name,
            )
        else:
            report = runner.run_grouped(
                analytics_query,
                section_name=scenario_name,
            )

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        group_count = sum(len(s.grouped_metrics) for s in report.sections)
        trend_point_count = sum(len(ts.points) for s in report.sections for ts in s.trend_series)

        query_params = _extract_query_params(analytics_query, metric_name)

        return QueryScenarioResult(
            scenario_name=scenario_name,
            query_params=query_params,
            report=report,
            group_count=group_count,
            trend_point_count=trend_point_count,
            generation_time_ms=elapsed_ms,
        )

    def compare(
        self,
        scenarios: Mapping[str, AnalyticsQuery],
        *,
        trend_scenarios: Mapping[str, tuple[AnalyticsQuery, str]] | None = None,
    ) -> AnalyticsBenchmarkReport:
        """Run multiple query scenarios and produce a comparison report.

        Args:
            scenarios: Mapping from scenario name to grouped query.
            trend_scenarios: Optional mapping from scenario name to
                (query, metric_name) for trend queries.

        Returns:
            AnalyticsBenchmarkReport comparing all scenarios.
        """
        start = time.monotonic()
        results: list[QueryScenarioResult] = []

        for scenario_name, analytics_query in scenarios.items():
            result = self.run_scenario(analytics_query, scenario_name)
            results.append(result)

        for scenario_name, (analytics_query, metric_name) in (trend_scenarios or {}).items():
            result = self.run_scenario(
                analytics_query,
                scenario_name,
                metric_name=metric_name,
            )
            results.append(result)

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        # Aggregate operational metrics from all reports
        all_reports = [r.report for r in results]
        aggregated = compute_analytics_metrics(all_reports)

        return AnalyticsBenchmarkReport(
            experiment_name=self.config.experiment_name,
            experiment_version=self.config.experiment_version,
            results=results,
            aggregated=aggregated,
            total_events=self.engine.event_count,
            total_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_query_params(
    analytics_query: AnalyticsQuery,
    metric_name: str | None,
) -> dict[str, str]:
    """Extract human-readable query parameters for traceability."""
    params: dict[str, str] = {
        "level": analytics_query.level.value,
        "window": analytics_query.window.value,
    }
    if analytics_query.group_by:
        params["group_by"] = analytics_query.group_by
    if analytics_query.metric_types:
        params["metric_types"] = ",".join(mt.value for mt in analytics_query.metric_types)
    if analytics_query.filters:
        params["filters"] = str(analytics_query.filters)
    if metric_name:
        params["metric_name"] = metric_name
    params["query_type"] = "trend" if metric_name else "grouped"
    return params
