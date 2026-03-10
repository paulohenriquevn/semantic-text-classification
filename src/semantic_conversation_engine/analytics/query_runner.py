"""Analytics query runner — structured report generation from AnalyticsQuery.

Sits above the aggregation engine and transforms raw AggregationResult / TrendPoint
lists into structured AnalyticsReport objects with named sections and execution
metadata.

The query runner is the consumption boundary: it decides whether to call query()
or trend(), builds structured report sections, attaches execution metadata, and
produces serializable output.

Pipeline:
    AnalyticsQuery → engine.query() / engine.trend() → report sections → AnalyticsReport
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from semantic_conversation_engine.analytics.models import (
    AggregationResult,
    AnalyticsQuery,
    TrendPoint,
)
from semantic_conversation_engine.analytics.report import (
    AnalyticsReport,
    AnalyticsSection,
    GroupedMetric,
    TrendSeries,
)

if TYPE_CHECKING:
    from semantic_conversation_engine.analytics.aggregators import SimpleAnalyticsEngine


class AnalyticsQueryRunner:
    """Transforms analytics queries into structured reports.

    Orchestrates the SimpleAnalyticsEngine to produce AnalyticsReport objects
    with named sections, execution metadata, and serializable output.

    Args:
        engine: Analytics engine to execute queries against.
    """

    def __init__(self, engine: SimpleAnalyticsEngine) -> None:
        self._engine = engine

    def run_grouped(
        self,
        analytics_query: AnalyticsQuery,
        *,
        section_name: str = "Grouped Results",
        section_description: str = "",
    ) -> AnalyticsReport:
        """Execute a dimensional query and produce a structured report.

        Args:
            analytics_query: Query specifying filters, grouping, and limits.
            section_name: Name for the report section.
            section_description: Description for the report section.

        Returns:
            AnalyticsReport with one section containing grouped metrics.
        """
        start = time.monotonic()
        results = self._engine.query(analytics_query)
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        grouped_metrics = _aggregation_results_to_grouped_metrics(results)
        section = AnalyticsSection(
            name=section_name,
            description=section_description,
            grouped_metrics=grouped_metrics,
        )

        metadata = _build_query_metadata(analytics_query, elapsed_ms, len(results))
        metadata["total_events"] = self._engine.event_count

        return AnalyticsReport(
            report_name=f"Grouped: {analytics_query.group_by or 'metric_type'}",
            generated_at=datetime.now(tz=UTC).isoformat(),
            sections=[section],
            metadata=metadata,
        )

    def run_trend(
        self,
        analytics_query: AnalyticsQuery,
        metric_name: str,
        *,
        section_name: str = "Trend Results",
        section_description: str = "",
    ) -> AnalyticsReport:
        """Execute a temporal query and produce a structured report.

        Args:
            analytics_query: Query specifying filters, time range, and window.
            metric_name: Metric to track over time.
            section_name: Name for the report section.
            section_description: Description for the report section.

        Returns:
            AnalyticsReport with one section containing a trend series.
        """
        start = time.monotonic()
        points = self._engine.trend(analytics_query, metric_name)
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        series = _trend_points_to_series(points, metric_name, analytics_query.window.value)
        section = AnalyticsSection(
            name=section_name,
            description=section_description,
            trend_series=[series],
        )

        metadata = _build_query_metadata(analytics_query, elapsed_ms, len(points))
        metadata["total_events"] = self._engine.event_count
        metadata["metric_name"] = metric_name

        return AnalyticsReport(
            report_name=f"Trend: {metric_name}",
            generated_at=datetime.now(tz=UTC).isoformat(),
            sections=[section],
            metadata=metadata,
        )

    def run_composite(
        self,
        queries: list[tuple[AnalyticsQuery, str, str]],
        *,
        report_name: str = "Composite Report",
        trend_queries: list[tuple[AnalyticsQuery, str, str, str]] | None = None,
    ) -> AnalyticsReport:
        """Execute multiple queries and combine into a single report.

        Supports both dimensional and temporal queries in one report.

        Args:
            queries: List of (query, section_name, section_description) for grouped queries.
            report_name: Name for the composite report.
            trend_queries: Optional list of (query, metric_name, section_name, section_description)
                for temporal queries.

        Returns:
            AnalyticsReport with one section per query.
        """
        start = time.monotonic()
        sections: list[AnalyticsSection] = []

        for analytics_query, section_name, section_desc in queries:
            results = self._engine.query(analytics_query)
            grouped_metrics = _aggregation_results_to_grouped_metrics(results)
            sections.append(
                AnalyticsSection(
                    name=section_name,
                    description=section_desc,
                    grouped_metrics=grouped_metrics,
                )
            )

        for analytics_query, metric_name, section_name, section_desc in trend_queries or []:
            points = self._engine.trend(analytics_query, metric_name)
            series = _trend_points_to_series(points, metric_name, analytics_query.window.value)
            sections.append(
                AnalyticsSection(
                    name=section_name,
                    description=section_desc,
                    trend_series=[series],
                )
            )

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        metadata: dict[str, Any] = {
            "total_events": self._engine.event_count,
            "grouped_sections": len(queries),
            "trend_sections": len(trend_queries or []),
            "generation_time_ms": elapsed_ms,
        }

        return AnalyticsReport(
            report_name=report_name,
            generated_at=datetime.now(tz=UTC).isoformat(),
            sections=sections,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregation_results_to_grouped_metrics(
    results: list[AggregationResult],
) -> list[GroupedMetric]:
    """Flatten AggregationResults into a list of GroupedMetric for report consumption."""
    grouped: list[GroupedMetric] = []
    for result in results:
        for metric in result.metrics:
            grouped.append(
                GroupedMetric(
                    group_key=result.group_key,
                    group_value=result.group_value,
                    metric_name=metric.name,
                    value=metric.value,
                    event_count=result.event_count,
                )
            )
    return grouped


def _trend_points_to_series(
    points: list[TrendPoint],
    metric_name: str,
    window: str,
) -> TrendSeries:
    """Convert TrendPoints into a TrendSeries for report consumption."""
    return TrendSeries(
        metric_name=metric_name,
        window=window,
        points=[
            {
                "timestamp": p.timestamp.isoformat(),
                "value": p.value,
                "count": p.count,
            }
            for p in points
        ],
    )


def _build_query_metadata(
    analytics_query: AnalyticsQuery,
    elapsed_ms: float,
    result_count: int,
) -> dict[str, Any]:
    """Build execution metadata from query parameters and timing."""
    metadata: dict[str, Any] = {
        "query_id": analytics_query.query_id,
        "level": analytics_query.level.value,
        "window": analytics_query.window.value,
        "result_count": result_count,
        "generation_time_ms": elapsed_ms,
    }

    if analytics_query.group_by:
        metadata["group_by"] = analytics_query.group_by
    if analytics_query.metric_types:
        metadata["metric_types"] = [mt.value for mt in analytics_query.metric_types]
    if analytics_query.start_time:
        metadata["start_time"] = analytics_query.start_time.isoformat()
    if analytics_query.end_time:
        metadata["end_time"] = analytics_query.end_time.isoformat()
    if analytics_query.filters:
        metadata["filters"] = analytics_query.filters

    return metadata
