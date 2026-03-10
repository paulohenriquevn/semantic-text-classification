"""Analytics operational metrics — pure functions over analytics reports.

All functions take AnalyticsReport objects or component lists and return
aggregated operational metrics about the analytics subsystem itself.
These measure query behavior and report quality, NOT pipeline quality
(which is handled by the aggregation layer).

Supported metrics:
    avg_generation_time_ms:   mean report generation latency
    avg_result_count:         mean number of groups/points per report
    empty_report_rate:        fraction of reports with no results
    avg_group_count:          mean number of dimensional groups per section
    avg_trend_point_count:    mean number of trend points per section
    compute_analytics_metrics: convenience aggregator returning all metrics

All functions are deterministic and stateless.
"""

from __future__ import annotations

from talkex.analytics.report import AnalyticsReport


def avg_generation_time_ms(reports: list[AnalyticsReport]) -> float:
    """Mean report generation latency in milliseconds.

    Reads ``generation_time_ms`` from report metadata.

    Args:
        reports: Analytics reports to measure.

    Returns:
        Average generation time. Returns 0.0 if reports is empty.
    """
    if not reports:
        return 0.0
    times = [float(r.metadata.get("generation_time_ms", 0.0)) for r in reports]
    return sum(times) / len(times)


def avg_result_count(reports: list[AnalyticsReport]) -> float:
    """Mean number of result items (groups + trend points) per report.

    Args:
        reports: Analytics reports to measure.

    Returns:
        Average result count. Returns 0.0 if reports is empty.
    """
    if not reports:
        return 0.0
    counts = [_count_results(r) for r in reports]
    return sum(counts) / len(counts)


def empty_report_rate(reports: list[AnalyticsReport]) -> float:
    """Fraction of reports with no results (no groups and no trend points).

    Args:
        reports: Analytics reports to measure.

    Returns:
        Empty rate in [0.0, 1.0]. Returns 0.0 if reports is empty.
    """
    if not reports:
        return 0.0
    empty_count = sum(1 for r in reports if _count_results(r) == 0)
    return empty_count / len(reports)


def avg_group_count(reports: list[AnalyticsReport]) -> float:
    """Mean number of dimensional groups per report.

    Args:
        reports: Analytics reports to measure.

    Returns:
        Average group count. Returns 0.0 if reports is empty.
    """
    if not reports:
        return 0.0
    counts = [sum(len(s.grouped_metrics) for s in r.sections) for r in reports]
    return sum(counts) / len(counts)


def avg_trend_point_count(reports: list[AnalyticsReport]) -> float:
    """Mean number of trend points per report.

    Args:
        reports: Analytics reports to measure.

    Returns:
        Average trend point count. Returns 0.0 if reports is empty.
    """
    if not reports:
        return 0.0
    counts = [sum(len(ts.points) for s in r.sections for ts in s.trend_series) for r in reports]
    return sum(counts) / len(counts)


def total_events_considered(reports: list[AnalyticsReport]) -> int:
    """Total events considered across all reports.

    Reads ``total_events`` from report metadata. When multiple reports
    share the same engine, this may double-count — useful as a volume metric.

    Args:
        reports: Analytics reports to measure.

    Returns:
        Sum of total_events across all reports.
    """
    return sum(r.metadata.get("total_events", 0) for r in reports)


def compute_analytics_metrics(reports: list[AnalyticsReport]) -> dict[str, object]:
    """Compute all operational metrics for a batch of analytics reports.

    Convenience function that calls all individual metric functions
    and returns a single aggregated dictionary.

    Args:
        reports: Analytics reports to measure.

    Returns:
        Dictionary with all operational metrics.
    """
    return {
        "report_count": len(reports),
        "avg_generation_time_ms": round(avg_generation_time_ms(reports), 4),
        "avg_result_count": round(avg_result_count(reports), 4),
        "empty_report_rate": round(empty_report_rate(reports), 4),
        "avg_group_count": round(avg_group_count(reports), 4),
        "avg_trend_point_count": round(avg_trend_point_count(reports), 4),
        "total_events_considered": total_events_considered(reports),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_results(report: AnalyticsReport) -> int:
    """Count total result items in a report (groups + trend points)."""
    groups = sum(len(s.grouped_metrics) for s in report.sections)
    points = sum(len(ts.points) for s in report.sections for ts in s.trend_series)
    return groups + points
