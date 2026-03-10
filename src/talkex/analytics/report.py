"""Analytics reporting — structured output for analytical queries.

Provides data structures for structured analytics reports with JSON/CSV export.
Reports combine dimensional groupings and temporal trends into named sections,
carrying execution metadata for auditability.

Report hierarchy:
    GroupedMetric  — single metric within a dimensional group
    TrendSeries   — time-series of a single metric
    AnalyticsSection — named section with grouped and/or temporal results
    AnalyticsReport  — complete report with sections, metadata, and export

Reports are deterministic and serializable for reproducibility.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GroupedMetric:
    """Single metric within a dimensional group.

    Represents one computed metric for a specific dimension value,
    ready for report consumption.

    Args:
        group_key: Dimension name (e.g., "channel", "label").
        group_value: Dimension value (e.g., "voice", "billing").
        metric_name: Name of the metric (e.g., "match_rate", "avg_value").
        value: Numeric value of the metric.
        event_count: Number of events in this group.
    """

    group_key: str
    group_value: str
    metric_name: str
    value: float
    event_count: int


@dataclass(frozen=True)
class TrendSeries:
    """Time-series of a single metric.

    Represents an ordered sequence of metric values over time,
    ready for visualization or drift detection.

    Args:
        metric_name: Name of the tracked metric.
        window: Temporal bucket size used.
        points: Ordered (timestamp, value, count) tuples.
    """

    metric_name: str
    window: str
    points: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AnalyticsSection:
    """Named section within an analytics report.

    Groups related dimensional and/or temporal results under a
    descriptive name (e.g., "Classification by Label", "Rule Match Rate Trend").

    Args:
        name: Human-readable section title.
        description: Brief explanation of what this section shows.
        grouped_metrics: Dimensional results in this section.
        trend_series: Temporal results in this section.
    """

    name: str
    description: str = ""
    grouped_metrics: list[GroupedMetric] = field(default_factory=list)
    trend_series: list[TrendSeries] = field(default_factory=list)


@dataclass(frozen=True)
class AnalyticsReport:
    """Complete analytics report with sections, metadata, and export.

    Top-level container for structured analytics output. Combines
    dimensional groupings and temporal trends into named sections,
    with execution metadata for auditability.

    Args:
        report_name: Human-readable report title.
        generated_at: When the report was generated (ISO 8601).
        sections: Named report sections.
        metadata: Execution metadata (filters, time range, event count, etc.).
    """

    report_name: str
    generated_at: str
    sections: list[AnalyticsSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize report to JSON string.

        Returns:
            Formatted JSON string with all sections, metrics, and metadata.
        """
        data = {
            "report_name": self.report_name,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
            "sections": [
                {
                    "name": s.name,
                    "description": s.description,
                    "grouped_metrics": [
                        {
                            "group_key": gm.group_key,
                            "group_value": gm.group_value,
                            "metric_name": gm.metric_name,
                            "value": gm.value,
                            "event_count": gm.event_count,
                        }
                        for gm in s.grouped_metrics
                    ],
                    "trend_series": [
                        {
                            "metric_name": ts.metric_name,
                            "window": ts.window,
                            "points": ts.points,
                        }
                        for ts in s.trend_series
                    ],
                }
                for s in self.sections
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def save_json(self, path: str | Path) -> None:
        """Save report as JSON file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_json())

    def to_csv(self) -> str:
        """Serialize grouped metrics to CSV string.

        One row per grouped metric across all sections.

        Returns:
            CSV string with section, group key/value, metric name, value, event count.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["section", "group_key", "group_value", "metric_name", "value", "event_count"])

        for section in self.sections:
            for gm in section.grouped_metrics:
                writer.writerow(
                    [
                        section.name,
                        gm.group_key,
                        gm.group_value,
                        gm.metric_name,
                        str(gm.value),
                        str(gm.event_count),
                    ]
                )

        return output.getvalue()

    def save_csv(self, path: str | Path) -> None:
        """Save grouped metrics as CSV file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_csv())

    def to_trend_csv(self) -> str:
        """Serialize trend series to CSV string.

        One row per trend point across all sections.

        Returns:
            CSV string with section, metric name, window, timestamp, value, count.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["section", "metric_name", "window", "timestamp", "value", "count"])

        for section in self.sections:
            for ts in section.trend_series:
                for point in ts.points:
                    writer.writerow(
                        [
                            section.name,
                            ts.metric_name,
                            ts.window,
                            str(point.get("timestamp", "")),
                            str(point.get("value", "")),
                            str(point.get("count", "")),
                        ]
                    )

        return output.getvalue()

    def save_trend_csv(self, path: str | Path) -> None:
        """Save trend series as CSV file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_trend_csv())
