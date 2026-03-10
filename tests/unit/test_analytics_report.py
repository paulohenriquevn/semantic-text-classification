"""Unit tests for analytics report — structured output for analytical queries.

Tests cover: GroupedMetric, TrendSeries, AnalyticsSection, AnalyticsReport
construction, frozen immutability, JSON/CSV serialization, and reexports.
"""

import json

from semantic_conversation_engine.analytics.report import (
    AnalyticsReport,
    AnalyticsSection,
    GroupedMetric,
    TrendSeries,
)

# ---------------------------------------------------------------------------
# GroupedMetric
# ---------------------------------------------------------------------------


class TestGroupedMetric:
    def test_construction(self) -> None:
        gm = GroupedMetric(
            group_key="channel",
            group_value="voice",
            metric_name="match_rate",
            value=0.85,
            event_count=100,
        )
        assert gm.group_key == "channel"
        assert gm.group_value == "voice"
        assert gm.metric_name == "match_rate"
        assert gm.value == 0.85
        assert gm.event_count == 100

    def test_frozen(self) -> None:
        gm = GroupedMetric(
            group_key="k",
            group_value="v",
            metric_name="m",
            value=1.0,
            event_count=1,
        )
        try:
            gm.value = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# TrendSeries
# ---------------------------------------------------------------------------


class TestTrendSeries:
    def test_construction(self) -> None:
        ts = TrendSeries(
            metric_name="match_rate",
            window="daily",
            points=[
                {"timestamp": "2026-03-10T00:00:00+00:00", "value": 0.8, "count": 50},
                {"timestamp": "2026-03-11T00:00:00+00:00", "value": 0.9, "count": 60},
            ],
        )
        assert ts.metric_name == "match_rate"
        assert ts.window == "daily"
        assert len(ts.points) == 2

    def test_defaults(self) -> None:
        ts = TrendSeries(metric_name="avg_value", window="hourly")
        assert ts.points == []

    def test_frozen(self) -> None:
        ts = TrendSeries(metric_name="test", window="daily")
        try:
            ts.metric_name = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# AnalyticsSection
# ---------------------------------------------------------------------------


class TestAnalyticsSection:
    def test_construction_with_grouped(self) -> None:
        gm = GroupedMetric(
            group_key="label",
            group_value="billing",
            metric_name="event_count",
            value=42.0,
            event_count=42,
        )
        section = AnalyticsSection(
            name="By Label",
            description="Classification by label",
            grouped_metrics=[gm],
        )
        assert section.name == "By Label"
        assert section.description == "Classification by label"
        assert len(section.grouped_metrics) == 1
        assert section.trend_series == []

    def test_construction_with_trend(self) -> None:
        ts = TrendSeries(metric_name="match_rate", window="daily", points=[])
        section = AnalyticsSection(
            name="Match Rate Trend",
            trend_series=[ts],
        )
        assert len(section.trend_series) == 1
        assert section.grouped_metrics == []

    def test_defaults(self) -> None:
        section = AnalyticsSection(name="Empty")
        assert section.description == ""
        assert section.grouped_metrics == []
        assert section.trend_series == []

    def test_frozen(self) -> None:
        section = AnalyticsSection(name="Test")
        try:
            section.name = "Other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# AnalyticsReport
# ---------------------------------------------------------------------------


class TestAnalyticsReport:
    def _make_report(self) -> AnalyticsReport:
        gm1 = GroupedMetric("channel", "voice", "match_rate", 0.85, 100)
        gm2 = GroupedMetric("channel", "chat", "match_rate", 0.72, 80)
        ts = TrendSeries(
            metric_name="avg_value",
            window="daily",
            points=[
                {"timestamp": "2026-03-10T00:00:00+00:00", "value": 0.8, "count": 50},
            ],
        )
        return AnalyticsReport(
            report_name="Test Report",
            generated_at="2026-03-10T12:00:00+00:00",
            sections=[
                AnalyticsSection(
                    name="By Channel",
                    description="Grouped by channel",
                    grouped_metrics=[gm1, gm2],
                ),
                AnalyticsSection(
                    name="Value Trend",
                    description="Avg value over time",
                    trend_series=[ts],
                ),
            ],
            metadata={"total_events": 180, "generation_time_ms": 1.5},
        )

    def test_construction(self) -> None:
        report = self._make_report()
        assert report.report_name == "Test Report"
        assert len(report.sections) == 2
        assert report.metadata["total_events"] == 180

    def test_defaults(self) -> None:
        report = AnalyticsReport(
            report_name="Empty",
            generated_at="2026-03-10T12:00:00+00:00",
        )
        assert report.sections == []
        assert report.metadata == {}

    def test_frozen(self) -> None:
        report = AnalyticsReport(
            report_name="Test",
            generated_at="2026-03-10T12:00:00+00:00",
        )
        try:
            report.report_name = "Other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_to_json(self) -> None:
        report = self._make_report()
        json_str = report.to_json()
        data = json.loads(json_str)

        assert data["report_name"] == "Test Report"
        assert data["generated_at"] == "2026-03-10T12:00:00+00:00"
        assert len(data["sections"]) == 2
        assert len(data["sections"][0]["grouped_metrics"]) == 2
        assert data["sections"][0]["grouped_metrics"][0]["group_value"] == "voice"
        assert len(data["sections"][1]["trend_series"]) == 1
        assert data["metadata"]["total_events"] == 180

    def test_to_json_roundtrip(self) -> None:
        report = self._make_report()
        json_str = report.to_json()
        data = json.loads(json_str)
        # Re-serialize to verify stability
        json_str2 = json.dumps(data, indent=2, ensure_ascii=False)
        assert json.loads(json_str2) == data

    def test_to_csv(self) -> None:
        report = self._make_report()
        csv_str = report.to_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 3  # header + 2 grouped metrics
        assert "section" in lines[0]
        assert "voice" in lines[1]
        assert "chat" in lines[2]

    def test_to_csv_empty_report(self) -> None:
        report = AnalyticsReport(
            report_name="Empty",
            generated_at="2026-03-10T12:00:00+00:00",
        )
        csv_str = report.to_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 1  # header only

    def test_to_trend_csv(self) -> None:
        report = self._make_report()
        csv_str = report.to_trend_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 2  # header + 1 trend point
        assert "metric_name" in lines[0]
        assert "avg_value" in lines[1]

    def test_to_trend_csv_empty(self) -> None:
        report = AnalyticsReport(
            report_name="Empty",
            generated_at="2026-03-10T12:00:00+00:00",
        )
        csv_str = report.to_trend_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 1  # header only

    def test_save_json(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        report = self._make_report()
        path = tmp_path / "report.json"
        report.save_json(path)
        data = json.loads(path.read_text())
        assert data["report_name"] == "Test Report"

    def test_save_csv(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        report = self._make_report()
        path = tmp_path / "report.csv"
        report.save_csv(path)
        content = path.read_text()
        assert "voice" in content

    def test_save_trend_csv(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        report = self._make_report()
        path = tmp_path / "trends.csv"
        report.save_trend_csv(path)
        content = path.read_text()
        assert "avg_value" in content


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestReportReexport:
    def test_importable_from_analytics_package(self) -> None:
        from semantic_conversation_engine.analytics import (
            AnalyticsReport,
            AnalyticsSection,
            GroupedMetric,
            TrendSeries,
        )

        assert AnalyticsReport is not None
        assert AnalyticsSection is not None
        assert GroupedMetric is not None
        assert TrendSeries is not None
