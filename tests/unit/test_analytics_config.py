"""Unit tests for analytics configuration — enums and frozen config.

Tests cover: enum values, AnalyticsConfig construction, defaults, frozen
immutability, strict mode, validation, and reexports.
"""

import pytest
from pydantic import ValidationError

from semantic_conversation_engine.analytics.config import (
    AggregationWindow,
    AnalyticsConfig,
    AnalyticsLevel,
    MetricType,
)

# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


class TestAnalyticsLevel:
    def test_values(self) -> None:
        assert AnalyticsLevel.TURN == "turn"
        assert AnalyticsLevel.CONTEXT_WINDOW == "context_window"
        assert AnalyticsLevel.CONVERSATION == "conversation"
        assert AnalyticsLevel.SYSTEM == "system"

    def test_count(self) -> None:
        assert len(AnalyticsLevel) == 4


class TestAggregationWindow:
    def test_values(self) -> None:
        assert AggregationWindow.HOURLY == "hourly"
        assert AggregationWindow.DAILY == "daily"
        assert AggregationWindow.WEEKLY == "weekly"
        assert AggregationWindow.MONTHLY == "monthly"

    def test_count(self) -> None:
        assert len(AggregationWindow) == 4


class TestMetricType:
    def test_values(self) -> None:
        assert MetricType.CLASSIFICATION == "classification"
        assert MetricType.RULE == "rule"
        assert MetricType.RETRIEVAL == "retrieval"
        assert MetricType.PIPELINE == "pipeline"

    def test_count(self) -> None:
        assert len(MetricType) == 4


# ---------------------------------------------------------------------------
# AnalyticsConfig — construction
# ---------------------------------------------------------------------------


class TestAnalyticsConfigConstruction:
    def test_defaults(self) -> None:
        config = AnalyticsConfig()
        assert config.default_level == AnalyticsLevel.CONVERSATION
        assert config.default_window == AggregationWindow.DAILY
        assert config.max_trend_points == 100
        assert config.include_metadata is True
        assert len(config.enabled_metric_types) == len(MetricType)

    def test_custom_values(self) -> None:
        config = AnalyticsConfig(
            default_level=AnalyticsLevel.TURN,
            default_window=AggregationWindow.HOURLY,
            max_trend_points=50,
            include_metadata=False,
            enabled_metric_types=[MetricType.CLASSIFICATION, MetricType.RULE],
        )
        assert config.default_level == AnalyticsLevel.TURN
        assert config.default_window == AggregationWindow.HOURLY
        assert config.max_trend_points == 50
        assert config.include_metadata is False
        assert len(config.enabled_metric_types) == 2


# ---------------------------------------------------------------------------
# AnalyticsConfig — validation
# ---------------------------------------------------------------------------


class TestAnalyticsConfigValidation:
    def test_max_trend_points_positive(self) -> None:
        with pytest.raises(ValidationError):
            AnalyticsConfig(max_trend_points=0)

    def test_max_trend_points_negative(self) -> None:
        with pytest.raises(ValidationError):
            AnalyticsConfig(max_trend_points=-1)


# ---------------------------------------------------------------------------
# AnalyticsConfig — strict mode
# ---------------------------------------------------------------------------


class TestAnalyticsConfigStrict:
    def test_rejects_string_for_int(self) -> None:
        with pytest.raises(ValidationError):
            AnalyticsConfig(max_trend_points="100")  # type: ignore[arg-type]

    def test_rejects_string_for_bool(self) -> None:
        with pytest.raises(ValidationError):
            AnalyticsConfig(include_metadata="yes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AnalyticsConfig — immutability
# ---------------------------------------------------------------------------


class TestAnalyticsConfigFrozen:
    def test_frozen(self) -> None:
        config = AnalyticsConfig()
        with pytest.raises(ValidationError):
            config.max_trend_points = 50  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestConfigReexport:
    def test_importable_from_analytics_package(self) -> None:
        from semantic_conversation_engine.analytics import (
            AggregationWindow,
            AnalyticsConfig,
            AnalyticsLevel,
            MetricType,
        )

        assert AnalyticsLevel is not None
        assert AggregationWindow is not None
        assert MetricType is not None
        assert AnalyticsConfig is not None
