"""Unit tests for MonitoringConfig (T0.2)."""

import pytest
from pydantic import ValidationError

from talkex.monitoring.config import MonitoringConfig


class TestMonitoringConfig:
    def test_defaults_are_sensible(self) -> None:
        cfg = MonitoringConfig()
        assert cfg.queue_maxsize > 0
        assert isinstance(cfg.notify_channel, str) and cfg.notify_channel
        assert cfg.dsn.startswith("postgresql://")

    def test_is_frozen(self) -> None:
        cfg = MonitoringConfig()
        with pytest.raises(ValidationError):
            cfg.queue_maxsize = 10  # type: ignore[misc]

    def test_rejects_non_positive_queue_maxsize(self) -> None:
        with pytest.raises(ValidationError):
            MonitoringConfig(queue_maxsize=0)
