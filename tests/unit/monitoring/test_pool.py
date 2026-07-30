"""Unit tests for pool config + construction (T1.1, no DB)."""

import pytest

from talkex.monitoring.config import MonitoringConfig
from talkex.monitoring.infrastructure.pool import MonitoringPool


class TestPoolConfig:
    def test_pool_config_defaults(self) -> None:
        cfg = MonitoringConfig()
        assert cfg.pool_max_size > 0
        assert cfg.pool_min_size >= 0
        assert cfg.retention_days == 30

    def test_pool_rejects_non_positive_max_size(self) -> None:
        with pytest.raises(ValueError):
            MonitoringPool("postgresql://x", max_size=0)

    def test_pool_reports_max_size(self) -> None:
        pool = MonitoringPool("postgresql://x", max_size=7)
        assert pool.max_size == 7
