"""Monitoring configuration (frozen/strict, per ADR-002)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MonitoringConfig(BaseModel):
    """Configuration for the M0 monitoring slice.

    Attributes:
        dsn: TimescaleDB connection string (default targets the dev compose on :5433).
        queue_maxsize: Bounded ingest channel size — the backpressure lever (blueprint D1).
        notify_channel: Postgres LISTEN/NOTIFY channel name for the live push (blueprint D3).
    """

    model_config = ConfigDict(frozen=True, strict=True)

    dsn: str = "postgresql://talkex:talkex@localhost:5433/talkex_monitoring"
    queue_maxsize: int = Field(default=256, gt=0)
    notify_channel: str = "talkex_alerts"
    pool_min_size: int = Field(default=2, ge=0)
    pool_max_size: int = Field(default=10, gt=0)
    retention_days: int = Field(default=30, gt=0)
