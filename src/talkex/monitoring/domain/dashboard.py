"""Dashboard KPI value objects for M6 (blueprint D4/D5).

`KpiQuery` is a bounded manager-dashboard request (time range + optional whitelisted queue/rule);
`KpiBucket` is one pre-bucketed rollup row read from the `alerts_kpi_5min` continuous aggregate.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class KpiQuery(BaseModel):
    """A bounded KPI dashboard request over the retention window (dimensions are fixed)."""

    model_config = ConfigDict(frozen=True)

    from_time: datetime
    to_time: datetime
    queue: str | None = None
    rule_name: str | None = None


class KpiBucket(BaseModel):
    """One 5-minute KPI rollup: alert count per (rule, queue, sentiment) bucket."""

    model_config = ConfigDict(frozen=True)

    bucket: datetime
    rule_name: str
    queue: str
    sentiment: str | None
    alert_count: int
