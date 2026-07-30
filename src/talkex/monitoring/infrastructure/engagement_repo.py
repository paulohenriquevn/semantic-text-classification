"""Timescale engagement repository — the alert-engagement proxy (M8 D4).

Computes `acted_on_rate` = labels created in a window ÷ alerts raised in that window. A QA label (M5
`POST /label`) IS a supervisor acting on an alert, so the two existing durable counts give the north-star
proxy with no new instrumentation (DRY). The RATE is a production signal only under a real pilot.
"""

from __future__ import annotations

from datetime import datetime

from talkex.monitoring.domain.health import EngagementMetric
from talkex.monitoring.infrastructure.pool import MonitoringPool


class TimescaleEngagementRepository:
    """Reads the acted-on-alert engagement proxy over a window."""

    def __init__(self, pool: MonitoringPool) -> None:
        self._pool = pool

    async def engagement(self, from_time: datetime, to_time: datetime) -> EngagementMetric:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) FROM alerts WHERE created_at >= %s AND created_at < %s",
                (from_time, to_time),
            )
            alerts_row = await cur.fetchone()
            cur = await conn.execute(
                "SELECT count(*) FROM labels WHERE created_at >= %s AND created_at < %s",
                (from_time, to_time),
            )
            labels_row = await cur.fetchone()
        alerts = int(alerts_row[0]) if alerts_row else 0
        labels = int(labels_row[0]) if labels_row else 0
        rate = (labels / alerts) if alerts > 0 else 0.0
        return EngagementMetric(alerts=alerts, labels=labels, acted_on_rate=rate)
