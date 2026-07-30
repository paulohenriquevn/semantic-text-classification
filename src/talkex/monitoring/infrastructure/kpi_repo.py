"""Timescale KPI read repository — queries the alerts_kpi_5min continuous aggregate (M6, blueprint D5).

Implements `KpiReadPort` by reading pre-bucketed rows straight from the CA (no raw scan). Filters are a
bounded whitelist (time range + optional queue/rule); values are always bound (injection-safe, reusing
the M5 discipline).
"""

from __future__ import annotations

from talkex.monitoring.domain.dashboard import KpiBucket, KpiQuery
from talkex.monitoring.infrastructure.pool import MonitoringPool


class TimescaleKpiRepository:
    """Reads KPI rollups from the continuous aggregate."""

    def __init__(self, pool: MonitoringPool) -> None:
        self._pool = pool

    async def kpi_rollups(self, query: KpiQuery) -> list[KpiBucket]:
        clause = ""
        params: list[object] = [query.from_time, query.to_time]
        if query.queue is not None:
            clause += " AND queue = %s"
            params.append(query.queue)
        if query.rule_name is not None:
            clause += " AND rule_name = %s"
            params.append(query.rule_name)
        sql = (
            "SELECT bucket, rule_name, queue, sentiment, alert_count "
            "FROM alerts_kpi_5min "
            "WHERE bucket >= %s AND bucket < %s" + clause + " "
            "ORDER BY bucket, rule_name, queue"
        )
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
        return [KpiBucket(bucket=r[0], rule_name=r[1], queue=r[2], sentiment=r[3], alert_count=int(r[4])) for r in rows]
