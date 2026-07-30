"""Integration tests for M6 KPI continuous aggregate against a real TimescaleDB (T0.1 / T3.1).

Proves the CA exists, refreshes incrementally (a new alert bumps only its bucket), and — crucially —
that the rollup survives after the raw alert chunk is dropped (rollup-outlives-raw, ADR-005).
"""

import psycopg
import pytest
from psycopg.types.json import Jsonb

from talkex.monitoring.config import MonitoringConfig

pytestmark = pytest.mark.integration

DSN = MonitoringConfig().dsn


async def _insert_alert(conn: psycopg.AsyncConnection, alert_id: str, rule: str, queue: str) -> None:
    await conn.execute(
        "INSERT INTO alerts (alert_id, conversation_id, window_id, rule_name, evidence, queue, sentiment) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (alert_id, "conv_kpi", "win_kpi", rule, Jsonb([]), queue, "negative"),
    )
    await conn.commit()


async def _refresh(conn: psycopg.AsyncConnection) -> None:
    # refresh_continuous_aggregate() cannot run inside a transaction block → use a fresh autocommit conn.
    ac = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    try:
        await ac.execute("CALL refresh_continuous_aggregate('alerts_kpi_5min', NULL, NULL)")
    finally:
        await ac.close()


async def _bucket_counts(conn: psycopg.AsyncConnection) -> dict[tuple[str, str], int]:
    await conn.commit()  # reset the snapshot so the autocommit refresh's materialization is visible
    cur = await conn.execute("SELECT rule_name, queue, sum(alert_count) FROM alerts_kpi_5min GROUP BY rule_name, queue")
    return {(r[0], r[1]): int(r[2]) for r in await cur.fetchall()}


class TestKpiRollup:
    async def test_ca_exists_and_refreshes(self, conn: psycopg.AsyncConnection) -> None:
        cur = await conn.execute(
            "SELECT 1 FROM timescaledb_information.continuous_aggregates WHERE view_name = 'alerts_kpi_5min'"
        )
        assert await cur.fetchone() is not None  # CA registered

        await _insert_alert(conn, "a_k1", "cancellation", "retention")
        await _insert_alert(conn, "a_k2", "cancellation", "retention")
        await _insert_alert(conn, "a_k3", "escalation", "billing")
        await _refresh(conn)

        counts = await _bucket_counts(conn)
        assert counts[("cancellation", "retention")] == 2  # grouped per (rule, queue)
        assert counts[("escalation", "billing")] == 1

    async def test_no_retention_policy_on_ca(self, conn: psycopg.AsyncConnection) -> None:
        # The rollup must outlive the raw purge → the CA has NO retention job (only a refresh job).
        cur = await conn.execute(
            "SELECT count(*) FROM timescaledb_information.jobs "
            "WHERE hypertable_name = (SELECT materialization_hypertable_name "
            "FROM timescaledb_information.continuous_aggregates WHERE view_name = 'alerts_kpi_5min') "
            "AND proc_name = 'policy_retention'"
        )
        row = await cur.fetchone()
        assert row is not None and row[0] == 0  # no retention policy on the KPI rollup

    async def test_incremental_refresh_bumps_bucket(self, conn: psycopg.AsyncConnection) -> None:
        await _insert_alert(conn, "a_inc1", "cancellation", "retention")
        await _refresh(conn)
        before = await _bucket_counts(conn)

        await _insert_alert(conn, "a_inc2", "cancellation", "retention")
        await _refresh(conn)
        after = await _bucket_counts(conn)

        assert after[("cancellation", "retention")] == before[("cancellation", "retention")] + 1

    async def test_rollup_survives_raw_chunk_drop(self, conn: psycopg.AsyncConnection) -> None:
        # Insert into an OLD chunk (Timescale drop_chunks only drops chunks fully in the past), then
        # materialize, then drop the old raw chunk — the rollup bucket must remain.
        await conn.execute(
            "INSERT INTO alerts "
            "(alert_id, conversation_id, window_id, rule_name, evidence, queue, sentiment, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, now() - interval '40 days')",
            ("a_drop1", "conv_kpi", "win_kpi", "cancellation", Jsonb([]), "retention", "negative"),
        )
        await conn.commit()
        await _refresh(conn)
        assert await _bucket_counts(conn)  # materialized at the 40-days-ago bucket

        # Drop raw chunks older than 30 days — the 40-day-old raw row is purged.
        ac = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
        try:
            await ac.execute("SELECT drop_chunks('alerts', older_than => now() - interval '30 days')")
        finally:
            await ac.close()

        await conn.commit()
        cur = await conn.execute("SELECT count(*) FROM alerts WHERE alert_id = 'a_drop1'")
        assert (await cur.fetchone())[0] == 0  # raw purged

        counts = await _bucket_counts(conn)
        assert counts.get(("cancellation", "retention"), 0) >= 1  # rollup outlives the raw purge
