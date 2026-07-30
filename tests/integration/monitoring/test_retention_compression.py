"""Integration tests for M1 storage lifecycle: retention, compression, indexes, CA (T0.1).

Requires the dev Timescale with migration 0002 applied
(`psql -f deploy/monitoring/migrations/0002_m1_retention_indexes.sql`).
"""

import psycopg
import pytest

pytestmark = pytest.mark.integration


class TestPoliciesAndIndexes:
    async def test_retention_policies_registered(self, conn: psycopg.AsyncConnection) -> None:
        cur = await conn.execute(
            "SELECT hypertable_name FROM timescaledb_information.jobs "
            "WHERE proc_name = 'policy_retention' AND hypertable_name IN ('turns', 'alerts')"
        )
        names = {r[0] for r in await cur.fetchall()}
        assert names == {"turns", "alerts"}

    async def test_compression_policies_registered(self, conn: psycopg.AsyncConnection) -> None:
        cur = await conn.execute(
            "SELECT hypertable_name FROM timescaledb_information.jobs "
            "WHERE proc_name = 'policy_compression' AND hypertable_name IN ('turns', 'alerts')"
        )
        names = {r[0] for r in await cur.fetchall()}
        assert names == {"turns", "alerts"}

    async def test_hnsw_gin_tsvector_indexes_exist(self, conn: psycopg.AsyncConnection) -> None:
        cur = await conn.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'turns'")
        idx = {r[0] for r in await cur.fetchall()}
        assert {"turns_embedding_hnsw", "turns_search_vector_gin", "turns_raw_text_trgm"} <= idx

    async def test_continuous_aggregate_exists(self, conn: psycopg.AsyncConnection) -> None:
        cur = await conn.execute(
            "SELECT view_name FROM timescaledb_information.continuous_aggregates WHERE view_name = 'turns_per_min'"
        )
        assert await cur.fetchone() is not None


class TestRetention:
    async def test_retention_drops_old_chunks(self, conn: psycopg.AsyncConnection) -> None:
        # Insert an old turn (40 days ago) and a fresh one, then drop chunks older than 30 days
        # (the manual equivalent of what the retention policy runs on schedule).
        await conn.execute(
            "INSERT INTO turns (turn_id, conversation_id, speaker, raw_text, start_offset, end_offset, created_at) "
            "VALUES ('turn_old', 'conv_ret', 'customer', 'antigo', 0, 6, now() - INTERVAL '40 days')"
        )
        await conn.execute(
            "INSERT INTO turns (turn_id, conversation_id, speaker, raw_text, start_offset, end_offset, created_at) "
            "VALUES ('turn_new', 'conv_ret', 'customer', 'recente', 0, 7, now())"
        )
        await conn.commit()

        await conn.execute("SELECT drop_chunks('turns', older_than => INTERVAL '30 days')")
        await conn.commit()

        cur = await conn.execute("SELECT turn_id FROM turns WHERE conversation_id = 'conv_ret' ORDER BY turn_id")
        remaining = {r[0] for r in await cur.fetchall()}
        assert "turn_old" not in remaining  # old chunk dropped
        assert "turn_new" in remaining  # fresh data retained
