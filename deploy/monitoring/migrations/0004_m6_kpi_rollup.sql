-- M6 aggregated dashboards (blueprint D1-D3 / ADR-005). Idempotent — safe to re-apply.
-- Adds queue + sentiment dimensions to alerts, a 5-minute KPI continuous aggregate over alerts
-- (bucket × rule × queue × sentiment), and an incremental refresh policy. Deliberately NO retention
-- policy on the CA: the rollup is long-term business memory that outlives the 30-day raw purge.

-- queue + sentiment dimensions (guarded — alerts carries a columnstore/compression policy, so a plain
-- ADD COLUMN IF NOT EXISTS can conflict on re-apply; add only when absent).
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'alerts' AND column_name = 'queue') THEN
        ALTER TABLE alerts ADD COLUMN queue text NOT NULL DEFAULT 'default';
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'alerts' AND column_name = 'sentiment') THEN
        ALTER TABLE alerts ADD COLUMN sentiment text;
    END IF;
END $$;

-- KPI rollup: incremental 5-minute buckets per (rule, queue, sentiment). Extends the shipped
-- turns_per_min continuous-aggregate pattern (0002) — count materialized once, read cheap.
CREATE MATERIALIZED VIEW IF NOT EXISTS alerts_kpi_5min
    WITH (timescaledb.continuous) AS
    SELECT time_bucket('5 minutes', created_at) AS bucket,
           rule_name,
           queue,
           sentiment,
           count(*) AS alert_count
    FROM alerts
    GROUP BY bucket, rule_name, queue, sentiment
    WITH NO DATA;

-- Incremental refresh, closed-buckets-only: end_offset excludes the still-forming bucket (the
-- Timescale realization of chatwoot's "today is skipped" rollup). Honest freshness SLO: <= ~5 min.
SELECT add_continuous_aggregate_policy('alerts_kpi_5min',
    start_offset => NULL,
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE);

-- NO retention policy on alerts_kpi_5min ON PURPOSE (ADR-005 hot/purge split): raw alerts purge at
-- 30 days (policy in 0002) while these KPI buckets persist as long-term business memory.
