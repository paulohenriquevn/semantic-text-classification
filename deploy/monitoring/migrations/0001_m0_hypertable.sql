-- M0 walking-skeleton schema (ADR-005 / blueprint D5).
-- Idempotent: safe to re-apply. Creates the timescaledb extension and two hypertables:
-- `turns` (per-Turn streaming inserts) and `alerts` (evidence-backed rule matches).

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS turns (
    turn_id         text        NOT NULL,
    conversation_id text        NOT NULL,
    speaker         text        NOT NULL,
    raw_text        text        NOT NULL,
    normalized_text text,
    start_offset    integer     NOT NULL,
    end_offset      integer     NOT NULL,
    metadata        jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (turn_id, created_at)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id        text        NOT NULL,
    conversation_id text        NOT NULL,
    window_id       text        NOT NULL,
    rule_name       text        NOT NULL,
    evidence        jsonb       NOT NULL DEFAULT '[]'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (alert_id, created_at)
);

-- Convert to hypertables (time dimension = created_at). if_not_exists keeps this idempotent.
SELECT create_hypertable('turns',  by_range('created_at'), if_not_exists => TRUE);
SELECT create_hypertable('alerts', by_range('created_at'), if_not_exists => TRUE);

-- Supervisor live query: recent turns/alerts per conversation (blueprint Q2 index precedent).
CREATE INDEX IF NOT EXISTS turns_conv_time_idx  ON turns  (conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS alerts_conv_time_idx ON alerts (conversation_id, created_at DESC);
