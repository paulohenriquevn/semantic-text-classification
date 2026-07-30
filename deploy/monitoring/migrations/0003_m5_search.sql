-- M5 hybrid search & QA (blueprint D2/D3). Idempotent — safe to re-apply.
-- Adds the `labels` table: QA audit/label actions destined for retraining. Deliberately a plain
-- table (NOT a hypertable) so labels survive the 30-day raw-data purge — retraining is long-term
-- business memory (ADR-005 hot/purge split). Search indexes shipped in 0002 (GIN + HNSW); this
-- migration only adds the label sink.

CREATE TABLE IF NOT EXISTS labels (
    label_id        text PRIMARY KEY,
    turn_id         text NOT NULL,
    conversation_id text NOT NULL,
    label           text NOT NULL,
    labeled_by      text NOT NULL DEFAULT 'qa',
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Read paths: by anchor turn (open the labeled window) and by recency (retraining export).
CREATE INDEX IF NOT EXISTS labels_turn_idx ON labels (turn_id);
CREATE INDEX IF NOT EXISTS labels_created_idx ON labels (created_at DESC);
