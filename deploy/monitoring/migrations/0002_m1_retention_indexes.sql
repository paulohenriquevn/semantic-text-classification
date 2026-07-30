-- M1 storage foundation (blueprint D1-D3 / ADR-005). Idempotent — safe to re-apply.
-- Adds: pg_trgm + pgvector extensions, a generated tsvector column + GIN (BM25-adjacent lexical),
-- a GIN trigram index on raw_text, an embedding column + HNSW index (populated in M2),
-- 30-day retention + compression policies, and a per-minute continuous aggregate.

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

-- BM25-adjacent lexical ranking: generated tsvector (PT-BR stemming) + GIN (blueprint D2).
-- Guarded via DO blocks: a plain ADD COLUMN IF NOT EXISTS still conflicts with columnstore
-- (compression) on re-apply, so we add the column only when it does not already exist.
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'turns' AND column_name = 'search_vector') THEN
        ALTER TABLE turns ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (to_tsvector('portuguese', raw_text)) STORED;
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS turns_search_vector_gin ON turns USING gin (search_vector);

-- Substring/lexical acceleration on raw text (chatwoot db/schema.rb:1171 precedent).
CREATE INDEX IF NOT EXISTS turns_raw_text_trgm ON turns USING gin (raw_text gin_trgm_ops);

-- Embedding column + HNSW index path (blueprint D1; populated by M2).
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'turns' AND column_name = 'embedding') THEN
        ALTER TABLE turns ADD COLUMN embedding vector(384);
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS turns_embedding_hnsw ON turns USING hnsw (embedding vector_cosine_ops);

-- 30-day retention (blueprint D3 / ADR-005) — purge = chunk drop, not row DELETE.
SELECT add_retention_policy('turns',  INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('alerts', INTERVAL '30 days', if_not_exists => TRUE);

-- Compression of chunks older than 3 days (blueprint D3).
ALTER TABLE turns  SET (timescaledb.compress, timescaledb.compress_segmentby = 'conversation_id');
ALTER TABLE alerts SET (timescaledb.compress, timescaledb.compress_segmentby = 'conversation_id');
SELECT add_compression_policy('turns',  INTERVAL '3 days', if_not_exists => TRUE);
SELECT add_compression_policy('alerts', INTERVAL '3 days', if_not_exists => TRUE);

-- Per-minute rollup for supervisor dashboards (blueprint D3 / continuous aggregate).
CREATE MATERIALIZED VIEW IF NOT EXISTS turns_per_min
    WITH (timescaledb.continuous) AS
    SELECT time_bucket('1 minute', created_at) AS bucket,
           conversation_id,
           count(*) AS turn_count
    FROM turns
    GROUP BY bucket, conversation_id
    WITH NO DATA;
