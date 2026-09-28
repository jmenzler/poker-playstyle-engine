-- migrations/023_corpus_versions.sql
-- Corpus version registry: maps a discrete version handle to an epoch cutoff_ts.
-- The engine pins to a version via added_at <= cutoff_ts; v0 (cutoff_ts=0) is the
-- genesis pin that matches every row.
CREATE TABLE IF NOT EXISTS corpus_versions (
    version    BIGSERIAL PRIMARY KEY,
    cutoff_ts  BIGINT NOT NULL,
    source     TEXT NOT NULL CHECK (source IN ('manual', 'autoloop', 'solver')),
    label      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO corpus_versions (cutoff_ts, source, label)
VALUES (0, 'manual', 'genesis');
