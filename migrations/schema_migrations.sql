-- migrations/schema_migrations.sql
-- Tracking table for the raw-SQL migration runner (Decision 4).
-- Idempotent: re-running this file is a no-op.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    checksum    TEXT NOT NULL
);
