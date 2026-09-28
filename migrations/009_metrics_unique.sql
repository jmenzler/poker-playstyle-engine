-- migrations/009_metrics_unique.sql
-- Phase 7 Plan 07-01: add composite UNIQUE constraint on metrics for ON CONFLICT idempotency.
--
-- WHY: Closes the schema half of WARN 1 / INTG-04 in `.planning/v1.0-MILESTONE-AUDIT.md`.
-- `src/metrics/sink.py::flush_session_metrics` is non-idempotent today — repeated calls
-- for the same session write duplicate rows. Plan 07-05 will switch the INSERT to
-- `ON CONFLICT (session_id, metric_name, cluster_key, ts) DO NOTHING`; that requires
-- this UNIQUE constraint to exist first (D-07-11c). `ts` MUST be part of the UNIQUE
-- because the `metrics` table is a TimescaleDB hypertable partitioned on `ts`, and
-- TimescaleDB requires every UNIQUE index/constraint to cover all partitioning
-- columns (per 07-RESEARCH.md Focus Area 1, Pitfall 1). Plan 07-05 pins
-- `ts = session_started_at` so re-flushes deterministically conflict instead of
-- producing duplicate rows with different timestamps.
--
-- WHY NULLS NOT DISTINCT: `cluster_key` is NULL for `novel_spot_rate`,
-- `latency_p50`, and `latency_p99` rows (see `src/metrics/sink.py:155-167`).
-- Postgres default treats NULL != NULL in UNIQUE constraints, so without
-- NULLS NOT DISTINCT multiple NULL-cluster_key rows for the same session would
-- all be accepted, defeating the idempotency goal (PG 15+ feature; per
-- 07-RESEARCH.md Focus Area 4).
--
-- WHY NO observations UNIQUE: per D-07-11a (supersedes D-07-3) — TimescaleDB
-- requires `ts` in every UNIQUE; `(obs_id, ts)` is already the PK on observations
-- and `src/sim/observation_writer.py` generates fresh `uuid4()` per row, so a
-- hypothetical `UNIQUE (obs_id)` constraint would never fire. BOOT-04 idempotency
-- is delivered by the Milvus PK on deterministic `decision_id` (Phase 2 lock —
-- `tools/extract_decisions.py:266`), NOT by a TSDB constraint.
--
-- IDEMPOTENCY: the DO $$ ... END $$ block wraps an existence check + ADD
-- CONSTRAINT, making re-apply a no-op even if the schema_migrations row was
-- removed manually. runner.migrate's SHA-256 checksum tracking prevents the
-- file itself from being re-applied on normal runs. PostgreSQL does not
-- support `ADD CONSTRAINT IF NOT EXISTS` for UNIQUE constraints directly, so
-- the DO block is the canonical idiom (same pattern as migration 013).
-- No `CONCURRENTLY` — runner.migrate uses single-file commit (07-RESEARCH.md
-- Focus Area 2).
--
-- REFERENCE:
--   - .planning/phases/07-close-gap-boot-cli-watermark/07-CONTEXT.md D-07-11a, D-07-11c
--   - .planning/phases/07-close-gap-boot-cli-watermark/07-RESEARCH.md Focus Area 1, 4
--   - .planning/v1.0-MILESTONE-AUDIT.md WARN 1 / INTG-04 (schema half)
--   - migration 010_autoloop_source.sql (header rationale template)
--   - migration 013_solver_source.sql (DO-block idempotency template)

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name = 'metrics'
          AND constraint_name = 'metrics_unique_per_session'
    ) THEN
        ALTER TABLE metrics
            ADD CONSTRAINT metrics_unique_per_session
            UNIQUE NULLS NOT DISTINCT (session_id, metric_name, cluster_key, ts);
    END IF;
END $$;
