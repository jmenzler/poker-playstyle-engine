-- migrations/018_solver_cache_decision_id.sql
-- Phase 15 Plan 07 — add decision_id column to solver_cache for per-DP dedup.
--
-- WHY: The solver queue now tracks solves per decision-point (per observation),
-- not per cluster_key. The per-DP dedup query needs sc.decision_id = o.decision_id
-- so placeholder spots and already-solved DPs are excluded without the infinite
-- re-fetch loop observed in the live canary run.
--
-- COLUMN: TEXT nullable — legacy rows have no decision_id; new rows written by
-- persist_solve carry the obs.decision_id from the priority SQL SELECT.
--
-- INDEX: idx_solver_cache_decision_id on (decision_id) supports the NOT EXISTS
-- subquery in _PRIORITY_SQL and _count_sparse.
--
-- IDEMPOTENCY: DO-block pattern per migrations/013_solver_source.sql +
-- migrations/017_patches_solver_source.sql — safe to re-run.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'solver_cache'
          AND column_name = 'decision_id'
    ) THEN
        ALTER TABLE solver_cache ADD COLUMN decision_id TEXT;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_solver_cache_decision_id
    ON solver_cache (decision_id);
