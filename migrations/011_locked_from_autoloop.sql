-- migrations/011_locked_from_autoloop.sql
-- Phase 6 — D-NEW-26: lock cluster from autoloop re-patching.
--
-- WHY: Edit Node drawer (D-NEW-26 in 06-CONTEXT.md addendum #2) exposes a
-- "[ □ lock this cluster from autoloop ]" toggle. When set, the autoloop driver
-- (src/autoloop/driver.py) must skip this cluster during candidate selection so a
-- manually-tuned action_dist is not silently overwritten on the next session loop.
-- Stored on strategy_nodes (per-row, not per-cluster-key) because PatchEngine writes
-- a new node per patch — the lock applies to the *currently active* node and carries
-- forward via the next-write workflow (manual edits preserve the flag explicitly).
--
-- IDEMPOTENCY: ADD COLUMN IF NOT EXISTS for the column; the partial index uses
-- CREATE INDEX IF NOT EXISTS. The runner.migrate checksum tracking in
-- schema_migrations prevents the full file from being re-applied on subsequent
-- runs, so the DDL only executes once; together these guards make the migration
-- safe to run multiple times without error.
--
-- DEFAULT FALSE: pre-migration rows (Phase 1-5) are never locked — autoloop sees
-- them with locked_from_autoloop=FALSE, which is the correct backfill semantics
-- (no retroactive lock). NOT NULL guarantees the autoloop driver never has to
-- handle NULL when filtering candidates.
--
-- REFERENCE: .planning/phases/06-study-tool/06-CONTEXT.md §Addendum #2 D-NEW-26.

ALTER TABLE strategy_nodes
    ADD COLUMN IF NOT EXISTS locked_from_autoloop BOOLEAN NOT NULL DEFAULT FALSE;

-- Partial index supports autoloop driver's "WHERE locked_from_autoloop = TRUE"
-- skip-list lookup. Partial (WHERE TRUE) keeps the index tiny — locked rows are
-- expected to be a small minority of strategy_nodes.
CREATE INDEX IF NOT EXISTS idx_strategy_nodes_locked
    ON strategy_nodes (cluster_key)
    WHERE locked_from_autoloop = TRUE;
