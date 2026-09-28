-- migrations/010_autoloop_source.sql
-- Phase 5 Plan 02: extend strategy_nodes.source CHECK constraint to include 'autoloop'.
--
-- WHY: Phase 5 PatchEngine (src/patch_engine.py) writes new strategy_nodes rows with
-- source='autoloop'. Migration 002 declared the CHECK constraint allowing only
-- ('hh', 'sim', 'solver', 'manual', 'seed'). Without this migration applied, every
-- PatchEngine.apply() call raises psycopg.errors.CheckViolation on the INSERT.
-- See: .planning/phases/05-auto-loop/05-RESEARCH.md §Common Pitfalls Pitfall 3.
--
-- IDEMPOTENCY: DROP CONSTRAINT IF EXISTS is a no-op when the constraint does not exist
-- (e.g., re-run scenario). The runner.migrate checksum tracking in schema_migrations
-- prevents the full file from being re-applied on subsequent runs, so the ADD CONSTRAINT
-- will only execute once. Together these guards make the migration safe to run multiple
-- times without error.
--
-- REFERENCE: .planning/phases/05-auto-loop/05-CONTEXT.md §Open Questions (gto_score for
-- autoloop-source nodes) and §Decisions Decision 2 (PatchEngine single write path).

ALTER TABLE strategy_nodes
    DROP CONSTRAINT IF EXISTS strategy_nodes_source_check;

ALTER TABLE strategy_nodes
    ADD CONSTRAINT strategy_nodes_source_check
    CHECK (source IN ('hh', 'sim', 'solver', 'manual', 'seed', 'autoloop'));
