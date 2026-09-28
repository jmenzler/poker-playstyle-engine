-- migrations/013_solver_source.sql
-- Phase 6 — OQ-1 resolution: extend strategy_nodes.source CHECK to include 'solver_verify'.
--
-- WHY: Phase 6 ships Stage B of the two-stage strategy-leak detector (D-05 in
-- 06-CONTEXT.md). Stage B parses postflop-cli output and writes the resulting
-- action distribution into strategy_nodes for caching. Two semantic categories:
--   - source='solver'         — precomputed/offline solver labels (Phase 4 path)
--   - source='solver_verify'  — Stage B on-demand verification triggered from the
--                                Strategy Leaks panel. v1 may still use 'solver'
--                                for these writes (per OQ-3 resolution note in the
--                                plan); 'solver_verify' is added now so future
--                                distillation (v2) can distinguish without an
--                                additional schema change.
--
-- TIGHT WIDENING (T-06-01 mitigation): the CHECK is widened by exactly one
-- well-defined value. The constraint remains a closed set — wildcard / free-text
-- source would be a tampering surface.
--
-- IDEMPOTENCY: the DO $$ … END $$ block wraps a DROP-CONSTRAINT-IF-EXISTS +
-- ADD-CONSTRAINT pair. runner.migrate checksum tracking prevents the full file
-- from being re-applied on subsequent runs. The IF EXISTS guard inside the DO
-- block keeps the migration self-healing if someone manually dropped the
-- constraint between runs.
--
-- REFERENCE: migration 010_autoloop_source.sql (same pattern, adds 'autoloop').

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.check_constraints
        WHERE constraint_name = 'strategy_nodes_source_check'
    ) THEN
        ALTER TABLE strategy_nodes DROP CONSTRAINT strategy_nodes_source_check;
    END IF;
    ALTER TABLE strategy_nodes ADD CONSTRAINT strategy_nodes_source_check
        CHECK (source IN ('hh', 'sim', 'solver', 'manual', 'seed', 'autoloop', 'solver_verify'));
END $$;
