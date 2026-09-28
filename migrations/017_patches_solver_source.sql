-- migrations/017_patches_solver_source.sql
-- Phase 15 — extend patches.source CHECK to include 'solver' and 'solver_verify'.
--
-- WHY: solver-distillation (Phase 15 queue driver) creates strategy_nodes with
-- source='solver'. PatchEngine.apply writes the same source value into BOTH
-- strategy_nodes AND the patches audit row. migration 013 widened
-- strategy_nodes_source_check to include 'solver'/'solver_verify' but the
-- matching patches_source_check was never widened, so a solver node's audit
-- INSERT fails patches_source_check. The two constraints must agree.
--
-- TIGHT WIDENING: closed set widened by exactly the two values already valid
-- for strategy_nodes. No wildcard / free-text source.
--
-- IDEMPOTENCY: DROP-IF-EXISTS + ADD inside a DO block; runner checksum tracking
-- prevents re-apply. Self-healing if the constraint was manually dropped.
--
-- REFERENCE: migrations/013_solver_source.sql (same pattern, strategy_nodes).

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.check_constraints
        WHERE constraint_name = 'patches_source_check'
    ) THEN
        ALTER TABLE patches DROP CONSTRAINT patches_source_check;
    END IF;
    ALTER TABLE patches ADD CONSTRAINT patches_source_check
        CHECK (source IN ('autoloop', 'manual', 'solver', 'solver_verify'));
END $$;
