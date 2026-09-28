-- migrations/021_patches_decision_id.sql
-- DP-level patch unification: tag each patch with its source decision point so a
-- patch's Milvus row (keyed by decision_id) can be located for rollback/detail.
ALTER TABLE patches ADD COLUMN decision_id TEXT;
