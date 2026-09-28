-- migrations/008_flagged_sparse.sql
-- Phase 4 Plan 02: sparseness flagging + max neighbor distance capture.
--
-- Adds two columns to the observations hypertable:
--   flagged_sparse BOOL NOT NULL DEFAULT FALSE
--       Set TRUE by KNNDecisionEngine when the kNN neighborhood is sparse:
--       (max_distance > SPARSE_DIST_TAU) OR (n_neighbors < SPARSE_N_MIN).
--       Migration default FALSE means pre-migration Phase 3 rows are never
--       exposed by the uncertain-spots WHERE flagged_sparse = TRUE filter —
--       exactly the desired backfill behaviour (no retroactive flagging).
--
--   max_neighbor_distance REAL (nullable, no default)
--       Maximum COSINE distance among kNN neighbors retrieved at decide-time.
--       NULL for pre-migration rows (no captured distance) and for NoStrategy
--       fallback writes (zero neighbors retrieved — NULL is semantically correct
--       vs 0.0 or inf).  Stored at write-time to avoid per-CLI-row Milvus
--       round-trips in the uncertain-spots query path (Phase 4 Decision 4 /
--       CONTEXT.md §OQ-4 "store at write-time").
--
-- Secondary sort index (max_neighbor_distance DESC WHERE flagged_sparse = TRUE)
-- is intentionally omitted from v1.  The partial index below reduces the scan
-- to <5% of rows; an in-memory sort on that subset is fast enough without a
-- covering index.  Re-evaluate with EXPLAIN ANALYZE in Phase 4.x or Plan 04
-- verification if the sort dominates.

ALTER TABLE observations ADD COLUMN IF NOT EXISTS flagged_sparse BOOL NOT NULL DEFAULT FALSE;

ALTER TABLE observations ADD COLUMN IF NOT EXISTS max_neighbor_distance REAL;

CREATE INDEX IF NOT EXISTS idx_observations_flagged_sparse
    ON observations (flagged_sparse) WHERE flagged_sparse = TRUE;
