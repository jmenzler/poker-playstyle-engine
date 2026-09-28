-- migrations/012_leak_suppressions.sql
-- Phase 6 — D-NEW-28 refinement #6: Strategy Leaks "[ accept as intentional ]" path.
--
-- WHY: Operator-driven suppression of a flagged strategy leak. When the operator
-- decides a flagged cluster is intentionally non-GTO (e.g., a population exploit
-- in their pool), they click [accept as intentional] in the Strategy Leaks panel.
-- That action writes a row here; the leak detector then filters out any
-- cluster_key with an active suppression row from future ranks.
--
-- APPEND-ONLY + REVERSIBLE: rows are never deleted. To re-surface a previously
-- suppressed cluster, UPDATE leak_suppressions SET active=FALSE WHERE
-- suppression_id=…; the index on (cluster_key, active) lets the leak detector
-- filter efficiently. Audit trail (ts + reason) preserved.
--
-- IDEMPOTENCY: CREATE TABLE IF NOT EXISTS guards the DDL; the index uses CREATE
-- INDEX IF NOT EXISTS. runner.migrate checksum tracking prevents re-application.
--
-- gen_random_uuid() requires pgcrypto OR (PG 13+) the built-in. TimescaleDB
-- images bundle pgcrypto by default; same pattern as migration 004 (patches.sql)
-- relies on the caller having already enabled pgcrypto or run with PG 13+.
--
-- REFERENCE: .planning/phases/06-study-tool/06-CONTEXT.md §Addendum #2 D-NEW-28.

CREATE TABLE IF NOT EXISTS leak_suppressions (
    suppression_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_key     TEXT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason          TEXT,
    active          BOOLEAN NOT NULL DEFAULT TRUE
);

-- Filter index: leak detector queries
--   SELECT 1 FROM leak_suppressions
--   WHERE cluster_key = $1 AND active = TRUE
-- per candidate cluster. Index makes this an index-only scan.
CREATE INDEX IF NOT EXISTS idx_leak_suppressions_active_cluster
    ON leak_suppressions (cluster_key, active);
