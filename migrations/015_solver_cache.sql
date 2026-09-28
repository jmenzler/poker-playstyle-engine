-- migrations/015_solver_cache.sql
-- Phase 9 — Solver cache: persisted solve results keyed by cluster_key.
--
-- WHY: Tier-1 (LOO) needs ground-truth action_dist per cluster; Tier-2
-- needs exploitability_pct. One solve produces both. This table is the
-- shared substrate for both eval tiers.
--
-- HYPERTABLE: chunked by solved_at on 7-day intervals — same chunk size as
-- migrations/014_matches.sql and migrations/005_metrics.sql. Low cardinality
-- of solver cache rows means we will never need finer chunking; 7d keeps
-- chunk count low for a single-developer tool.
--
-- COMPOSITE PRIMARY KEY: TimescaleDB requires the partitioning column to be
-- part of any UNIQUE or PRIMARY KEY constraint on the hypertable. PK =
-- (cluster_key, solved_at) satisfies this constraint. A solver_cache row is a
-- cached calculation, not an active policy node — do NOT add to strategy_nodes
-- (RESEARCH OQ-6).
--
-- IDEMPOTENCY: CREATE TABLE IF NOT EXISTS guards the DDL; create_hypertable
-- uses if_not_exists => TRUE (no error if already a hypertable). Index uses
-- CREATE INDEX IF NOT EXISTS. Migration runner idempotency tracking prevents
-- re-application (STOR-04).

CREATE TABLE IF NOT EXISTS solver_cache (
    cluster_key         TEXT NOT NULL,
    action_dist         JSONB NOT NULL,
    exploitability_pct  DOUBLE PRECISION NOT NULL,
    spot_features       JSONB,
    solver_version      TEXT,
    solved_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (cluster_key, solved_at)
);

SELECT create_hypertable('solver_cache', 'solved_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_solver_cache_cluster_key
    ON solver_cache (cluster_key, solved_at DESC);
