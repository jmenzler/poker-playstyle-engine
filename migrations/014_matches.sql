-- migrations/014_matches.sql
-- Phase 6 — D-NEW-30: Eval head-to-head matches against baseline opponents.
--
-- WHY: Eval tab (sidebar slot 7, added in 06-CONTEXT.md addendum #3) runs the
-- engine against fixed-policy baselines (random / always-call / always-raise /
-- tight-passive / LAG-profile / TAG-profile / prev-engine-v{N}) over N hands,
-- reports bb/100 with confidence interval, surfaces per-street + per-texture edge,
-- top-5 profitable/leaky cluster_keys. One row per match. Used for regression
-- detection ("did autoloop break the engine?") and sanity-floor checks
-- ("engine MUST beat random").
--
-- HYPERTABLE: chunked by started_at on 7-day intervals — same chunk size as
-- migrations/005_metrics.sql. Low cardinality of matches (<100/week per
-- D-NEW-30 note + capacity-acceptance T-06-05) means we will never need finer
-- chunking; 7d keeps chunk count low for a single-developer tool.
--
-- COMPOSITE PRIMARY KEY: TimescaleDB requires the partitioning column to be part
-- of any UNIQUE or PRIMARY KEY constraint on the hypertable. PK = (match_id,
-- started_at) satisfies this while preserving match_id uniqueness for foreign-key
-- references (e.g., a future per-hand-of-match details table can FK on
-- (match_id, started_at)).
--
-- STATUS CHECK accommodates BOTH lifecycle and verdict states:
--   lifecycle:  queued → running → done|failed|cancelled
--   verdict:    won|lost|inconclusive|regression  (terminal verdicts that replace
--               'done' once the bb/100 + CI are evaluated against the verdict
--               policy in src/eval/run_match.py)
--   Schema accommodates both; the application layer (Eval drawer / leaderboard
--   query) decides which set applies in which view.
--
-- engine_version sourced from pyproject.toml at match-start time (T-06-04
-- mitigation: NEVER accept from API request body).
--
-- IDEMPOTENCY: CREATE TABLE IF NOT EXISTS guards the DDL; create_hypertable
-- uses if_not_exists => TRUE (no error if already a hypertable). Indexes use
-- CREATE INDEX IF NOT EXISTS. runner.migrate checksum tracking prevents
-- re-application.
--
-- REFERENCE: .planning/phases/06-study-tool/06-CONTEXT.md §Addendum #3 D-NEW-30.

CREATE TABLE IF NOT EXISTS matches (
    match_id        UUID NOT NULL DEFAULT gen_random_uuid(),
    opponent        TEXT NOT NULL,
    hands           INTEGER NOT NULL,
    seed            BIGINT NOT NULL,
    bb_per_100      DOUBLE PRECISION,
    ci_low          DOUBLE PRECISION,
    ci_high         DOUBLE PRECISION,
    per_street      JSONB,
    per_texture     JSONB,
    top5_profitable JSONB,
    top5_leaky      JSONB,
    status          TEXT NOT NULL CHECK (status IN (
                        'queued', 'running', 'done', 'failed', 'cancelled',
                        'won', 'lost', 'inconclusive', 'regression'
                    )),
    engine_version  TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    PRIMARY KEY (match_id, started_at)
);

SELECT create_hypertable('matches', 'started_at',
                         chunk_time_interval => INTERVAL '7 days',
                         if_not_exists => TRUE);

-- Leaderboard query: per-opponent rollup ordered by recent first.
CREATE INDEX IF NOT EXISTS idx_matches_opponent_status
    ON matches (opponent, status, started_at DESC);

-- Status-only filter (e.g., "all running matches" for the queue panel).
CREATE INDEX IF NOT EXISTS idx_matches_status_started
    ON matches (status, started_at DESC);
