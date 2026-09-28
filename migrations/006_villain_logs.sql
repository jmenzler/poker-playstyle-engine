-- migrations/006_villain_logs.sql
-- VillainLogs: passive log of opponent actions per decision point.
-- v1: passive only (no overlay training). v2: feeds villain pool model (OPP-01).

CREATE TABLE IF NOT EXISTS villain_logs (
    log_id          UUID NOT NULL,
    session_id      TEXT NOT NULL,
    cluster_key     TEXT,
    villain_id      TEXT NOT NULL,                  -- pool identifier; HM3/HM4 player_id
    position        TEXT,                            -- UTG/MP/CO/BTN/SB/BB
    action_taken    TEXT NOT NULL,
    bet_size_pct    REAL,                            -- normalized to pot
    ts              TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (log_id, ts)
);

-- Not yet a hypertable. If write volume in Phase 5+ justifies, convert via:
--   SELECT create_hypertable('villain_logs', 'ts', migrate_data => TRUE, if_not_exists => TRUE);
-- For v1, keep as regular table -- write rate from HH ingest is low.

CREATE INDEX IF NOT EXISTS idx_villain_logs_villain_ts
    ON villain_logs (villain_id, ts DESC);
