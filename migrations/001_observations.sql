-- migrations/001_observations.sql
-- Observations: every decision-point produces one row.
-- Hypertable with 1-day chunks to handle 30-50k hands/hr sim throughput (Pitfall 7).

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS observations (
    obs_id          UUID NOT NULL,
    cluster_key     TEXT NOT NULL,
    embedding       REAL[] NOT NULL,        -- 128-dim float vector
    action_taken    TEXT NOT NULL,
    ev_realized     NUMERIC(18, 6),
    source          TEXT NOT NULL CHECK (source IN ('hh', 'sim')),
    session_id      TEXT NOT NULL,
    solver_label    JSONB,                  -- populated by labeling pipeline (Phase 4)
    ts              TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (obs_id, ts)                -- hypertable PK must include partition column
);

SELECT create_hypertable(
    'observations', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
