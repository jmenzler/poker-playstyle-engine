-- migrations/005_metrics.sql
-- Metrics hypertable: per-session, per-cluster scalar measurements
-- (ev_loss, novel_spot_rate, decision-engine p50/p99 latency).

CREATE TABLE IF NOT EXISTS metrics (
    metric_id       UUID NOT NULL,
    session_id      TEXT NOT NULL,
    cluster_key     TEXT,
    metric_name     TEXT NOT NULL,
    value           NUMERIC(18, 6) NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (metric_id, ts)
);

SELECT create_hypertable(
    'metrics', 'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);
