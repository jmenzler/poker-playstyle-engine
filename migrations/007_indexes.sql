-- migrations/007_indexes.sql
-- Cross-table non-PK indexes (kept out of individual table migrations for clarity).

CREATE INDEX IF NOT EXISTS idx_observations_cluster_key
    ON observations (cluster_key);

CREATE INDEX IF NOT EXISTS idx_observations_session
    ON observations (session_id);

CREATE INDEX IF NOT EXISTS idx_observations_session_ts
    ON observations (session_id, ts DESC);

-- Metrics: support dashboard queries -- fast scan per session by metric type
CREATE INDEX IF NOT EXISTS idx_metrics_session_metric
    ON metrics (session_id, metric_name);

CREATE INDEX IF NOT EXISTS idx_metrics_cluster_metric
    ON metrics (cluster_key, metric_name) WHERE cluster_key IS NOT NULL;

-- METR-04 support: active strategy_nodes count grouped by source < 100ms on 50k nodes
CREATE INDEX IF NOT EXISTS idx_strategy_nodes_source_active
    ON strategy_nodes (source) WHERE active = TRUE;
