-- migrations/002_strategy_nodes.sql
-- StrategyNodes: one row per cluster_key x source x generation.
-- Patch-only write path enforced at pre-commit (PITFALLS.md §9); direct INSERT/UPDATE
-- outside src/patch_engine.py is rejected by .pre-commit-config.yaml.

CREATE TABLE IF NOT EXISTS strategy_nodes (
    node_id         UUID PRIMARY KEY,
    cluster_key     TEXT NOT NULL,
    embedding       REAL[] NOT NULL,
    action_dist     JSONB NOT NULL,
    gto_score       REAL NOT NULL,
    confidence      REAL NOT NULL,
    source          TEXT NOT NULL CHECK (source IN ('hh', 'sim', 'solver', 'manual', 'seed')),
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_by   UUID REFERENCES strategy_nodes(node_id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- STOR-03: index supports decision engine's filtered lookup by cluster + active=true
CREATE INDEX IF NOT EXISTS idx_strategy_nodes_cluster_active
    ON strategy_nodes (cluster_key, active);
