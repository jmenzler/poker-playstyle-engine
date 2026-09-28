-- migrations/003_weight_overlay.sql
-- WeightOverlay: applies modifier weights on top of a strategy_node action_dist.
-- v1 supports manual + autoloop overlays; future v2 villain-pool overlay (deferred).

CREATE TABLE IF NOT EXISTS weight_overlay (
    overlay_id      UUID PRIMARY KEY,
    cluster_key     TEXT NOT NULL,
    overlay_weights JSONB NOT NULL,   -- map: action_id -> multiplier (sum normalized post-apply)
    source          TEXT NOT NULL CHECK (source IN ('autoloop', 'manual', 'seed')),
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_by   UUID REFERENCES weight_overlay(overlay_id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Lookup by cluster + active for runtime application:
CREATE INDEX IF NOT EXISTS idx_weight_overlay_cluster_active
    ON weight_overlay (cluster_key, active);
