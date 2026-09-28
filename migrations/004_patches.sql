-- migrations/004_patches.sql
-- Patches: append-only audit trail for every strategy_nodes write
-- (PTCH-02, PTCH-03). Captures pre/post ev_loss + status.

CREATE TABLE IF NOT EXISTS patches (
    patch_id        UUID PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    source          TEXT NOT NULL CHECK (source IN ('autoloop', 'manual')),
    cluster_key     TEXT NOT NULL,
    prev_node_id    UUID REFERENCES strategy_nodes(node_id),  -- nullable for first-ever patch on a cluster
    new_node_id     UUID NOT NULL REFERENCES strategy_nodes(node_id),
    pre_ev_loss     REAL,
    post_ev_loss    REAL,
    status          TEXT NOT NULL DEFAULT 'applied' CHECK (status IN ('applied', 'rolled_back')),
    validation      JSONB,  -- VALN-02: A/B validation result (seed, n_hands, confidence_interval)
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_patches_cluster_ts
    ON patches (cluster_key, ts DESC);

CREATE INDEX IF NOT EXISTS idx_patches_status
    ON patches (status) WHERE status = 'applied';
