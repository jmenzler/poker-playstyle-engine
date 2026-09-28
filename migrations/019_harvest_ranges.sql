-- migrations/019_harvest_ranges.sql
-- Dedicated store for CFR-narrowed per-seat ranges produced by solve_harvest.
--
-- WHY: The harvest solve emits surviving combos (w>1e-4) per decision-point per
-- seat. Storing them here keeps repeat-view latency instant and allows a future
-- batch re-solve to backfill the entire corpus. PK (decision_id, seat) supports
-- per-DP upsert dedup without a timestamp in the key — this is a low-cardinality
-- lookup table, not a time-series, so a plain table is correct (no hypertable).
--
-- seat: 0=OOP, 1=IP (matches postflop-solver player indexing).
-- payload: per-seat slice of the harvest output — combos/weights/equity/
--          strategy/ev_detail/actions/hero_action — stored as JSONB.
-- idx_harvest_ranges_hand_id: supports the whole-hand load in one query
--                             (load_harvest_ranges_by_hand).

CREATE TABLE IF NOT EXISTS harvest_ranges (
    hand_id      TEXT        NOT NULL,
    decision_id  TEXT        NOT NULL,
    seat         INT         NOT NULL,
    payload      JSONB       NOT NULL,
    solved_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (decision_id, seat)
);

CREATE INDEX IF NOT EXISTS idx_harvest_ranges_hand_id
    ON harvest_ranges (hand_id);
