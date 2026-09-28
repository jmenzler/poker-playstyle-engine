-- migrations/020_hands_index.sql
-- Hand-level index backing the Hands-tab all-source browser.
--
-- WHY: the Hands tab must list every hand (real Hold'em Manager + SIM) with
-- source/solved filters. observations is per-decision-point and sim-source only
-- (D-07-11e keeps it empty on the HM path), so it cannot back a hand-level list.
-- This table is the hand-level entity: one row per hand, populated by
-- tools/build_hands_index.py from the decision corpus (HM) + observations (SIM).
-- It does NOT replace observations and does not violate D-07-11e.
--
-- "solved" is NOT a column — it is derived at query time via an EXISTS on
-- solver_cache (decision_id LIKE hand_id || '_dp%'), so it can never go stale.
--
-- source: 'hh' = real Hold'em Manager hand, 'sim' = simulated hand.
-- played_ts: HM = parsed HM3 handtimestamp; SIM = session ts. NULL when unknown
--            (sorts last via ORDER BY played_ts DESC NULLS LAST).

CREATE TABLE IF NOT EXISTS hands_index (
    hand_id         TEXT        PRIMARY KEY,
    source          TEXT        NOT NULL,
    hero_position   TEXT,
    stake           TEXT,
    n_decisions     INT         NOT NULL DEFAULT 0,
    street_reached  TEXT,
    played_ts       TIMESTAMPTZ,
    created_ts      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_hands_index_source
    ON hands_index (source);

CREATE INDEX IF NOT EXISTS idx_hands_index_played_ts
    ON hands_index (played_ts DESC);
