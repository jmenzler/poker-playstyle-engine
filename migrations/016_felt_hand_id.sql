-- migrations/016_felt_hand_id.sql
-- Phase 10 Plan 02: per-decision felt snapshot + decision-point IDs (D-02, D-05).
--
-- Adds three columns to the observations hypertable:
--   felt_snapshot JSONB (nullable, no default)
--       The 11 GameState fields verbatim at the moment of the sim decision:
--       street, hero_position, hero_hole_cards, board_cards, pot_size_bb,
--       effective_stack_bb, hero_facing_bet_bb, hero_bet_size_bb,
--       action_sequence, opponents_remaining, prior_street_aggressor.
--       Stored once at writer.record() time (D-04); replay = cheap JSONB read.
--       NULL for HM-ingest rows (hm path does not call the writer with felt).
--
--   hand_id TEXT (nullable, no default)
--       Minted by the harness as {session_id}_h{hand_idx}. NULL for HM rows.
--
--   decision_id TEXT (nullable, no default)
--       Minted by the harness as {hand_id}_dp{decisions_this_hand}. NULL for
--       HM rows. Plain index only — hypertable PK is (obs_id, ts); the
--       partition column ts is required in any deduplication constraint.
--       Plain index suffices; session_id UUID makes collision astronomically
--       impossible (D-05 / RESEARCH Open Question #2).
--
-- Zero-downtime pattern: ADD COLUMN IF NOT EXISTS on a live hypertable applies
-- to all existing chunks without rewriting data. Nullable columns with no
-- DEFAULT cause no row backfill cost (same pattern as migrations/008_flagged_sparse.sql).

ALTER TABLE observations ADD COLUMN IF NOT EXISTS felt_snapshot JSONB;

ALTER TABLE observations ADD COLUMN IF NOT EXISTS hand_id TEXT;

ALTER TABLE observations ADD COLUMN IF NOT EXISTS decision_id TEXT;

CREATE INDEX IF NOT EXISTS idx_observations_hand_id ON observations (hand_id);
