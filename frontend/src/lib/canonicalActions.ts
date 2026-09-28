// Must stay in sync with src/decision_engine/blending.py CANONICAL_ACTIONS
// (cross-language vocab invariant): same labels, same order.
export const CANONICAL_ACTIONS = [
  'check',
  'fold',
  'call',
  'open_2_2bb',
  'open_3bb',
  '3bet_3x',
  '3bet_4x',
  '4bet_2_5x',
  'bet_25',
  'bet_33',
  'bet_50',
  'bet_75',
  'bet_100',
  'bet_150',
  'bet_overbet',
  'raise_min',
  'raise_2_5x',
  'raise_3x',
  'raise_pot',
  'allin',
] as const;

export type CanonicalAction = (typeof CANONICAL_ACTIONS)[number];
