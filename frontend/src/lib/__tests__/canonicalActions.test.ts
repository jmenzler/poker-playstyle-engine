import { describe, it, expect } from 'vitest';
import { CANONICAL_ACTIONS } from '../canonicalActions';

// Mirror of src/decision_engine/blending.py CANONICAL_ACTIONS — if this drifts,
// the EditNodeDrawer patch vocab no longer matches what the backend accepts.
const BACKEND = [
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
];

describe('CANONICAL_ACTIONS', () => {
  it('matches the backend vocab labels and order exactly', () => {
    expect([...CANONICAL_ACTIONS]).toEqual(BACKEND);
  });

  it('uses backend labels, not the old drifted ones', () => {
    expect(CANONICAL_ACTIONS).toContain('bet_overbet');
    expect(CANONICAL_ACTIONS).not.toContain('overbet');
    for (const a of ['open_2_2bb', 'open_3bb', '3bet_3x', '3bet_4x', '4bet_2_5x']) {
      expect(CANONICAL_ACTIONS).toContain(a);
    }
  });
});
