import { describe, it, expect } from 'vitest';
import {
  buildActionSequence,
  buildSpotDict,
  validateSpot,
  usedCards,
  BOARD_SLOTS,
  type BuilderState,
} from '../spotBuilder';

function state(overrides: Partial<BuilderState>): BuilderState {
  return {
    street: 'flop',
    heroPosition: 'BTN',
    preflopAction: 'open',
    raiserPosition: 'BTN',
    heroHole: ['Ah', 'Kh'],
    board: ['7h', '2d', 'Jc'],
    ...overrides,
  };
}

const RAISE_STEMS = ['open', 'raise', '3bet', '4bet', '5bet'];
function isRaiseToken(tok: string): boolean {
  const verb = tok.split(':')[1] ?? '';
  return RAISE_STEMS.some((s) => verb.startsWith(s));
}

describe('buildActionSequence', () => {
  it('limp yields [] (the limp bucket, not a default fallthrough)', () => {
    expect(buildActionSequence(state({ preflopAction: 'limp' }))).toEqual([]);
  });

  it('open + raiser BTN yields exactly ["BTN:open"] (1 raise -> srp)', () => {
    expect(buildActionSequence(state({ preflopAction: 'open', raiserPosition: 'BTN' }))).toEqual([
      'BTN:open',
    ]);
  });

  it('3bet yields exactly 2 raise-stem tokens with last POS === raiserPosition', () => {
    const seq = buildActionSequence(state({ preflopAction: '3bet', raiserPosition: 'CO' }));
    expect(seq.filter(isRaiseToken)).toHaveLength(2);
    expect(seq[seq.length - 1].split(':')[0]).toBe('CO');
  });

  it('4bet yields exactly 3 raise-stem tokens with last POS === raiserPosition', () => {
    const seq = buildActionSequence(state({ preflopAction: '4bet', raiserPosition: 'BB' }));
    expect(seq.filter(isRaiseToken)).toHaveLength(3);
    expect(seq[seq.length - 1].split(':')[0]).toBe('BB');
  });

  it('intermediate placeholder seats are neither hero nor raiser', () => {
    const seq = buildActionSequence(
      state({ preflopAction: '4bet', heroPosition: 'BTN', raiserPosition: 'CO' }),
    );
    for (const tok of seq.slice(0, -1)) {
      const pos = tok.split(':')[0];
      expect(pos).not.toBe('BTN');
      expect(pos).not.toBe('CO');
    }
  });

  it('returns [] when raiserPosition empty (no raiser chosen yet)', () => {
    expect(buildActionSequence(state({ preflopAction: 'open', raiserPosition: '' }))).toEqual([]);
  });
});

describe('buildSpotDict', () => {
  it('carries the derived action_sequence + structured fields', () => {
    const dict = buildSpotDict(state({ preflopAction: 'open', raiserPosition: 'BTN' }));
    expect(dict).toMatchObject({
      street: 'flop',
      hero_position: 'BTN',
      hero_hole: ['Ah', 'Kh'],
      board: ['7h', '2d', 'Jc'],
      action_sequence: ['BTN:open'],
    });
  });
});

describe('validateSpot', () => {
  it('flags flop with 2 board cards (board-count mismatch)', () => {
    const v = validateSpot(state({ board: ['7h', '2d'] }));
    expect(v.ok).toBe(false);
    expect(v.errors.some((e) => e.includes('board must have 3'))).toBe(true);
  });

  it('flags a duplicate card across hole + board', () => {
    const v = validateSpot(state({ heroHole: ['Ah', 'Kh'], board: ['Ah', '2d', 'Jc'] }));
    expect(v.ok).toBe(false);
    expect(v.errors.some((e) => e.includes('duplicate'))).toBe(true);
  });

  it('flags hero hole length != 2', () => {
    const v = validateSpot(state({ heroHole: ['Ah'] }));
    expect(v.ok).toBe(false);
    expect(v.errors.some((e) => e.includes('hero hole'))).toBe(true);
  });

  it('flags an illegal card string', () => {
    const v = validateSpot(state({ heroHole: ['Zx', 'Kh'] }));
    expect(v.ok).toBe(false);
    expect(v.errors.some((e) => e.includes('illegal card'))).toBe(true);
  });

  it('returns ok:true for a clean flop spot', () => {
    expect(validateSpot(state({})).ok).toBe(true);
  });

  it('returns ok:true for a clean preflop spot (0 board cards)', () => {
    const v = validateSpot(state({ street: 'preflop', board: [] }));
    expect(v.ok).toBe(true);
  });
});

describe('usedCards + BOARD_SLOTS', () => {
  it('usedCards returns the union of hole + board', () => {
    const set = usedCards(state({ heroHole: ['Ah', 'Kh'], board: ['7h', '2d', 'Jc'] }));
    expect(set).toEqual(new Set(['Ah', 'Kh', '7h', '2d', 'Jc']));
  });

  it('BOARD_SLOTS maps street -> count', () => {
    expect(BOARD_SLOTS).toEqual({ preflop: 0, flop: 3, turn: 4, river: 5 });
  });
});
