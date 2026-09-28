import { describe, it, expect } from 'vitest';
import fixture from '../__fixtures__/synthetic-ranges.json';
import {
  buildGridModel,
  actionVerb,
  actionColor,
} from '../gridModel';
import type { GridCell } from '../gridModel';

const dp0 = (fixture as unknown as ReturnType<typeof Array>[0][])[0] as {
  hero_seat: number;
  ip: Parameters<typeof buildGridModel>[0];
  oop: Parameters<typeof buildGridModel>[0];
};
const heroSeat = dp0.ip;
const villainSeat = dp0.oop;

describe('actionVerb', () => {
  it('maps FOLD to fold', () => expect(actionVerb('FOLD')).toBe('fold'));
  it('maps CALL to call', () => expect(actionVerb('CALL')).toBe('call'));
  it('maps CHECK to call', () => expect(actionVerb('CHECK')).toBe('call'));
  it('maps RAISE 1678 to raise', () => expect(actionVerb('RAISE 1678')).toBe('raise'));
  it('maps ALLIN 4240 to allin', () => expect(actionVerb('ALLIN 4240')).toBe('allin'));
  it('maps BET to raise', () => expect(actionVerb('BET')).toBe('raise'));
});

describe('actionColor', () => {
  it('fold is blue', () => expect(actionColor('FOLD')).toBe('#2563eb'));
  it('call is green', () => expect(actionColor('CALL')).toBe('#10b981'));
  it('RAISE 1678 is red', () => expect(actionColor('RAISE 1678')).toBe('#ef4444'));
  it('ALLIN 4240 is dark red', () => expect(actionColor('ALLIN 4240')).toBe('#7f1d1d'));
});

describe('buildGridModel — action mode hero', () => {
  const model = buildGridModel(heroSeat, 'action', { isHero: true });
  const cells = model.cells;

  it('has 169 cells', () => expect(cells.length).toBe(169));

  it('class 22 is present and all-call', () => {
    const cell = cells.find((c) => c.label === '22')!;
    expect(cell.present).toBe(true);
    const callSeg = cell.segments.find((s) => s.verb === 'call');
    expect(callSeg).toBeDefined();
    const segSum = cell.segments.reduce((s, sg) => s + sg.freq, 0);
    expect(segSum).toBeCloseTo(1, 2);
  });

  it('22 call segment has green color', () => {
    const cell = cells.find((c) => c.label === '22')!;
    const callSeg = cell.segments.find((s) => s.verb === 'call')!;
    expect(callSeg.color).toBe('#10b981');
  });

  it('22 reach is in (0,1]', () => {
    const cell = cells.find((c) => c.label === '22')!;
    expect(cell.reach).toBeGreaterThan(0);
    expect(cell.reach).toBeLessThanOrEqual(1);
  });

  it('out-of-range class has present=false', () => {
    const absentClasses = cells.filter((c) => !c.present);
    expect(absentClasses.length).toBeGreaterThan(0);
    const absent = absentClasses[0];
    expect(absent.segments.length).toBe(0);
    expect(absent.reach).toBe(0);
  });

  it('mode is action', () => expect(model.mode).toBe('action'));
});

describe('buildGridModel — action mode legend', () => {
  const model = buildGridModel(heroSeat, 'action', { isHero: true });
  const legend = model.legend;

  it('legend length equals actions count', () => {
    expect(legend.length).toBe(heroSeat.actions!.length);
  });

  it('legend freqPct sum ≈ 100', () => {
    const sum = legend.reduce((s, l) => s + l.freqPct, 0);
    expect(sum).toBeCloseTo(100, 0);
  });

  it('CALL combos > 0', () => {
    const callItem = legend.find((l) => l.verb === 'call');
    expect(callItem).toBeDefined();
    expect(callItem!.combos).toBeGreaterThan(0);
  });

  it('legend colors match actionColor', () => {
    for (const item of legend) {
      expect(item.color).toBe(actionColor(item.action));
    }
  });
});

describe('buildGridModel — range mode', () => {
  const model = buildGridModel(heroSeat, 'range', { isHero: true });
  const present = model.cells.filter((c) => c.present);

  it('legend is empty for range mode', () => expect(model.legend.length).toBe(0));

  it('max present class intensity is 1', () => {
    const max = Math.max(...present.map((c) => c.intensity));
    expect(max).toBeCloseTo(1, 5);
  });

  it('all present cell intensities are in (0,1]', () => {
    for (const c of present) {
      expect(c.intensity).toBeGreaterThan(0);
      expect(c.intensity).toBeLessThanOrEqual(1 + 1e-9);
    }
  });
});

describe('buildGridModel — equity mode', () => {
  const model = buildGridModel(heroSeat, 'equity', { isHero: true });
  const present = model.cells.filter((c) => c.present);

  it('all intensities in [0,1]', () => {
    for (const c of present) {
      expect(c.intensity).toBeGreaterThanOrEqual(0);
      expect(c.intensity).toBeLessThanOrEqual(1 + 1e-9);
    }
  });

  it('at least one cell has intensity > 0', () => {
    expect(present.some((c) => c.intensity > 0)).toBe(true);
  });

  it('a high-equity class has higher intensity than a low-equity class', () => {
    const sorted = [...present].sort((a, b) => b.value - a.value);
    const high = sorted[0];
    const low = sorted[sorted.length - 1];
    expect(high.intensity).toBeGreaterThan(low.intensity);
  });
});

describe('buildGridModel — ev-loss mode', () => {
  const model = buildGridModel(heroSeat, 'ev-loss', { isHero: true, exploitabilityPct: 0.3 });
  const present = model.cells.filter((c: GridCell) => c.present);

  it('all intensities in [0,1]', () => {
    for (const c of present) {
      expect(c.intensity).toBeGreaterThanOrEqual(0);
      expect(c.intensity).toBeLessThanOrEqual(1 + 1e-9);
    }
  });

  it('class 22 (all-CALL = hero_action) has near-zero loss', () => {
    const cell = model.cells.find((c) => c.label === '22')!;
    expect(cell.present).toBe(true);
    expect(cell.intensity).toBeCloseTo(0, 3);
  });
});

describe('buildGridModel — action mode reach is per-combo, not combo-count biased', () => {
  // Two classes, each fully played at per-combo weight 1.0: a pair (6 combos)
  // and an offsuit (12 combos). Their reach must be equal — reach reflects
  // average per-combo weight, independent of how many combos a class spans.
  const pairCombos = ['2c2d', '2c2h', '2c2s', '2d2h', '2d2s', '2h2s'];
  const offsuitCombos = [
    'Ac2d', 'Ac2h', 'Ac2s', 'Ad2c', 'Ad2h', 'Ad2s',
    'Ah2c', 'Ah2d', 'Ah2s', 'As2c', 'As2d', 'As2h',
  ];
  const combos = [...pairCombos, ...offsuitCombos];
  const weights = combos.map(() => 1.0);
  const seat = {
    combos,
    weights,
    equity: combos.map(() => 0.5),
    strategy: combos.map(() => 1.0),
    actions: ['CALL'],
    hero_action: 'CALL',
  } as Parameters<typeof buildGridModel>[0];

  const model = buildGridModel(seat, 'action', { isHero: true });
  const pairCell = model.cells.find((c) => c.label === '22')!;
  const offsuitCell = model.cells.find((c) => c.label === 'A2o')!;

  it('both classes present', () => {
    expect(pairCell.present).toBe(true);
    expect(offsuitCell.present).toBe(true);
  });

  it('reach is equal despite 6 vs 12 combos', () => {
    expect(pairCell.reach).toBeCloseTo(offsuitCell.reach, 6);
    expect(pairCell.reach).toBeCloseTo(1, 6);
  });
});

describe('buildGridModel — villain seat (oop, no strategy) in action mode', () => {
  const model = buildGridModel(villainSeat, 'action', { isHero: false });
  const present = model.cells.filter((c) => c.present);

  it('falls back to range fill: segments are empty', () => {
    for (const c of present) {
      expect(c.segments.length).toBe(0);
    }
  });

  it('falls back to range fill: intensity spread present (some > 0)', () => {
    expect(present.some((c) => c.intensity > 0)).toBe(true);
  });

  it('legend is empty for villain action mode (range fallback)', () => {
    expect(model.legend.length).toBe(0);
  });
});
