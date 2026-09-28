import { describe, it, expect } from 'vitest';
import {
  comboToClass,
  buildComboClassMap,
  buildEquityFillProvider,
  buildEvLossFillProvider,
} from '../comboUtils';

describe('comboToClass', () => {
  it('suited combos map to classname with "s" suffix', () => {
    expect(comboToClass('AhKh')).toBe('AKs');
  });

  it('offsuit combos map to classname with "o" suffix', () => {
    expect(comboToClass('AhKd')).toBe('AKo');
  });

  it('pocket pairs map to two-char classname', () => {
    expect(comboToClass('AhAs')).toBe('AA');
  });

  it('hi-card ordering is independent of input card order', () => {
    expect(comboToClass('KhAh')).toBe('AKs');
  });
});

describe('buildComboClassMap', () => {
  it('groups suited combos into the correct class', () => {
    const { classToData } = buildComboClassMap(['AhKh', 'AsKs', 'AhKd']);
    expect(classToData['AKs'].combos.length).toBe(2);
  });

  it('groups offsuit combos into the correct class', () => {
    const { classToData } = buildComboClassMap(['AhKh', 'AsKs', 'AhKd']);
    expect(classToData['AKo'].combos.length).toBe(1);
  });

  it('indices map back to input positions', () => {
    const combos = ['AhKh', 'AsKs', 'AhKd'];
    const { classToData } = buildComboClassMap(combos);
    for (const cls of Object.values(classToData)) {
      for (const idx of cls.indices) {
        expect(idx).toBeGreaterThanOrEqual(0);
        expect(idx).toBeLessThan(combos.length);
        expect(cls.combos).toContain(combos[idx]);
      }
    }
  });
});

describe('evLoss noise', () => {
  it('returns 0 from the provider when ev-loss is below noise floor', () => {
    const combos = ['AhKh'];
    const weights = [1];
    const actions = ['CHECK', 'BET'];
    // ev_detail: action-major layout [ev_check, ev_bet] for the one combo
    // heroAction = 'CHECK', baseline is max(ev_bet=0.5, ev_check=0.4) = 0.5, loss = 0.5 - 0.4 = 0.1
    // noiseFloor = exploitabilityPct * 17 = 10 * 17 = 170 — but ev_detail is in bb units; use small loss
    // To get below floor: loss = 0.001, noiseFloor = 0.01 * 17 = 0.17 → 0.001 < 0.17 → returns 0
    const evDetail = [0.099, 0.1]; // heroAction=CHECK, ev_check=0.099, ev_bet=0.1, loss=0.001
    const exploitabilityPct = 1; // noiseFloor = 1 * 17 = 17 mbb → 0.017bb → 0.001 < 0.017 → 0
    const provider = buildEvLossFillProvider(combos, weights, evDetail, actions, 'CHECK', exploitabilityPct);
    expect(provider('AKs')).toBe(0);
  });

  it('returns > 0 from the provider when ev-loss is above noise floor', () => {
    const combos = ['AhKh'];
    const weights = [1];
    const actions = ['CHECK', 'BET'];
    // ev_detail: ev_check=0.0, ev_bet=1.0, heroAction=CHECK, loss = 1.0 - 0.0 = 1.0
    // noiseFloor = 0 * 17 = 0 → any positive loss is above floor
    const evDetail = [0.0, 1.0];
    const exploitabilityPct = 0;
    const provider = buildEvLossFillProvider(combos, weights, evDetail, actions, 'CHECK', exploitabilityPct);
    expect(provider('AKs')).toBeGreaterThan(0);
  });
});

describe('equity fill', () => {
  it('returns weight-averaged class equity for the given hand class', () => {
    const combos = ['AhKh', 'AsKs'];
    const weights = [1, 3];
    const equity = [0.4, 0.8];
    const provider = buildEquityFillProvider(combos, weights, equity);
    // Both are AKs; weight-avg = (1*0.4 + 3*0.8) / (1+3) = 2.8 / 4 = 0.7
    expect(provider('AKs')).toBeCloseTo(0.7, 5);
  });

  it('returns 0 for a hand class with no surviving combos', () => {
    const combos = ['AhKh'];
    const weights = [1];
    const equity = [0.6];
    const provider = buildEquityFillProvider(combos, weights, equity);
    expect(provider('22')).toBe(0);
  });
});
