// Pure combo-string utilities: combo→169-class rollup + fill-mode provider builders.
// Combo format: 4-char rank+suit+rank+suit (e.g. "AhKh"). Verified A1 against live API.

import type { FillModeProvider } from '@/lib/rangeFillPlaceholders';

const RANK_ORDER = 'AKQJT98765432';

export const EV_LOSS_NOISE_MULT = 17;

export function comboToClass(combo: string): string {
  const r1 = combo[0], s1 = combo[1];
  const r2 = combo[2], s2 = combo[3];

  if (r1 === r2) return r1 + r2;

  const hiIsFirst = RANK_ORDER.indexOf(r1) < RANK_ORDER.indexOf(r2);
  const hi = hiIsFirst ? r1 : r2;
  const lo = hiIsFirst ? r2 : r1;
  const hiSuit = hiIsFirst ? s1 : s2;
  const loSuit = hiIsFirst ? s2 : s1;
  const suffix = hiSuit === loSuit ? 's' : 'o';
  return `${hi}${lo}${suffix}`;
}

interface ComboClassData {
  combos: string[];
  indices: number[];
}

export interface ComboClassMap {
  classToData: Record<string, ComboClassData>;
}

export function buildComboClassMap(combos: string[]): ComboClassMap {
  const classToData: Record<string, ComboClassData> = {};
  for (let i = 0; i < combos.length; i++) {
    const cls = comboToClass(combos[i]);
    if (!classToData[cls]) classToData[cls] = { combos: [], indices: [] };
    classToData[cls].combos.push(combos[i]);
    classToData[cls].indices.push(i);
  }
  return { classToData };
}

export function buildRangeFillProvider(combos: string[], weights: number[]): FillModeProvider {
  const total = weights.reduce((s, w) => s + w, 0);
  const classWeightMap: Record<string, number> = {};
  for (let i = 0; i < combos.length; i++) {
    const cls = comboToClass(combos[i]);
    classWeightMap[cls] = (classWeightMap[cls] ?? 0) + weights[i] / total;
  }
  return (cls) => classWeightMap[cls] ?? 0;
}

export function buildEquityFillProvider(
  combos: string[],
  weights: number[],
  equity: number[],
): FillModeProvider {
  const classEquity: Record<string, number> = {};
  const classWeight: Record<string, number> = {};
  for (let i = 0; i < combos.length; i++) {
    const cls = comboToClass(combos[i]);
    classEquity[cls] = (classEquity[cls] ?? 0) + weights[i] * equity[i];
    classWeight[cls] = (classWeight[cls] ?? 0) + weights[i];
  }
  return (cls) => {
    const w = classWeight[cls] ?? 0;
    return w > 0 ? (classEquity[cls] ?? 0) / w : 0;
  };
}

export function buildEvLossFillProvider(
  combos: string[],
  weights: number[],
  evDetail: number[],
  actions: string[],
  heroAction: string | null,
  exploitabilityPct: number,
): FillModeProvider {
  const nCombos = combos.length;
  const heroActionIdx = heroAction ? actions.indexOf(heroAction) : -1;
  const noiseFloor = exploitabilityPct * EV_LOSS_NOISE_MULT;

  const classEvLoss: Record<string, number> = {};
  const classWeight: Record<string, number> = {};
  for (let h = 0; h < nCombos; h++) {
    const evs = actions.map((_, a) => evDetail[a * nCombos + h]);
    const maxEv = Math.max(...evs);
    const heroEv = heroActionIdx >= 0 ? evDetail[heroActionIdx * nCombos + h] : maxEv;
    const loss = Math.max(0, maxEv - heroEv) < noiseFloor ? 0 : Math.max(0, maxEv - heroEv);
    const cls = comboToClass(combos[h]);
    classEvLoss[cls] = (classEvLoss[cls] ?? 0) + weights[h] * loss;
    classWeight[cls] = (classWeight[cls] ?? 0) + weights[h];
  }
  return (cls) => {
    const w = classWeight[cls] ?? 0;
    return w > 0 ? (classEvLoss[cls] ?? 0) / w : 0;
  };
}

export function buildClassActionMap(
  combos: string[],
  weights: number[],
  strategy: number[],
  actions: string[],
): Record<string, Record<string, number>> {
  const nCombos = combos.length;
  const classActionWeightSum: Record<string, Record<string, number>> = {};
  const classWeight: Record<string, number> = {};

  for (let h = 0; h < nCombos; h++) {
    const cls = comboToClass(combos[h]);
    const w = weights[h];
    if (!classActionWeightSum[cls]) {
      classActionWeightSum[cls] = {};
      classWeight[cls] = 0;
    }
    classWeight[cls] += w;
    for (let a = 0; a < actions.length; a++) {
      const freq = strategy[a * nCombos + h];
      classActionWeightSum[cls][actions[a]] = (classActionWeightSum[cls][actions[a]] ?? 0) + w * freq;
    }
  }

  const result: Record<string, Record<string, number>> = {};
  for (const cls of Object.keys(classActionWeightSum)) {
    const totalW = classWeight[cls] ?? 0;
    result[cls] = {};
    for (const action of actions) {
      result[cls][action] = totalW > 0 ? (classActionWeightSum[cls][action] ?? 0) / totalW : 0;
    }
  }
  return result;
}

export function buildActionFillProvider(
  combos: string[],
  weights: number[],
  strategy: number[],
  actions: string[],
  primaryAction: string,
): FillModeProvider {
  const classActionMap = buildClassActionMap(combos, weights, strategy, actions);
  return (cls) => classActionMap[cls]?.[primaryAction] ?? 0;
}
