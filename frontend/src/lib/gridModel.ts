import {
  buildClassActionMap,
  buildEquityFillProvider,
  buildEvLossFillProvider,
} from '@/lib/comboUtils';
import type { RangeSeatPayload } from '@/lib/rangeClient';

export type FillMode = 'action' | 'range' | 'equity' | 'ev-loss';

export interface ActionSegment {
  action: string;
  verb: 'fold' | 'call' | 'raise' | 'allin';
  freq: number;
  color: string;
}

export interface GridCell {
  label: string;
  row: number;
  col: number;
  present: boolean;
  reach: number;
  segments: ActionSegment[];
  intensity: number;
  color: string;
  value: number;
}

export interface ActionLegendItem {
  action: string;
  verb: string;
  freqPct: number;
  combos: number;
  color: string;
}

export interface GridModel {
  mode: FillMode;
  cells: GridCell[];
  legend: ActionLegendItem[];
}

const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];

export function actionVerb(action: string): 'fold' | 'call' | 'raise' | 'allin' {
  const word = action.trim().split(/\s+/)[0].toLowerCase();
  if (word === 'fold') return 'fold';
  if (word === 'call' || word === 'check') return 'call';
  if (word === 'bet' || word === 'raise') return 'raise';
  if (word === 'allin' || word === 'shove') return 'allin';
  return 'fold';
}

export function actionColor(action: string): string {
  const verb = actionVerb(action);
  if (verb === 'fold') return '#2563eb';
  if (verb === 'call') return '#10b981';
  if (verb === 'raise') return '#ef4444';
  return '#7f1d1d';
}

function rangeColor(intensity: number): string {
  if (intensity <= 0) return '#111';
  const v = Math.round(20 + 160 * intensity);
  return `rgb(${v}, ${v}, ${Math.round(v * 1.1)})`;
}

function equityColor(t: number): string {
  const r = Math.round(239 - 175 * t);
  const g = Math.round(60 + 125 * t);
  return `rgb(${r}, ${g}, 70)`;
}

function evLossColor(t: number): string {
  const r = Math.round(60 + 180 * t);
  const g = Math.round(180 - 140 * t);
  return `rgb(${r}, ${g}, 70)`;
}

function classLabel(row: number, col: number): string {
  const r1 = RANKS[row];
  const r2 = RANKS[col];
  if (row === col) return r1 + r2;
  if (col > row) return r1 + r2 + 's';
  return r2 + r1 + 'o';
}

function buildClassWeightMap(
  combos: string[],
  weights: number[],
): Record<string, number> {
  const map: Record<string, number> = {};
  for (let i = 0; i < combos.length; i++) {
    const cls = comboClassOf(combos[i]);
    map[cls] = (map[cls] ?? 0) + weights[i];
  }
  return map;
}

function buildClassComboCountMap(combos: string[]): Record<string, number> {
  const map: Record<string, number> = {};
  for (const c of combos) {
    const cls = comboClassOf(c);
    map[cls] = (map[cls] ?? 0) + 1;
  }
  return map;
}

function comboClassOf(combo: string): string {
  const r1 = combo[0], s1 = combo[1], r2 = combo[2], s2 = combo[3];
  if (r1 === r2) return r1 + r2;
  const hiFirst = RANKS.indexOf(r1) < RANKS.indexOf(r2);
  const hi = hiFirst ? r1 : r2;
  const lo = hiFirst ? r2 : r1;
  const hiS = hiFirst ? s1 : s2;
  const loS = hiFirst ? s2 : s1;
  return hi + lo + (hiS === loS ? 's' : 'o');
}

export function buildGridModel(
  seat: RangeSeatPayload,
  mode: FillMode,
  opts: { isHero: boolean; exploitabilityPct?: number },
): GridModel {
  const { combos, weights, equity } = seat;
  const isHero = opts.isHero;
  const hasStrategy = isHero && !!seat.strategy && !!seat.actions;

  const effectiveMode: FillMode =
    (mode === 'action' || mode === 'ev-loss') && !hasStrategy ? 'range' : mode;

  const classWeightMap = buildClassWeightMap(combos, weights);
  const classComboCount = buildClassComboCountMap(combos);

  const presentClasses = new Set(Object.keys(classWeightMap));

  let classActionMap: Record<string, Record<string, number>> = {};
  if (effectiveMode === 'action' && hasStrategy) {
    classActionMap = buildClassActionMap(combos, weights, seat.strategy!, seat.actions!);
  }

  let equityProvider: ((cls: string) => number) | null = null;
  if (effectiveMode === 'equity') {
    equityProvider = buildEquityFillProvider(combos, weights, equity);
  }

  let evLossProvider: ((cls: string) => number) | null = null;
  if (effectiveMode === 'ev-loss' && hasStrategy) {
    evLossProvider = buildEvLossFillProvider(
      combos,
      weights,
      seat.ev_detail!,
      seat.actions!,
      seat.hero_action ?? null,
      opts.exploitabilityPct ?? 0.3,
    );
  }

  let maxClassAvgWeight = 1;
  if ((effectiveMode === 'range' || effectiveMode === 'action') && presentClasses.size > 0) {
    maxClassAvgWeight = Math.max(
      ...Array.from(presentClasses).map(
        (c) => classWeightMap[c] / (classComboCount[c] ?? 1),
      ),
    );
  }

  let maxEvLoss = 1;
  if (effectiveMode === 'ev-loss' && evLossProvider && presentClasses.size > 0) {
    maxEvLoss = Math.max(...Array.from(presentClasses).map((c) => evLossProvider!(c)));
    if (maxEvLoss <= 0) maxEvLoss = 1;
  }

  const cells: GridCell[] = [];

  for (let row = 0; row < 13; row++) {
    for (let col = 0; col < 13; col++) {
      const label = classLabel(row, col);
      const present = presentClasses.has(label);

      if (!present) {
        cells.push({
          label, row, col, present: false,
          reach: 0, segments: [], intensity: 0,
          color: '#111', value: 0,
        });
        continue;
      }

      if (effectiveMode === 'action') {
        const actionFreqs = classActionMap[label] ?? {};
        const avgWeight = classWeightMap[label] / (classComboCount[label] ?? 1);
        const classReach = maxClassAvgWeight > 0 ? avgWeight / maxClassAvgWeight : 0;
        const segments: ActionSegment[] = seat.actions!
          .map((act) => ({ action: act, freq: actionFreqs[act] ?? 0 }))
          .filter((s) => s.freq > 1e-6)
          .map((s) => ({
            action: s.action,
            verb: actionVerb(s.action),
            freq: s.freq,
            color: actionColor(s.action),
          }));

        cells.push({
          label, row, col, present: true,
          reach: classReach,
          segments,
          intensity: classReach,
          color: '#111',
          value: classWeightMap[label],
        });
      } else if (effectiveMode === 'range') {
        const avgWeight = classWeightMap[label] / (classComboCount[label] ?? 1);
        const intensity = maxClassAvgWeight > 0 ? avgWeight / maxClassAvgWeight : 0;
        cells.push({
          label, row, col, present: true,
          reach: 0, segments: [],
          intensity,
          color: rangeColor(intensity),
          value: avgWeight,
        });
      } else if (effectiveMode === 'equity') {
        const fracEq = equityProvider!(label);
        const intensity = Math.max(0, Math.min(1, (fracEq - 0.30) / 0.50));
        cells.push({
          label, row, col, present: true,
          reach: 0, segments: [],
          intensity,
          color: equityColor(intensity),
          value: fracEq,
        });
      } else {
        const avgLoss = evLossProvider!(label);
        const intensity = Math.max(0, Math.min(1, avgLoss / maxEvLoss));
        cells.push({
          label, row, col, present: true,
          reach: 0, segments: [],
          intensity,
          color: evLossColor(intensity),
          value: avgLoss,
        });
      }
    }
  }

  const legend: ActionLegendItem[] = buildLegend(seat, hasStrategy, effectiveMode);

  return { mode, cells, legend };
}

function buildLegend(
  seat: RangeSeatPayload,
  hasStrategy: boolean,
  effectiveMode: FillMode,
): ActionLegendItem[] {
  if (effectiveMode !== 'action' || !hasStrategy) return [];

  const { combos, weights, strategy, actions } = seat;
  const n = combos.length;
  const total = weights.reduce((s, w) => s + w, 0);

  return actions!.map((act, a) => {
    let combosSum = 0;
    for (let h = 0; h < n; h++) {
      combosSum += weights[h] * strategy![a * n + h];
    }
    return {
      action: act,
      verb: actionVerb(act),
      freqPct: total > 0 ? (combosSum / total) * 100 : 0,
      combos: combosSum,
      color: actionColor(act),
    };
  });
}
