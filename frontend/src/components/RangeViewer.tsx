import { useState } from 'react';
import RangeGrid, { type FreqMap, type ActionDist } from './RangeGrid';
import {
  placeholderEquity,
  placeholderEvLoss,
  placeholderArrival,
  PLACEHOLDER_FILL_MODES,
} from '@/lib/rangeFillPlaceholders';
import {
  buildEquityFillProvider,
  buildEvLossFillProvider,
} from '@/lib/comboUtils';

const RANKS_HI = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
const COMBO_COUNTS: Record<string, number> = {};
for (let i = 0; i < RANKS_HI.length; i++) {
  for (let j = 0; j < RANKS_HI.length; j++) {
    let label: string;
    if (i === j) { label = RANKS_HI[i] + RANKS_HI[j]; COMBO_COUNTS[label] = 6; }
    else if (j > i) { label = RANKS_HI[i] + RANKS_HI[j] + 's'; COMBO_COUNTS[label] = 4; }
    else { label = RANKS_HI[j] + RANKS_HI[i] + 'o'; COMBO_COUNTS[label] = 12; }
  }
}

function rangeAggregate(
  freqMap: FreqMap | undefined,
  action: string,
  comboCounts?: Record<string, number>,
): { pct: number; combosWeighted: number; totalCombos: number } {
  if (!freqMap) return { pct: 0, combosWeighted: 0, totalCombos: 0 };
  let total = 0;
  let weighted = 0;
  for (const [label, freq] of Object.entries(freqMap)) {
    const combos = comboCounts?.[label] ?? COMBO_COUNTS[label] ?? 4;
    const f = action === 'any'
      ? Math.max(0, 1 - (freq.fold ?? 0))
      : (freq as unknown as Record<string, number>)[action] ?? 0;
    weighted += f * combos;
    total += combos;
  }
  return {
    pct: total > 0 ? (weighted / total) * 100 : 0,
    combosWeighted: weighted,
    totalCombos: total,
  };
}

const ACTION_COLORS_LEGEND = {
  raise: '#ef4444',
  call: '#10b981',
  fold: '#3a3a3a',
};

function RangeLegend({ freqMap, comboCounts }: { freqMap?: FreqMap; comboCounts?: Record<string, number> }) {
  const agg = {
    call: rangeAggregate(freqMap, 'call', comboCounts),
    raise: rangeAggregate(freqMap, 'raise', comboCounts),
    fold: rangeAggregate(freqMap, 'fold', comboCounts),
    any: rangeAggregate(freqMap, 'any', comboCounts),
  };
  const denom = agg.any.totalCombos;
  return (
    <div className="range-legend">
      {(['raise', 'call', 'fold'] as const).map((action) => (
        <div key={action} className="legend-row">
          <span className="legend-sw" style={{ background: ACTION_COLORS_LEGEND[action] }}></span>
          <span className="legend-act">{action.charAt(0).toUpperCase() + action.slice(1)}</span>
          <span className="legend-pct mono-num">{agg[action].pct.toFixed(1)}%</span>
          <span className="legend-share dim">
            {action !== 'fold'
              ? `(${agg.any.combosWeighted > 0 ? (agg[action].combosWeighted / agg.any.combosWeighted * 100).toFixed(1) : '0.0'}%)`
              : '—'}
          </span>
          <span className="legend-combos mono-num">{agg[action].combosWeighted.toFixed(2)}/{denom}</span>
        </div>
      ))}
    </div>
  );
}

type FillMode = 'action' | 'equity' | 'evLoss' | 'arrival';

export interface RangeViewerTab {
  label: string;
  actor: string;
  freqMap?: FreqMap;
  actionDist?: ActionDist;
  comboCounts?: Record<string, number>;
  heroGrid?: boolean;
  combos?: string[];
  weights?: number[];
  equity?: number[];
  ev_detail?: number[];
  actions?: string[];
  hero_action?: string | null;
  exploitabilityPct?: number;
}

interface RangeViewerProps {
  tabs?: RangeViewerTab[];
  heroHand?: string | null;
  defaultFillMode?: FillMode;
}

function RangeToolbar({
  fillMode,
  onFillModeChange,
  heroGrid = true,
}: {
  fillMode: FillMode;
  onFillModeChange: (m: FillMode) => void;
  heroGrid?: boolean;
}) {
  const modes: { v: FillMode; label: string }[] = [
    { v: 'action', label: 'action' },
    { v: 'equity', label: 'equity' },
    { v: 'evLoss', label: 'ev-loss' },
    { v: 'arrival', label: 'arrival' },
  ];
  return (
    <div className="range-toolbar">
      <span className="cap" style={{ fontSize: 9 }}>fill</span>
      <div className="range-fill-seg">
        {modes.map((m) => {
          const heroOnly = m.v === 'action' || m.v === 'evLoss';
          const disabled = heroOnly && !heroGrid;
          return (
            <button
              key={m.v}
              className={'range-fill-btn' + (fillMode === m.v ? ' active' : '')}
              onClick={() => !disabled && onFillModeChange(m.v)}
              disabled={disabled}
              type="button"
            >{m.label}</button>
          );
        })}
      </div>
      <span className="spacer"></span>
      {PLACEHOLDER_FILL_MODES.has(fillMode) && (
        <span className="range-placeholder-badge">placeholder</span>
      )}
      <span className="dim" style={{ fontSize: 10 }}>solver · 100bb · 6-max</span>
    </div>
  );
}

export default function RangeViewer({ tabs = [], heroHand, defaultFillMode = 'action' }: RangeViewerProps) {
  const [fillMode, setFillMode] = useState<FillMode>(defaultFillMode);
  const [activeIdx, setActiveIdx] = useState(0);

  const clampedIdx = Math.min(activeIdx, Math.max(0, tabs.length - 1));

  if (tabs.length === 0) {
    return (
      <div className="section">
        <div className="section-head">
          <div className="section-head-l">
            <span className="section-head-title">RANGE VIEWER</span>
            <span className="section-head-meta">· solver ranges · 13×13 grid</span>
          </div>
        </div>
        <div className="section-body muted" style={{ fontSize: 11, padding: '16px 0', textAlign: 'center' }}>
          run a query to load engine-sourced action ranges
        </div>
      </div>
    );
  }

  const active = tabs[clampedIdx];

  const fillProvider = fillMode === 'equity'
    ? (active.combos && active.weights && active.equity
        ? buildEquityFillProvider(active.combos, active.weights, active.equity)
        : placeholderEquity)
    : fillMode === 'evLoss'
    ? (active.combos && active.weights && active.ev_detail && active.actions && active.hero_action != null
        ? buildEvLossFillProvider(
            active.combos,
            active.weights,
            active.ev_detail,
            active.actions,
            active.hero_action,
            active.exploitabilityPct ?? 1.0,
          )
        : placeholderEvLoss)
    : fillMode === 'arrival'
    ? placeholderArrival
    : undefined;

  return (
    <div className="section">
      <div className="section-head">
        <div className="section-head-l">
          <span className="section-head-title">RANGE VIEWER</span>
          <span className="section-head-meta">· {active.actor} · {active.label}</span>
        </div>
      </div>

      {tabs.length > 1 && (
        <div className="range-tabs">
          {tabs.map((t, i) => (
            <button
              key={i}
              className={'range-tab' + (i === clampedIdx ? ' active' : '')}
              onClick={() => setActiveIdx(i)}
              type="button"
            >
              <span className="rt-pos">{t.actor}</span>
              <span className="rt-verb">{t.label}</span>
            </button>
          ))}
        </div>
      )}

      <div className="section-body" style={{ padding: 14 }}>
        <RangeToolbar fillMode={fillMode} onFillModeChange={setFillMode} heroGrid={active.heroGrid ?? true} />
        <RangeGrid
          freqMap={active.freqMap}
          heroHand={heroHand}
          fillMode={fillMode}
          fillProvider={fillProvider}
          actionDist={active.actionDist}
        />
        {fillMode === 'action' && (
          <>
            <div className="range-foot dim">
              <span>diagonal = pairs · upper-right = suited · lower-left = offsuit</span>
              <span className="spacer"></span>
              {heroHand && <span>hero hand: <b style={{ color: 'var(--color-accent)' }}>{heroHand}</b></span>}
            </div>
            <RangeLegend freqMap={active.freqMap} comboCounts={active.comboCounts} />
          </>
        )}
        {PLACEHOLDER_FILL_MODES.has(fillMode) && (
          <div className="range-heatmap-legend">
            {fillMode === 'equity' && <>equity vs random · <span className="dim">red 30% → green 80% · placeholder — backlog 999.8</span></>}
            {fillMode === 'evLoss' && <>EV loss bb/100 · <span className="dim">green 0 → red 0.2 · placeholder — backlog 999.8</span></>}
            {fillMode === 'arrival' && <>arrival frequency · <span className="dim">placeholder — backlog 999.8</span></>}
          </div>
        )}
      </div>
    </div>
  );
}
