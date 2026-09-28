import type { FillModeProvider } from '@/lib/rangeFillPlaceholders';

const RANKS_HI = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];

const ACTION_COLORS = {
  raise: '#ef4444',
  call: '#10b981',
  fold: '#3a3a3a',
};

export interface ActionDist {
  call?: number;
  raise?: number;
  fold?: number;
  [key: string]: number | undefined;
}

export interface HandFreq {
  call: number;
  raise: number;
  fold: number;
}

export type FreqMap = Record<string, HandFreq>;

function buildSplitGradient(action_dist: ActionDist): string {
  const total = (action_dist.call ?? 0) + (action_dist.raise ?? 0) + (action_dist.fold ?? 0);
  const fold = total > 0 ? (action_dist.fold ?? 0) / total : 1;
  const call = total > 0 ? (action_dist.call ?? 0) / total : 0;
  return `linear-gradient(to top,
    ${ACTION_COLORS.fold} 0%,
    ${ACTION_COLORS.fold} ${fold * 100}%,
    ${ACTION_COLORS.call} ${fold * 100}%,
    ${ACTION_COLORS.call} ${(fold + call) * 100}%,
    ${ACTION_COLORS.raise} ${(fold + call) * 100}%,
    ${ACTION_COLORS.raise} 100%)`;
}

interface HandCellProps {
  label: string;
  freq?: HandFreq;
  isHero: boolean;
  fillMode: string;
  fillProvider?: FillModeProvider;
  actionDist?: ActionDist;
  onCellClick?: (handClass: string) => void;
}

function HandCell({ label, freq, isHero, fillMode, fillProvider, actionDist, onCellClick }: HandCellProps) {
  let background: string;

  if (fillMode !== 'action' && fillProvider) {
    const val = fillProvider(label);
    if (fillMode === 'equity') {
      const t = Math.max(0, Math.min(1, (val - 30) / 50));
      const r = Math.round(239 - 175 * t);
      const g = Math.round(60 + 125 * t);
      background = `rgb(${r}, ${g}, 70)`;
    } else if (fillMode === 'evLoss' || fillMode === 'ev-loss') {
      const t = Math.max(0, Math.min(1, val / 0.2));
      const r = Math.round(60 + 180 * t);
      const g = Math.round(180 - 140 * t);
      background = `rgb(${r}, ${g}, 70)`;
    } else if (fillMode === 'range') {
      const t = Math.max(0, Math.min(1, val));
      const v = Math.round(20 + 160 * t);
      background = t > 0 ? `rgb(${v}, ${v}, ${Math.round(v * 1.1)})` : '#111';
    } else {
      const t = Math.max(0, Math.min(1, val / 0.9));
      const v = Math.round(40 + 140 * t);
      background = `rgb(${v}, ${v}, ${Math.round(v * 1.05)})`;
    }
  } else if (fillMode === 'action' && actionDist) {
    background = buildSplitGradient(actionDist);
  } else if (freq) {
    const total = freq.call + freq.raise + freq.fold;
    const fold = total > 0 ? freq.fold / total : 1;
    const call = total > 0 ? freq.call / total : 0;
    background = `linear-gradient(to top,
      ${ACTION_COLORS.fold} 0%,
      ${ACTION_COLORS.fold} ${fold * 100}%,
      ${ACTION_COLORS.call} ${fold * 100}%,
      ${ACTION_COLORS.call} ${(fold + call) * 100}%,
      ${ACTION_COLORS.raise} ${(fold + call) * 100}%,
      ${ACTION_COLORS.raise} 100%)`;
  } else {
    background = ACTION_COLORS.fold;
  }

  const isPair = label.length === 2;
  const isSuited = label.endsWith('s');
  const isOffsuit = label.endsWith('o');

  return (
    <div
      className={'hc' + (isHero ? ' hc-hero' : '')}
      style={{ background, cursor: onCellClick ? 'pointer' : undefined }}
      onClick={onCellClick ? () => onCellClick(label) : undefined}
    >
      <span className={
        'hc-label' +
        (isPair ? ' hc-pair' : '') +
        (isSuited ? ' hc-suited' : '') +
        (isOffsuit ? ' hc-offsuit' : '')
      }>{label}</span>
    </div>
  );
}

interface RangeGridProps {
  freqMap?: FreqMap;
  heroHand?: string | null;
  fillMode: string;
  fillProvider?: FillModeProvider;
  actionDist?: ActionDist;
  onCellClick?: (handClass: string) => void;
}

export default function RangeGrid({ freqMap, heroHand, fillMode, fillProvider, actionDist, onCellClick }: RangeGridProps) {
  return (
    <div className="rangegrid range-grid">
      {RANKS_HI.map((r1, i) => (
        <>
          {RANKS_HI.map((r2, j) => {
            let label: string;
            if (i === j) label = r1 + r2;
            else if (j > i) label = r1 + r2 + 's';
            else label = r2 + r1 + 'o';
            return (
              <HandCell
                key={label}
                label={label}
                freq={freqMap?.[label]}
                isHero={label === heroHand}
                fillMode={fillMode}
                fillProvider={fillProvider}
                actionDist={actionDist}
                onCellClick={onCellClick}
              />
            );
          })}
        </>
      ))}
    </div>
  );
}
