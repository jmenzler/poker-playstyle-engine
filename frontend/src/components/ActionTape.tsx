import { useState } from 'react';
import type { TapeStep } from '@/lib/spotBuilder';

const PREFLOP_VERBS = ['open', 'limp', 'call', '3bet', '4bet', '5bet', 'fold'];
const POSTFLOP_VERBS = ['check', 'bet', 'call', 'raise', 'fold'];
const RAISES_PREFLOP = new Set(['open', '3bet', '4bet', '5bet']);
const RAISES_POSTFLOP = new Set(['bet', 'raise']);

function defaultSize(verb: string, isPreflop: boolean): string {
  if (isPreflop) {
    if (verb === 'open') return '2.5';
    if (verb === '3bet') return '8';
    if (verb === '4bet') return '21';
    if (verb === '5bet') return '60';
  } else {
    if (verb === 'bet') return '50';
    if (verb === 'raise') return '3x';
  }
  return '';
}

export interface ActionTapeConfig {
  street: 'preflop' | 'flop' | 'turn' | 'river';
  emitsPosition: boolean;
  verbSet?: string[];
  heroPos: string;
  villainSeat?: string;
  defaultPos?: string;
  positionOptions?: string[];
}

interface ActionTapeProps {
  config: ActionTapeConfig;
  steps: TapeStep[];
  onChange: (steps: TapeStep[]) => void;
  onOpenRangeFor?: (stepIdx: number) => void;
}

export default function ActionTape({ config, steps, onChange, onOpenRangeFor }: ActionTapeProps) {
  const { street, heroPos, villainSeat, defaultPos, positionOptions } = config;
  const isPreflop = street === 'preflop';
  const defaultVerbs = isPreflop ? PREFLOP_VERBS : POSTFLOP_VERBS;
  const verbsToShow = config.verbSet && config.verbSet.length > 0 ? config.verbSet : defaultVerbs;
  const sizeOnlyForRaises = isPreflop ? RAISES_PREFLOP : RAISES_POSTFLOP;
  const posOptions = positionOptions && positionOptions.length > 0 ? positionOptions : ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB'];

  const [userVerb, setUserVerb] = useState<string | null>(null);
  const [userPos, setUserPos] = useState<string | null>(null);
  const [size, setSize] = useState('');

  const verb = userVerb && verbsToShow.includes(userVerb) ? userVerb : (verbsToShow[0] || 'check');
  const pos = userPos || defaultPos || posOptions[0] || 'UTG';

  const sizeValue = userVerb !== null ? size : defaultSize(verb, isPreflop);

  function handleVerbChange(newVerb: string) {
    setUserVerb(newVerb);
    setSize(defaultSize(newVerb, isPreflop));
  }

  function handleSizeChange(newSize: string) {
    setSize(newSize);
  }

  function add() {
    const sizeToUse = sizeValue;
    const amount = sizeOnlyForRaises.has(verb) && sizeToUse ? (parseFloat(sizeToUse) || undefined) : undefined;
    const step: TapeStep = {
      ...(config.emitsPosition ? { pos } : {}),
      action: verb,
      ...(amount != null ? { amount } : {}),
    };
    onChange([...steps, step]);
  }

  function removeAt(idx: number) {
    onChange(steps.filter((_, j) => j !== idx));
  }

  return (
    <div>
      {steps.length === 0 ? (
        <div className="tape-empty-line">— no {street} action yet · add steps below —</div>
      ) : (
        <div className="tape-list">
          {steps.map((t, i) => {
            const stepPos = t.pos || (i % 2 === 0 ? heroPos : (villainSeat || ''));
            const isHero = stepPos === heroPos;
            const isFold = t.action === 'fold';
            const isRaise = RAISES_POSTFLOP.has(t.action) || RAISES_PREFLOP.has(t.action);
            return (
              <div
                key={i}
                className={
                  'tape-list-step' +
                  (isHero ? ' hero' : '') +
                  (isFold ? ' folded' : '') +
                  (isRaise ? ' raise' : '')
                }
              >
                <span className="tls-bullet"></span>
                <span className="tls-pos">{stepPos || ''}</span>
                <span className="tls-eye-col">
                  {onOpenRangeFor && !isFold && (
                    <span
                      className="tls-eye"
                      title={`view ${stepPos} ${t.action} solver range`}
                      onClick={(e) => { e.stopPropagation(); onOpenRangeFor(i); }}
                    >⧈</span>
                  )}
                </span>
                <span className="tls-verb">{t.action}</span>
                <span className="tls-size">
                  {t.amount != null
                    ? (isPreflop ? `${t.amount}bb` : `${t.amount}%`)
                    : ''}
                </span>
                <span className="dim" style={{ fontSize: 10 }}>
                  {isHero ? 'hero' : ''}
                </span>
                <span className="tls-x" onClick={() => removeAt(i)} title="remove this step">×</span>
              </div>
            );
          })}
        </div>
      )}

      <div className="add-action-row">
        <span className="tls-bullet"></span>
        {config.emitsPosition && (
          <select value={pos} onChange={(e) => setUserPos(e.target.value)}>
            {posOptions.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        )}
        <select value={verb} onChange={(e) => handleVerbChange(e.target.value)}>
          {verbsToShow.map((v) => <option key={v} value={v}>{v}</option>)}
        </select>
        {sizeOnlyForRaises.has(verb) ? (
          <input
            type="text"
            value={sizeValue}
            onChange={(e) => handleSizeChange(e.target.value)}
            placeholder={isPreflop ? 'bb' : '%'}
          />
        ) : (
          <span className="dim" style={{ fontSize: 10 }}>—</span>
        )}
        <button className="aar-add" onClick={add} disabled={!verb} type="button">+ add action</button>
      </div>
    </div>
  );
}
