import { useState, useEffect } from 'react';
import PlayingCard from './PlayingCard';

const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'] as const;
const SUITS = ['s', 'h', 'd', 'c'] as const;
const SUIT_GLYPH: Record<string, string> = { s: '♠', h: '♥', d: '♦', c: '♣' };

export interface CardPickerProps {
  target: 'hole' | 'board';
  slots: number;
  selected: string[];
  usedElsewhere: Set<string>;
  onChange: (cards: string[]) => void;
  onClose?: () => void;
}

export default function CardPicker({
  target,
  slots,
  selected,
  usedElsewhere,
  onChange,
  onClose,
}: CardPickerProps) {
  const [filter, setFilter] = useState<string>('all');
  const [hover, setHover] = useState<string | null>(null);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') { onClose?.(); }
      if (e.key === '1') setFilter('s');
      if (e.key === '2') setFilter('h');
      if (e.key === '3') setFilter('d');
      if (e.key === '4') setFilter('c');
      if (e.key === '0') setFilter('all');
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  function toggle(card: string) {
    if (selected.includes(card)) {
      onChange(selected.filter((c) => c !== card));
      return;
    }
    if (selected.length >= slots) return;
    onChange([...selected, card]);
  }

  const usedSet = new Set([...usedElsewhere]);
  const atCapacity = selected.length >= slots;

  return (
    <div className="picker" role="dialog" aria-label="card picker">
      <div className="picker-toolbar">
        <span className="cap">filter suit</span>
        <div className="filter-suit">
          <button
            className={filter === 'all' ? 'active' : ''}
            onClick={() => setFilter('all')}
            type="button"
          >all</button>
          {SUITS.map((s) => (
            <button
              key={s}
              className={`s-${s}${filter === s ? ' active' : ''}`}
              onClick={() => setFilter(s)}
              type="button"
            >{SUIT_GLYPH[s]} {s === 's' ? 'spades' : s === 'h' ? 'hearts' : s === 'd' ? 'diamonds' : 'clubs'}</button>
          ))}
        </div>
        <span className="spacer"></span>
        <span className="dim" style={{ fontSize: 10 }}>
          hover preview: <span className="muted">{hover || '—'}</span>
          {' · '}{target} — {selected.length} / {slots}
        </span>
      </div>

      <div className="picker-grid">
        {SUITS.map((s) => (
          <>
            <div key={s + '-label'} className={`picker-row-label s-${s}`}>{SUIT_GLYPH[s]}</div>
            {RANKS.map((r) => {
              const card = r + s;
              const isUsed = usedSet.has(card);
              const isSelected = selected.includes(card);
              const isDimmed = filter !== 'all' && filter !== s;
              const isDisabled = isUsed || (!isSelected && atCapacity) || isDimmed;
              return (
                <button
                  key={card}
                  className={
                    'picker-card' +
                    (isUsed ? ' disabled' : '') +
                    (isSelected ? ' selected' : '') +
                    (isDimmed ? ' disabled' : '')
                  }
                  disabled={isDisabled}
                  onMouseEnter={() => setHover(card)}
                  onMouseLeave={() => setHover((h) => (h === card ? null : h))}
                  onClick={() => { if (!isDisabled) toggle(card); }}
                  title={isUsed ? `${card} already used` : card}
                  type="button"
                >
                  <PlayingCard card={card} />
                  {isUsed && !isDimmed && (
                    <span className="picker-card-disabled-overlay">used</span>
                  )}
                </button>
              );
            })}
          </>
        ))}
      </div>

      <div className="picker-foot">
        <div className="picker-foot-shortcuts">
          <span><b>1234</b> filter suit</span>
          <span><b>0</b> all</span>
          <span><b>esc</b> dismiss</span>
        </div>
        <div className="muted">duplicates disabled · click to select · {slots - selected.length} slots remaining</div>
      </div>
    </div>
  );
}
