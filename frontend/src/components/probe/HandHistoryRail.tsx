import type { EncodeForm, EncodeResult } from '@/lib/probeEncoder';
import PlayingCard from '../PlayingCard';

interface HandHistoryRailProps {
  form: EncodeForm;
  enc?: EncodeResult;
  focusedStreet: string | null;
  onFocus: (street: string) => void;
  reachedMap: Record<string, boolean>;
  onEditBoard?: (street: string) => void;
}

function classifyPreflop(tape: EncodeForm['preflop']): string {
  if (tape.some((t) => t.verb === '5bet')) return '5BP+';
  if (tape.some((t) => t.verb === '4bet')) return '4BP';
  if (tape.some((t) => t.verb === '3bet')) return '3BP';
  if (tape.some((t) => t.verb === 'open')) return 'SRP';
  if (tape.some((t) => t.verb === 'limp')) return 'LIMP';
  return 'WALK';
}

interface ChipArgs {
  streetKey: string;
  label: string;
  primary: React.ReactNode;
  secondary?: string | null;
  tag?: React.ReactNode;
  tagClass?: string;
  dim: boolean;
  invalid: boolean;
  canEdit: boolean;
  focusedStreet: string | null;
  decisionStreet: string;
  onFocus: (street: string) => void;
  onEditBoard?: (street: string) => void;
}

function Chip({ streetKey: chipKey, label, primary, secondary, tag, tagClass, dim, invalid, canEdit, focusedStreet, decisionStreet, onFocus, onEditBoard }: ChipArgs) {
  const isActive = focusedStreet === chipKey || (focusedStreet == null && decisionStreet === chipKey);
  return (
    <button
      className={
        'hh-rail-chip' +
        (isActive ? ' active' : '') +
        (dim ? ' dim' : '') +
        (invalid ? ' invalid' : '')
      }
      onClick={() => onFocus(chipKey)}
      type="button"
    >
      <div className="hh-rail-chip-label">
        <span>
          {label}
          {canEdit && onEditBoard && (
            <span
              className="hh-rail-chip-edit"
              title="edit cards — clear and re-pick"
              onClick={(e) => { e.stopPropagation(); onEditBoard(chipKey); }}
            > ✎</span>
          )}
        </span>
        {tag && <span className={'hh-rail-chip-label-tag ' + (tagClass || '')}>{tag}</span>}
      </div>
      <div className="hh-rail-chip-body">
        <div className={'hh-rail-chip-primary' + (!primary ? ' muted' : '')}>{primary || '—'}</div>
        {secondary && <div className="hh-rail-chip-secondary">{secondary}</div>}
      </div>
    </button>
  );
}

export default function HandHistoryRail({ form, enc, focusedStreet, onFocus, reachedMap, onEditBoard }: HandHistoryRailProps) {
  const pre = classifyPreflop(form.preflop);
  const heroPos = form.heroPos;

  const decisionStreet = enc?.parts.street ?? (() => {
    if (form.river.board.filter(Boolean).length > 0 || form.river.actions.length > 0) return 'river';
    if (form.turn.board.filter(Boolean).length > 0 || form.turn.actions.length > 0) return 'turn';
    if (form.flop.board.filter(Boolean).length > 0 || form.flop.actions.length > 0) return 'flop';
    return 'preflop';
  })();

  function streetChip(name: 'flop' | 'turn' | 'river') {
    const data = form[name];
    const need = { flop: 3, turn: 1, river: 1 }[name];
    const cards = data.board.slice(0, need).filter(Boolean) as string[];
    const acts = data.actions.length;
    return { cards, acts, complete: cards.length === need };
  }
  const flop = streetChip('flop');
  const turn = streetChip('turn');
  const river = streetChip('river');

  const chipProps = { focusedStreet, decisionStreet, onFocus, onEditBoard };

  return (
    <div className="hh-rail">
      <Chip
        streetKey="preflop"
        label="preflop"
        tag={
          (heroPos || form.preflop.length > 0)
            ? <span className="hh-pre-tag">
                <span className={'hh-pre-pos' + (heroPos ? '' : ' dim')}>{heroPos || '—'}</span>
                <span className="hh-pre-sep">|</span>
                <span className={'hh-pre-pot' + (form.preflop.length > 0 ? '' : ' dim')}>
                  {form.preflop.length > 0 ? pre : '—'}
                </span>
              </span>
            : '—'
        }
        tagClass=""
        primary={
          form.heroHole.every(Boolean)
            ? <span>{form.heroHole.map((c, i) => (
                <span key={i} className="mb-card">
                  <PlayingCard card={c ?? undefined} />
                </span>
              ))}</span>
            : null
        }
        secondary={null}
        dim={false}
        invalid={!heroPos || !form.heroHole.every(Boolean)}
        canEdit={false}
        {...chipProps}
      />
      <span className="hh-rail-arrow">›</span>
      <Chip
        streetKey="flop"
        label="flop"
        tag={flop.acts > 0 ? `${flop.acts} act` : '—'}
        primary={
          flop.cards.length > 0
            ? <span>{flop.cards.map((c, i) => (
                <span key={i} className="mb-card">
                  <PlayingCard card={c ?? undefined} />
                </span>
              ))}</span>
            : null
        }
        secondary={null}
        dim={!reachedMap.flop}
        invalid={false}
        canEdit={flop.complete}
        {...chipProps}
      />
      <span className="hh-rail-arrow">›</span>
      <Chip
        streetKey="turn"
        label="turn"
        tag={turn.acts > 0 ? `${turn.acts} act` : '—'}
        primary={
          turn.cards.length > 0
            ? <span>{turn.cards.map((c, i) => (
                <span key={i} className="mb-card">
                  <PlayingCard card={c ?? undefined} />
                </span>
              ))}</span>
            : null
        }
        secondary={null}
        dim={!reachedMap.turn}
        invalid={false}
        canEdit={turn.complete}
        {...chipProps}
      />
      <span className="hh-rail-arrow">›</span>
      <Chip
        streetKey="river"
        label="river"
        tag={river.acts > 0 ? `${river.acts} act` : '—'}
        primary={
          river.cards.length > 0
            ? <span>{river.cards.map((c, i) => (
                <span key={i} className="mb-card">
                  <PlayingCard card={c ?? undefined} />
                </span>
              ))}</span>
            : null
        }
        secondary={null}
        dim={!reachedMap.river}
        invalid={false}
        canEdit={river.complete}
        {...chipProps}
      />
    </div>
  );
}
