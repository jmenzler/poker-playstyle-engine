import type { EncodeForm, EncodeResult } from '@/lib/probeEncoder';
import FeltSnapshot from '@/components/FeltSnapshot';

interface SpotPreviewProps {
  form: EncodeForm;
  enc: EncodeResult;
}

export default function SpotPreview({ form, enc }: SpotPreviewProps) {
  const heroPos = form.heroPos;

  // Pot = sum of each player's final committed chips. Raise verbs carry size as
  // the raise-TO total (not an increment), so the committed map overwrites rather
  // than accumulates; summing raise-TO values would over-count the pot.
  const committed: Record<string, number> = { SB: 0.5, BB: 1.0 };
  let currentBet = 1.0;
  for (const t of form.preflop) {
    if (t.verb === 'fold') continue;
    if (t.verb === 'limp') {
      committed[t.pos] = 1.0;
    } else if (t.verb === 'call') {
      committed[t.pos] = t.size ?? currentBet;
    } else if (typeof t.size === 'number' && t.size > 0) {
      committed[t.pos] = t.size;
      currentBet = t.size;
    }
  }
  const potBb = Object.values(committed).reduce((s, c) => s + c, 0);
  const maxCommitted = Object.values(committed).reduce((m, c) => Math.max(m, c), 0);
  const effStack = Math.max(0, 100 - maxCommitted);

  const street = enc.parts.street;
  const potType = enc.parts.potType;
  const ipoop = enc.parts.position;
  const boardClass = enc.parts.boardClass;

  const board = [
    ...form.flop.board,
    ...form.turn.board,
    ...form.river.board,
  ].filter(Boolean) as string[];
  const heroHole = form.heroHole.filter(Boolean) as string[];
  const actionSequence = form.preflop
    .filter((t) => t.verb !== 'fold')
    .map((t) => `${t.pos}:${t.verb}${t.size != null ? `_${t.size}` : ''}`);

  return (
    <div className="spot-preview">
      <div className="spot-preview-head">
        <span className="spot-preview-head-title">SPOT PREVIEW</span>
        <span className="muted">
          {street} · {potType} · {ipoop || '—'}
        </span>
      </div>
      <div className="spot-preview-body">
        <FeltSnapshot
          board={board}
          heroHole={heroHole}
          pot={potBb}
          effectiveStack={effStack}
          street={street}
          heroPos={heroPos ?? undefined}
          actionSequence={actionSequence}
        />

        <div className="spot-stats">
          <div className="spot-stats-cell">
            <div className="spot-stats-cell-label">street</div>
            <div className="spot-stats-cell-value">{street}</div>
          </div>
          <div className="spot-stats-cell">
            <div className="spot-stats-cell-label">pot type</div>
            <div className="spot-stats-cell-value accent">{potType}</div>
          </div>
          <div className="spot-stats-cell">
            <div className="spot-stats-cell-label">pos</div>
            <div className="spot-stats-cell-value">{ipoop || '—'}</div>
          </div>
          <div className="spot-stats-cell">
            <div className="spot-stats-cell-label">texture</div>
            <div className={'spot-stats-cell-value' + (!boardClass ? ' muted' : '')}>
              {boardClass || '—'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
