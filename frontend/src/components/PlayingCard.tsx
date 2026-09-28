import type { CSSProperties } from 'react';
import './playing-card.css';

const CARD_RE = /^[2-9TJQKA][cdhs]$/;
const SUITS: Record<string, string> = { c: '♣', d: '♦', h: '♥', s: '♠' };

export default function PlayingCard({
  card,
  className = '',
  style,
}: {
  card?: string;
  className?: string;
  style?: CSSProperties;
}) {
  const face = card && CARD_RE.test(card) ? card : null;
  const tone = face ? (/[hd]$/.test(face) ? 'red' : 'black') : 'back';

  return (
    <span
      role="img"
      aria-label={face ?? 'back'}
      className={`playing-card playing-card--${tone} ${className}`}
      style={style}
    >
      {face && <span aria-hidden="true" className="playing-card-face">
        <span>{face[0] === 'T' ? '10' : face[0]}</span>
        <span>{SUITS[face[1]]}</span>
      </span>}
    </span>
  );
}
