import '../routes/replayer-felt.css';
import PlayingCard from './PlayingCard';

const N_SLOTS = 6;
const RX = 320;
const RY = 170;
const BASE_ANGLE = Math.PI / 2;

function seatPos(slot: number, heroSlot: number, rx: number, ry: number): React.CSSProperties {
  const theta = BASE_ANGLE + ((slot - heroSlot) / N_SLOTS) * 2 * Math.PI;
  return {
    left: `calc(50% + ${rx * Math.cos(theta)}px)`,
    top: `calc(50% + ${ry * Math.sin(theta)}px)`,
    transform: 'translate(-50%, -50%)',
  };
}

function Card({ card }: { card?: string }) {
  return <PlayingCard card={card} className="felt-card" style={{ width: 56, height: 'auto' }} />;
}

export interface SnapshotProps {
  board?: string[];
  heroHole?: string[];
  pot?: number;
  effectiveStack?: number;
  street?: string;
  heroPos?: string;
  actionSequence?: string[];
  actionTaken?: string;
}

export default function FeltSnapshot({
  board = [],
  heroHole = [],
  pot,
  effectiveStack,
  street,
  heroPos,
  actionSequence,
  actionTaken,
}: SnapshotProps) {
  const hasVisual = board.length > 0 || heroHole.length > 0;

  return (
    <div>
      {hasVisual && (
        <div data-felt className="flex justify-center py-2">
          <div className="felt-table" style={{ width: 640, height: 340, position: 'relative' }}>
            {/* Board cards centered above pot */}
            <div
              style={{
                position: 'absolute',
                left: '50%',
                top: 'calc(50% - 70px)',
                transform: 'translate(-50%, -50%)',
                display: 'flex',
                gap: 4,
              }}
            >
              {board.map((card, i) => <Card key={i} card={card} />)}
            </div>
            {/* Center pot */}
            <div className="felt-pot">
              <span className="felt-chip" />
              <span>{pot != null ? `${pot}bb` : '—'}</span>
            </div>
            {/* Effective stack below pot */}
            {effectiveStack != null && (
              <div
                style={{
                  position: 'absolute',
                  left: '50%',
                  top: 'calc(50% + 20px)',
                  transform: 'translate(-50%, -50%)',
                  fontSize: 10,
                  color: '#666666',
                }}
              >
                eff: {effectiveStack}bb
              </div>
            )}
            {/* Hero seat bottom-center (slot 0) */}
            <div className="felt-seat" style={seatPos(0, 0, RX, RY)}>
              <div className="flex justify-center gap-1 mt-1">
                {heroHole.map((card, i) => <Card key={i} card={card} />)}
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Text description — always rendered */}
      <div className="text-[12px] text-fg-muted mt-2">
        {street} · {heroPos} · pot {pot != null ? `${pot}bb` : '—'} · eff {effectiveStack != null ? `${effectiveStack}bb` : '—'}
        {actionSequence && actionSequence.length > 0 && (
          <><br />actions: {actionSequence.join(' → ')}</>
        )}
        {actionTaken && (
          <><br />hero: {actionTaken}</>
        )}
      </div>
    </div>
  );
}
