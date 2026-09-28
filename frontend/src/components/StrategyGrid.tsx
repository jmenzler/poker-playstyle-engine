import type { GridModel, GridCell } from '@/lib/gridModel';

interface StrategyGridProps {
  model: GridModel;
  onCellClick?: (label: string) => void;
}

function ActionCell({ cell, onClick }: { cell: GridCell; onClick?: () => void }) {
  const { reach, segments } = cell;
  const filledPct = reach * 100;

  return (
    <div
      className="hc"
      style={{
        position: 'relative',
        background: '#111',
        cursor: onClick ? 'pointer' : undefined,
        overflow: 'hidden',
      }}
      onClick={onClick}
    >
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: `${filledPct}%`,
          display: 'flex',
          flexDirection: 'column-reverse',
        }}
      >
        {segments.map((seg) => (
          <div
            key={seg.action}
            style={{
              height: `${seg.freq * 100}%`,
              background: seg.color,
              flexShrink: 0,
            }}
          />
        ))}
      </div>
      <span className="hc-label" style={{ position: 'relative', zIndex: 1 }}>
        {cell.label}
      </span>
    </div>
  );
}

function ScalarCell({ cell, onClick }: { cell: GridCell; onClick?: () => void }) {
  return (
    <div
      className={'hc' + (!cell.present ? ' hc-dim' : '')}
      style={{
        background: cell.color,
        cursor: onClick ? 'pointer' : undefined,
      }}
      onClick={onClick}
    >
      <span className="hc-label">{cell.label}</span>
    </div>
  );
}

export default function StrategyGrid({ model, onCellClick }: StrategyGridProps) {
  return (
    <div className="rangegrid range-grid">
      {model.cells.map((cell) => {
        const handleClick = onCellClick ? () => onCellClick(cell.label) : undefined;
        if (model.mode === 'action' && cell.present && cell.segments.length > 0) {
          return (
            <ActionCell key={cell.label} cell={cell} onClick={handleClick} />
          );
        }
        return (
          <ScalarCell key={cell.label} cell={cell} onClick={handleClick} />
        );
      })}
    </div>
  );
}
