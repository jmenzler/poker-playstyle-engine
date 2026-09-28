import type { ActionLegendItem } from '@/lib/gridModel';

interface ActionLegendProps {
  legend: ActionLegendItem[];
}

export default function ActionLegend({ legend }: ActionLegendProps) {
  if (legend.length === 0) return null;
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 12px', marginTop: 6 }}>
      {legend.map((item) => (
        <div key={item.action} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <div
            style={{
              width: 10,
              height: 10,
              background: item.color,
              borderRadius: 2,
              flexShrink: 0,
            }}
          />
          <span className="cap" style={{ fontSize: 9 }}>{item.action}</span>
          <span className="muted" style={{ fontSize: 9 }}>{item.freqPct.toFixed(1)}%</span>
          <span className="muted" style={{ fontSize: 9 }}>{item.combos.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}
