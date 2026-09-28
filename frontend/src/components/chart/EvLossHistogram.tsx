// EvLossHistogram — small histogram for the Strategy Leaks expand sub-row
// (UI-SPEC v2.2 §Histogram panel). v1 ships a minimal bar-chart with no axis
// labels — the section heading + n_count above provides enough context.
// Recharts is reserved for the Dashboard EV_LOSS hero chart; this widget uses
// inline divs with Tailwind to stay tiny.

interface EvLossHistogramProps {
  values: number[];
  bins?: number;
}

export default function EvLossHistogram({
  values,
  bins = 11,
}: EvLossHistogramProps) {
  if (!values.length) {
    return <div className="text-fg-muted text-[10px]">no data</div>;
  }
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const counts = new Array(bins).fill(0) as number[];
  values.forEach((v) => {
    const idx = Math.min(bins - 1, Math.floor(((v - lo) / span) * bins));
    counts[idx]++;
  });
  const max = Math.max(...counts) || 1;
  return (
    <div
      className="flex items-end gap-1 border border-border p-1 bg-page"
      style={{ height: 48 }}
      aria-label="ev_loss histogram"
    >
      {counts.map((c, i) => (
        <div
          key={i}
          className="flex-1 bg-accent"
          style={{
            height: `${(c / max) * 100}%`,
            minHeight: '1px',
          }}
          title={`bin ${i}: ${c}`}
        />
      ))}
    </div>
  );
}
