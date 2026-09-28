import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { truncateClusterKey } from '@/lib/format';

interface GapRow {
  decision_id: string;
  hand_id: string;
  cluster_key: string;
  max_neighbor_distance: number;
  is_multiway: boolean;
  n_players_active: number;
  street: string;
  send_to_solver_disabled: boolean;
}

type Filter = 'all' | 'hu-only' | 'multiway';

const FILTER_QUERY: Record<Filter, string> = {
  all: 'all',
  'hu-only': 'hu',
  multiway: 'multiway',
};

function statusChip(row: GapRow): { label: string; cls: string } {
  if (row.send_to_solver_disabled) {
    return { label: 'pending', cls: 'text-fg-muted' };
  }
  return { label: 'pending', cls: 'text-fg-muted' };
}

export default function GapPanel() {
  const [gaps, setGaps] = useState<GapRow[]>([]);
  const [filter, setFilter] = useState<Filter>('all');
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setErr(null);
    api
      .get<GapRow[]>(`/api/gaps?limit=50&filter=${FILTER_QUERY[filter]}`)
      .then((rows) => {
        setGaps(rows);
        setLoading(false);
      })
      .catch((e) => {
        setErr(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
  }, [filter]);

  const tabCls = (t: Filter) =>
    filter === t
      ? 'bg-[#d4d4d4] text-[#000000] px-2 py-0.5 text-[11px]'
      : 'border border-border text-fg text-[11px] px-2 py-0.5';

  return (
    <section className="border border-border p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="text-[10px] text-fg-caption">GAP QUEUE</div>
        <div className="flex gap-1">
          <button onClick={() => setFilter('all')} className={tabCls('all')}>
            [ all ]
          </button>
          <button onClick={() => setFilter('hu-only')} className={tabCls('hu-only')}>
            [ hu-only ]
          </button>
          <button onClick={() => setFilter('multiway')} className={tabCls('multiway')}>
            [ multiway ]
          </button>
        </div>
      </div>

      {loading && (
        <div className="text-fg-muted text-[12px]">loading gap queue…</div>
      )}

      {!loading && err && (
        <div className="text-destructive text-[12px]">ERROR: {err}</div>
      )}

      {!loading && !err && gaps.length === 0 && (
        <div className="text-fg-muted text-[12px] text-center py-4">
          no flagged_sparse gaps in queue
        </div>
      )}

      {!loading && !err && gaps.length > 0 && (
        <table className="w-full text-[12px]">
          <tbody>
            {gaps.map((row, i) => {
              const chip = statusChip(row);
              return (
                <tr
                  key={row.decision_id}
                  className="border-b border-border-faint hover:bg-surface-2"
                  style={{ height: 36 }}
                >
                  <td style={{ width: 24 }} className="text-[11px] text-fg-muted pr-1">
                    {i + 1}
                  </td>
                  <td style={{ width: 56 }} className="text-[12px] tabular-nums pr-2">
                    {row.max_neighbor_distance.toFixed(3)}
                  </td>
                  <td className="flex-1 text-[11px] text-fg-cluster truncate pr-2 max-w-[200px]">
                    {truncateClusterKey(row.cluster_key)}
                  </td>
                  <td style={{ width: 48 }} className="text-[11px] text-fg-muted pr-2">
                    {row.street}
                  </td>
                  <td style={{ width: 32 }} className="pr-1">
                    {row.is_multiway && (
                      <span className="text-warning text-[10px]">MW</span>
                    )}
                  </td>
                  <td style={{ width: 72 }} className={`text-[11px] pr-2 ${chip.cls}`}>
                    {chip.label}
                  </td>
                  <td>
                    <a
                      href={`/gap-resolver?decision_id=${encodeURIComponent(row.decision_id)}&hand_id=${encodeURIComponent(row.hand_id)}`}
                      className="text-[11px] text-fg hover:text-accent"
                    >
                      [ resolve → ]
                    </a>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}
