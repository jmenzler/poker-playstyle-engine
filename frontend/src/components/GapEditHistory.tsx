import { useEffect, useState } from 'react';
import { api } from '@/lib/api';

interface HistoryEntry {
  ts: string;
  action_dist: Record<string, number>;
  status: 'applied' | 'rolled_back' | string;
}

interface GapEditHistoryProps {
  decisionId: string;
}

function formatActionDist(dist: Record<string, number>): string {
  return Object.entries(dist)
    .map(([action, freq]) => `${action}: ${Math.round(freq * 100)}%`)
    .join(' · ');
}

export default function GapEditHistory({ decisionId }: GapEditHistoryProps) {
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!open || loaded) return;
    api
      .get<HistoryEntry[]>(`/api/gaps/${encodeURIComponent(decisionId)}/history`)
      .then((rows) => {
        setEntries(rows);
        setLoaded(true);
      })
      .catch((e) => {
        setErr(e instanceof Error ? e.message : String(e));
        setLoaded(true);
      });
  }, [open, loaded, decisionId]);

  const n = entries.length;

  return (
    <div>
      <button
        onClick={() => setOpen((o) => !o)}
        className="text-[11px] text-fg border border-border px-2 py-0.5"
      >
        {open ? `[ edit history (${n}) ↑ ]` : `[ edit history (${n}) ]`}
      </button>

      {open && (
        <div className="mt-2">
          {err && <div className="text-destructive text-[12px]">ERROR: {err}</div>}

          {!err && loaded && entries.length === 0 && (
            <div className="text-fg-muted text-[12px]">no manual edits yet for this decision</div>
          )}

          {!err && entries.length > 0 && (
            <table className="w-full text-[12px] mt-1">
              <tbody>
                {[...entries].reverse().map((entry, i) => (
                  <tr key={i} className="border-b border-border-faint">
                    <td style={{ width: 120 }} className="text-[11px] text-fg-muted py-1 pr-2">
                      {entry.ts.slice(0, 19)}
                    </td>
                    <td className="flex-1 text-[11px] text-fg py-1 pr-2">
                      {formatActionDist(entry.action_dist)}
                    </td>
                    <td style={{ width: 80 }} className="text-[11px] py-1">
                      {entry.status === 'rolled_back' ? (
                        <span className="text-fg-muted">rolled back</span>
                      ) : (
                        <span className="text-success">{entry.status || 'applied'}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
