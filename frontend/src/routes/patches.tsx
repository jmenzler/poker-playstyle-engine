import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { api } from '@/lib/api';
import { truncateClusterKey } from '@/lib/format';

// Patches panel per UI-SPEC v2.2 §Patches.
// Reverse-chronological table; per-row rollback button + bulk selection bar.
// FastAPI endpoint: GET /api/patches?limit=N (Plan 06-07 `study.patches.list_patches`).
//
// Note: a POST /api/patches/{patch_id}/rollback endpoint may not yet exist —
// the rollback button surfaces an error if the backend returns 404. The CLI
// surface (`poker-engine patches rollback ...`) is the authoritative path.

interface Patch {
  patch_id: string;
  ts: string;
  cluster_key: string;
  source: string;
  pre_ev_loss: number | null;
  post_ev_loss: number | null;
  status: string;
  prev_patch_id?: string | null;
}

export default function Patches() {
  const navigate = useNavigate();
  const [rows, setRows] = useState<Patch[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  useEffect(() => {
    reload();
  }, []);

  function reload() {
    setErr(null);
    api
      .get<Patch[]>('/api/patches?limit=50')
      .then(setRows)
      .catch((e) => setErr(String(e)));
  }

  function toggle(patchId: string, checked: boolean) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (checked) next.add(patchId);
      else next.delete(patchId);
      return next;
    });
  }

  async function rollback(patchId: string) {
    if (
      !confirm(
        `rollback patch ${patchId.slice(0, 8)}? this is auditable but not reversible without re-patch.`,
      )
    )
      return;
    try {
      await api.post(`/api/patches/${patchId}/rollback`, {});
      reload();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      alert(`rollback failed: ${msg}`);
    }
  }

  async function rollbackSelected() {
    const ids = Array.from(selected);
    if (ids.length === 0) return;
    if (
      !confirm(
        `rollback ${ids.length} selected patch(es)? this is auditable but not reversible without re-patch.`,
      )
    )
      return;
    const failures: string[] = [];
    for (const id of ids) {
      try {
        await api.post(`/api/patches/${id}/rollback`, {});
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        failures.push(`${id.slice(0, 8)}: ${msg}`);
      }
    }
    setSelected(new Set());
    reload();
    if (failures.length > 0) alert(`rollback failed for:\n${failures.join('\n')}`);
  }

  return (
    <div className="space-y-4">
      <h1 className="text-[14px] font-bold">:: PATCHES :: reverse chronological</h1>
      {err && <div className="text-destructive text-[12px]">ERROR: {err}</div>}
      <table className="w-full text-[12px]">
        <thead className="text-fg-muted border-b border-border">
          <tr>
            <th className="text-left p-2 w-8"></th>
            <th className="text-left p-2">patch_id</th>
            <th className="text-left p-2">ts</th>
            <th className="text-left p-2">cluster_key</th>
            <th className="text-left p-2">source</th>
            <th className="text-right p-2">pre_ev</th>
            <th className="text-right p-2">post_ev</th>
            <th className="text-left p-2">status</th>
            <th className="text-left p-2">actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.patch_id} className="border-b border-border-faint hover:bg-surface-2">
              <td className="p-2">
                <input
                  type="checkbox"
                  checked={selected.has(r.patch_id)}
                  onChange={(e) => toggle(r.patch_id, e.target.checked)}
                />
              </td>
              <td className="p-2">
                <button
                  onClick={() => navigate(`/patches/${encodeURIComponent(r.patch_id)}`)}
                  className="text-accent hover:underline"
                >
                  {r.patch_id.slice(0, 8)}
                </button>
              </td>
              <td className="p-2 text-fg-muted">{r.ts?.slice(0, 19)}</td>
              <td className="p-2 text-fg-cluster">{truncateClusterKey(r.cluster_key)}</td>
              <td className="p-2">{r.source}</td>
              <td className="p-2 text-right">{r.pre_ev_loss?.toFixed(3) ?? '—'}</td>
              <td className="p-2 text-right">{r.post_ev_loss?.toFixed(3) ?? '—'}</td>
              <td className="p-2">{r.status}</td>
              <td className="p-2">
                <button onClick={() => rollback(r.patch_id)} className="text-destructive">
                  [ rollback ]
                </button>
              </td>
            </tr>
          ))}
          {rows.length === 0 && !err && (
            <tr>
              <td colSpan={9} className="p-4 text-fg-muted text-center">
                no patches yet
              </td>
            </tr>
          )}
        </tbody>
      </table>
      {selected.size > 0 && (
        <div className="sticky bottom-0 bg-surface border-t border-border p-2 text-[12px]">
          {selected.size} selected ·{' '}
          <button onClick={rollbackSelected} className="text-destructive">
            [ rollback selected {selected.size} ]
          </button>
        </div>
      )}
    </div>
  );
}
