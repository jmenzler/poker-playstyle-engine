import { useEffect, useRef, useState } from 'react';
import { api, attachJobListeners, getApiBase } from '@/lib/api';
import { truncateClusterKey, formatCI, nAnnotation } from '@/lib/format';
import EditNodeDrawer from '@/components/drawer/EditNodeDrawer';

// Strategy Leaks panel per D-NEW-28 + UI-SPEC v2.2 §Strategy Leaks panel.
//
// - GET /api/leaks?type=strategy&min_n=N&limit=N
// - Per-row [verify] → POST /api/verify; subscribes to /api/verify/sse/{job_id}
//   raw EventSource (per-row state map keeps verify/SSE local rather than
//   funneling through useJobSse which is single-job).
// - Stage A glyph ~ (gray), Stage B ✓ (emerald); verifying ● (amber);
//   failed ✗ (red).
// - Multi-select → [verify selected N] bulk; suppress hits /api/suppressions.
// - [edit] opens EditNodeDrawer.

interface StrategyRow {
  cluster_key: string;
  ev_loss: number;
  ci_low: number;
  ci_high: number;
  n_obs: number;
  source: string;
  stage: 'A' | 'B';
  shrunk: boolean;
  score: number;
}

interface RowJob {
  jobId: string;
  status: 'running' | 'done' | 'failed';
}

export default function Strategy() {
  const [rows, setRows] = useState<StrategyRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [rowJobs, setRowJobs] = useState<Record<string, RowJob>>({});
  const [editCluster, setEditCluster] = useState<string | null>(null);
  const [minN, setMinN] = useState(20);
  const openStreams = useRef<Map<string, EventSource>>(new Map());

  useEffect(() => {
    reload();
    // re-fetch when minN changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [minN]);

  useEffect(() => {
    const streams = openStreams.current;
    return () => {
      for (const es of streams.values()) es.close();
      streams.clear();
    };
  }, []);

  function reload() {
    setErr(null);
    api
      .get<{ strategy: StrategyRow[] }>(
        `/api/leaks?type=strategy&min_n=${minN}&limit=50`,
      )
      .then((d) => setRows(d.strategy ?? []))
      .catch((e) => setErr(String(e)));
  }

  function toggleSelect(clusterKey: string, checked: boolean) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (checked) next.add(clusterKey);
      else next.delete(clusterKey);
      return next;
    });
  }

  async function verify(clusterKey: string) {
    try {
      const r = await api.post<{ job_id: string }>('/api/verify', {
        cluster_key: clusterKey,
        force: false,
      });
      setRowJobs((j) => ({
        ...j,
        [clusterKey]: { jobId: r.job_id, status: 'running' },
      }));
      // Subscribe via raw EventSource — per-row state map keeps a clean
      // visual progression PENDING → VERIFYING → VERIFIED/REJECTED.
      const es = new EventSource(
        `${getApiBase()}/api/verify/sse/${r.job_id}`,
      );
      openStreams.current.get(clusterKey)?.close();
      openStreams.current.set(clusterKey, es);
      const dropStream = () => {
        if (openStreams.current.get(clusterKey) === es) {
          openStreams.current.delete(clusterKey);
        }
      };
      // Shared listener discriminates a named server error (real failure) from a
      // transient transport blip (let EventSource auto-reconnect, don't mark failed).
      attachJobListeners(es, {
        onDone: () => {
          setRowJobs((j) => ({
            ...j,
            [clusterKey]: { ...j[clusterKey], status: 'done' },
          }));
          dropStream();
          reload();
        },
        onError: () => {
          setRowJobs((j) => ({
            ...j,
            [clusterKey]: { ...j[clusterKey], status: 'failed' },
          }));
          dropStream();
        },
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      alert(`verify failed: ${msg}`);
    }
  }

  async function verifySelected() {
    for (const ck of selected) {
      // sequential; backend semaphore is 1 anyway
      await verify(ck);
    }
    setSelected(new Set());
  }

  async function suppress(clusterKey: string) {
    if (!confirm(`mark ${clusterKey.slice(0, 20)}... as intentional?`)) return;
    try {
      await api.post('/api/suppressions', {
        cluster_key: clusterKey,
        reason: 'manual suppress from strategy panel',
      });
      reload();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      alert(`suppress failed: ${msg}`);
    }
  }

  function stageGlyph(row: StrategyRow): string {
    const job = rowJobs[row.cluster_key];
    if (job?.status === 'running') return '●';
    if (job?.status === 'failed') return '✗';
    if (row.stage === 'B') return '✓';
    return '~';
  }

  function stageColor(row: StrategyRow): string {
    const job = rowJobs[row.cluster_key];
    if (job?.status === 'running') return 'text-warning';
    if (job?.status === 'failed') return 'text-destructive';
    if (row.stage === 'B') return 'text-success';
    return 'text-fg-muted';
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h1 className="text-[14px] font-bold">
          :: STRATEGY LEAKS :: top {rows.length} :: ev_loss × log(n_obs)
        </h1>
        <div className="flex gap-2 text-[12px] items-center">
          <span className="text-fg-muted">min_n</span>
          <input
            type="number"
            value={minN}
            onChange={(e) => setMinN(parseInt(e.target.value, 10) || 20)}
            className="w-16 bg-page border border-border px-1 py-0.5"
            min={1}
          />
        </div>
      </div>
      {err && <div className="text-destructive text-[12px]">ERROR: {err}</div>}
      <table className="w-full text-[12px]">
        <thead className="text-fg-muted border-b border-border">
          <tr>
            <th className="text-left p-2 w-6"></th>
            <th className="text-left p-2 w-6">st</th>
            <th className="text-left p-2">cluster_key</th>
            <th className="text-left p-2">ev_loss [CI]</th>
            <th className="text-right p-2">n_obs</th>
            <th className="text-left p-2">source</th>
            <th className="text-left p-2">actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const job = rowJobs[r.cluster_key];
            return (
              <tr
                key={r.cluster_key}
                className="border-b border-border-faint hover:bg-surface-2"
              >
                <td className="p-2">
                  <input
                    type="checkbox"
                    checked={selected.has(r.cluster_key)}
                    onChange={(e) =>
                      toggleSelect(r.cluster_key, e.target.checked)
                    }
                  />
                </td>
                <td className={`p-2 ${stageColor(r)}`}>{stageGlyph(r)}</td>
                <td className="p-2 text-fg-cluster">
                  {truncateClusterKey(r.cluster_key)}
                </td>
                <td className="p-2">
                  {formatCI(r.ev_loss, r.ci_low, r.ci_high)}{' '}
                  <span className="text-[10px] text-fg-caption">
                    {nAnnotation(r.n_obs, r.shrunk)}
                  </span>
                </td>
                <td className="p-2 text-right">{r.n_obs}</td>
                <td className="p-2">{r.source}</td>
                <td className="p-2 flex gap-1">
                  {job?.status === 'running' ? (
                    <button
                      disabled
                      className="text-warning border border-warning px-2 py-0.5"
                    >
                      [ verifying… ]
                    </button>
                  ) : (
                    <button
                      onClick={() => verify(r.cluster_key)}
                      className="border border-border px-2 py-0.5"
                    >
                      [ verify ]
                    </button>
                  )}
                  <button
                    onClick={() => setEditCluster(r.cluster_key)}
                    className="border border-border px-2 py-0.5"
                  >
                    [ edit ]
                  </button>
                  <button
                    onClick={() => suppress(r.cluster_key)}
                    className="border border-destructive text-destructive px-2 py-0.5"
                  >
                    [ suppress ]
                  </button>
                </td>
              </tr>
            );
          })}
          {rows.length === 0 && !err && (
            <tr>
              <td colSpan={7} className="p-4 text-fg-muted text-center">
                no strategy leaks above threshold.
              </td>
            </tr>
          )}
        </tbody>
      </table>
      {selected.size > 0 && (
        <div className="sticky bottom-0 bg-surface border-t border-border p-2 text-[12px]">
          {selected.size} selected ·{' '}
          <button
            onClick={verifySelected}
            className="border border-border px-2 py-0.5"
          >
            [ verify selected {selected.size} ]
          </button>
        </div>
      )}
      <EditNodeDrawer
        open={!!editCluster}
        onClose={() => setEditCluster(null)}
        clusterKey={editCluster}
        onSaved={() => {
          setEditCluster(null);
          reload();
        }}
      />
    </div>
  );
}
