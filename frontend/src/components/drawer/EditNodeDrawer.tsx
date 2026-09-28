import { useEffect, useState } from 'react';
import Drawer from './Drawer';
import { api } from '@/lib/api';
import { truncateClusterKey } from '@/lib/format';
import { CANONICAL_ACTIONS } from '@/lib/canonicalActions';

// Edit Node drawer per D-NEW-26 + UI-SPEC v2.2 §Drawer Patterns.
//
// - Loads context from /api/probe?cluster_key=... (engine_response + neighbors + solver)
// - 4-column compare row: current / proposed / kNN / solver_truth
// - action_dist editor with sum-row validation (Δ ±0.001)
// - Lock toggle hits POST /api/edit-node/lock
// - Save patch hits POST /api/edit-node (sum-validated)

interface ProbeNeighbor {
  cluster_key: string;
  distance: number;
  action_dist: Record<string, number> | null;
}

interface ProbeEngineResponse {
  source: string | null;
  n_obs: number;
  action_dist: Record<string, number> | null;
  ev_loss: number | null;
}

interface ProbeResponse {
  engine_response: ProbeEngineResponse;
  knn_neighbors: ProbeNeighbor[];
  solver_truth: { action_dist: Record<string, number> } | null;
}

interface NodeContext {
  current: Record<string, number>;
  knn_blended?: Record<string, number>;
  solver_truth?: Record<string, number>;
  source: string;
  n_obs: number;
  ev_loss: number | null;
  locked: boolean;
}

interface EditNodeDrawerProps {
  open: boolean;
  onClose: () => void;
  clusterKey: string | null;
  onSaved?: () => void;
}

export default function EditNodeDrawer({
  open,
  onClose,
  clusterKey,
  onSaved,
}: EditNodeDrawerProps) {
  const [ctx, setCtx] = useState<NodeContext | null>(null);
  const [proposed, setProposed] = useState<Record<string, number>>({});
  const [reason, setReason] = useState('');
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!open || !clusterKey) {
      // Reset drawer state when closed or no cluster — intentional state sync
      // with the (open, clusterKey) props. Cascading renders are benign here
      // because the !open path doesn't render any state-dependent UI.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setCtx(null);
      setProposed({});
      setReason('');
      setErr(null);
      return;
    }
    setErr(null);
    // Fetch context via Probe endpoint — it already returns engine_response +
    // neighbors + solver_truth in the shape we need for the 4-col compare row.
    api
      .get<ProbeResponse>(
        `/api/probe?cluster_key=${encodeURIComponent(clusterKey)}&k=5`,
      )
      .then((d) => {
        const er = d.engine_response;
        const blended = d.knn_neighbors?.[0]?.action_dist ?? undefined;
        setCtx({
          current: er.action_dist ?? {},
          knn_blended: blended ?? undefined,
          solver_truth: d.solver_truth?.action_dist,
          source: er.source ?? 'unknown',
          n_obs: er.n_obs,
          ev_loss: er.ev_loss,
          // Backend doesn't yet return strategy_nodes.locked_from_autoloop in
          // probe response — defaults to false; toggling hits /lock endpoint.
          locked: false,
        });
        setProposed({ ...(er.action_dist ?? {}) });
      })
      .catch((e) => setErr(String(e)));
  }, [open, clusterKey]);

  const sum = Object.values(proposed).reduce((s, v) => s + (v || 0), 0);
  const valid = Math.abs(sum - 1.0) <= 0.001;

  function setFreq(action: string, v: number) {
    setProposed((p) => ({ ...p, [action]: v }));
  }

  function normalize() {
    const s = sum || 1;
    setProposed(
      Object.fromEntries(
        Object.entries(proposed).map(([k, v]) => [k, (v || 0) / s]),
      ),
    );
  }

  async function save() {
    if (!clusterKey || !valid) return;
    setSaving(true);
    setErr(null);
    try {
      await api.post('/api/edit-node', {
        cluster_key: clusterKey,
        action_dist: proposed,
        reason,
      });
      onSaved?.();
      onClose();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(msg);
    } finally {
      setSaving(false);
    }
  }

  async function toggleLock() {
    if (!clusterKey || !ctx) return;
    try {
      await api.post('/api/edit-node/lock', {
        cluster_key: clusterKey,
        locked: !ctx.locked,
      });
      setCtx({ ...ctx, locked: !ctx.locked });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(msg);
    }
  }

  const title = clusterKey
    ? `:: EDIT NODE :: cluster ${truncateClusterKey(clusterKey, 30)} :: source=${ctx?.source ?? '...'}`
    : ':: EDIT NODE :: no cluster';

  // Build action row union — canonical actions first, then any extras present
  // in current/proposed/knn/solver that aren't canonical (defensive).
  const allActions = new Set<string>(CANONICAL_ACTIONS);
  if (ctx) {
    Object.keys(ctx.current).forEach((k) => allActions.add(k));
    if (ctx.knn_blended) Object.keys(ctx.knn_blended).forEach((k) => allActions.add(k));
    if (ctx.solver_truth) Object.keys(ctx.solver_truth).forEach((k) => allActions.add(k));
  }
  Object.keys(proposed).forEach((k) => allActions.add(k));

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title={title}
      footer={
        <div className="flex gap-2">
          <button
            onClick={normalize}
            className="text-fg-muted border border-border px-2 py-1 text-[12px]"
          >
            [ ⌘shift-r normalize ]
          </button>
          <button
            onClick={onClose}
            className="text-destructive border border-destructive px-2 py-1 text-[12px]"
          >
            [ discard ]
          </button>
          <button
            onClick={save}
            disabled={!valid || saving}
            className={`border px-2 py-1 text-[12px] ${
              valid && !saving
                ? 'text-success border-success'
                : 'text-fg-muted border-border opacity-50'
            }`}
          >
            [ save patch ]
          </button>
        </div>
      }
    >
      {err && (
        <div className="text-destructive mb-2 text-[12px]">ERROR: {err}</div>
      )}
      {!ctx && !err && (
        <div className="text-fg-muted text-[12px]">loading cluster context...</div>
      )}
      {ctx && (
        <>
          {/* Top context bar */}
          <div className="text-[12px] text-fg-muted mb-4 pb-2 border-b border-border-faint">
            n_obs: {ctx.n_obs} · ev_loss: {ctx.ev_loss?.toFixed(3) ?? '—'} · source:{' '}
            {ctx.source}
          </div>

          {/* 4-column compare row (D-NEW-26 #3) */}
          <div className="mb-4">
            <div className="text-[10px] text-fg-caption mb-2">
              ACTION DIST COMPARE
            </div>
            <table className="w-full text-[11px]">
              <thead className="text-fg-muted">
                <tr>
                  <th className="text-left p-1">action</th>
                  <th className="text-right p-1">current</th>
                  <th className="text-right p-1">proposed</th>
                  <th className="text-right p-1">kNN</th>
                  <th className="text-right p-1">solver</th>
                </tr>
              </thead>
              <tbody>
                {Array.from(allActions).map((a) => {
                  const cur = ctx.current[a] ?? 0;
                  const prop = proposed[a] ?? 0;
                  const knn = ctx.knn_blended?.[a] ?? 0;
                  const sol = ctx.solver_truth?.[a];
                  if (cur === 0 && prop === 0 && knn === 0 && (sol ?? 0) === 0)
                    return null;
                  return (
                    <tr key={a} className="border-b border-border-faint">
                      <td className="p-1 text-fg">{a}</td>
                      <td className="p-1 text-right text-fg-muted">
                        {cur.toFixed(3)}
                      </td>
                      <td className="p-1 text-right">
                        <input
                          type="number"
                          step="0.01"
                          min="0"
                          max="1"
                          value={prop.toFixed(3)}
                          onChange={(e) =>
                            setFreq(a, parseFloat(e.target.value) || 0)
                          }
                          className="w-20 text-right bg-page border border-border px-1"
                        />
                      </td>
                      <td className="p-1 text-right text-fg-muted">
                        {knn.toFixed(3)}
                      </td>
                      <td className="p-1 text-right text-success">
                        {sol !== undefined ? sol.toFixed(3) : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Sum row */}
          <div
            className={`text-[12px] mb-4 ${
              valid ? 'text-success' : 'text-destructive'
            }`}
          >
            sum: {sum.toFixed(3)} / 1.000 (Δ {(sum - 1).toFixed(3)})
          </div>

          {/* Lock toggle */}
          <div className="mb-4">
            <label className="flex items-center gap-2 text-[12px]">
              <input
                type="checkbox"
                checked={ctx.locked}
                onChange={toggleLock}
              />
              [ {ctx.locked ? '■' : '□'} lock this cluster from autoloop ]
            </label>
          </div>

          {/* Reason textarea */}
          <div>
            <label className="text-[10px] text-fg-caption">REASON</label>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              className="w-full mt-1 h-20 bg-page border border-border p-2 text-[12px]"
              placeholder="why this edit?"
            />
          </div>
        </>
      )}
    </Drawer>
  );
}
