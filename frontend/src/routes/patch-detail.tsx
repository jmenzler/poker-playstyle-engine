import { useEffect, useState, useMemo, useCallback } from 'react';
import { useParams } from 'react-router';
import { api } from '@/lib/api';
import InlineReplayer, { ReplayStep } from '@/components/InlineReplayer';
import GapEditHistory from '@/components/GapEditHistory';
import EditNodeDrawer from '@/components/drawer/EditNodeDrawer';
import { truncateClusterKey } from '@/lib/format';
import './replayer-felt.css';

// Read-only review of one DP-level patch: the gap-resolver felt in review mode
// (full hand, god-view cards, no authoring) plus an original-vs-override compare.

interface PatchChainItem {
  patch_id: string;
  ts: string;
  source: string;
  pre_ev_loss: number | null;
  post_ev_loss: number | null;
}

interface PatchDetail {
  patch_id: string;
  ts: string;
  cluster_key: string;
  source: string;
  pre_ev_loss: number | null;
  post_ev_loss: number | null;
  status: string;
  prev_node_id: string | null;
  new_node_id: string | null;
  decision_id: string | null;
  hand_id: string | null;
  override_action_dist: Record<string, number> | null;
  original_action_dist: Record<string, number> | null;
  chain?: PatchChainItem[];
}

interface ProbeResponse {
  engine_response: { source: string | null; action_dist: Record<string, number> | null };
}

export default function PatchDetail() {
  const { patch_id } = useParams<{ patch_id: string }>();

  const [detail, setDetail] = useState<PatchDetail | null>(null);
  const [steps, setSteps] = useState<ReplayStep[]>([]);
  const [probeOriginal, setProbeOriginal] = useState<Record<string, number> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [editOpen, setEditOpen] = useState(false);

  const load = useCallback(() => {
    if (!patch_id) return;
    setErr(null);
    api
      .get<PatchDetail>(`/api/patches/${encodeURIComponent(patch_id)}`)
      .then((d) => {
        setDetail(d);
        if (d.hand_id) {
          api
            .get<ReplayStep[]>(`/api/hands/by-hand/${encodeURIComponent(d.hand_id)}/replay`)
            .then(setSteps)
            .catch(() => setSteps([]));
        }
        // First-ever patch (no prev node) has no stored original — the engine's
        // kNN blend for the cluster is the faithful "before".
        if (d.original_action_dist == null) {
          api
            .get<ProbeResponse>(`/api/probe?cluster_key=${encodeURIComponent(d.cluster_key)}`)
            .then((p) => setProbeOriginal(p.engine_response.action_dist))
            .catch(() => setProbeOriginal(null));
        }
      })
      .catch((e) => setErr(String(e)));
  }, [patch_id]);

  useEffect(() => {
    setDetail(null);
    setSteps([]);
    setProbeOriginal(null);
    load();
  }, [load]);

  // Anchor the felt on the seat that acted at the patched decision.
  const heroPosition = useMemo<string | null>(() => {
    if (!detail?.decision_id || !steps.length) return null;
    return steps.find((s) => s.decision_id === detail.decision_id)?.actor ?? null;
  }, [detail, steps]);

  const original = detail?.original_action_dist ?? probeOriginal;
  const override = detail?.override_action_dist ?? null;

  const compareActions = useMemo<string[]>(() => {
    const keys = new Set<string>([...Object.keys(original ?? {}), ...Object.keys(override ?? {})]);
    return Array.from(keys);
  }, [original, override]);

  if (err) {
    return (
      <div className="min-h-screen bg-page text-fg font-mono p-6">
        <div className="text-destructive text-[12px]">ERROR: {err}</div>
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="min-h-screen bg-page text-fg font-mono p-6">
        <div className="text-fg-muted text-[12px]">loading patch {patch_id}…</div>
      </div>
    );
  }

  const idShort = detail.patch_id.slice(0, 8);
  const clusterKeyTruncated = truncateClusterKey(detail.cluster_key, 40);

  return (
    <div className="min-h-screen bg-page text-fg font-mono p-4 space-y-3">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <h1 className="text-[14px] font-bold truncate">
            :: PATCH DETAIL :: patch_{idShort} · {clusterKeyTruncated}
          </h1>
          <div className="text-[12px] text-fg-muted mt-1">
            source: {detail.source} · status: {detail.status} · {detail.ts?.slice(0, 19)}
          </div>
        </div>
        <button
          onClick={() => setEditOpen(true)}
          className="border border-border px-2 py-0.5 text-[11px] text-fg flex-shrink-0"
        >
          [ modify ]
        </button>
      </div>

      {/* Metadata block */}
      <div className="text-[11px] text-fg-muted grid grid-cols-2 gap-x-6 gap-y-0.5 border-b border-border-faint pb-2">
        <div>decision_id: <span className="text-fg">{detail.decision_id ?? '—'}</span></div>
        <div>cluster_key: <span className="text-fg-cluster">{detail.cluster_key}</span></div>
        <div>pre_ev_loss: <span className="text-fg">{detail.pre_ev_loss?.toFixed(3) ?? '—'}</span></div>
        <div>post_ev_loss: <span className="text-fg">{detail.post_ev_loss?.toFixed(3) ?? '—'}</span></div>
      </div>

      {/* Original vs override compare */}
      <div>
        <div className="text-[10px] text-fg-caption mb-1">ORIGINAL vs OVERRIDE</div>
        <table className="w-full text-[11px]">
          <thead className="text-fg-muted">
            <tr>
              <th className="text-left p-1">action</th>
              <th className="text-right p-1">original{original === probeOriginal && original ? ' (knn)' : ''}</th>
              <th className="text-right p-1">override</th>
            </tr>
          </thead>
          <tbody>
            {compareActions.map((a) => {
              const orig = original?.[a] ?? 0;
              const over = override?.[a] ?? 0;
              return (
                <tr key={a} className="border-b border-border-faint">
                  <td className="p-1 text-fg">{a}</td>
                  <td className="p-1 text-right text-fg-muted">{orig.toFixed(3)}</td>
                  <td className="p-1 text-right text-success">{over.toFixed(3)}</td>
                </tr>
              );
            })}
            {compareActions.length === 0 && (
              <tr>
                <td colSpan={3} className="p-1 text-fg-muted">
                  no distribution available
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Hand replay — review mode: full hand, god-view cards, no authoring */}
      {detail.hand_id && steps.length > 0 ? (
        <InlineReplayer
          steps={steps}
          rx={360}
          ry={215}
          heroPosition={heroPosition}
          gapDecisionId={detail.decision_id}
        />
      ) : (
        <div className="text-fg-muted text-[12px]">
          {detail.hand_id ? `loading replay for hand_${detail.hand_id}…` : 'no replay (patch has no decision_id)'}
        </div>
      )}

      {detail.decision_id && <GapEditHistory decisionId={detail.decision_id} />}

      <EditNodeDrawer
        open={editOpen}
        onClose={() => setEditOpen(false)}
        clusterKey={detail.cluster_key}
        onSaved={() => {
          setEditOpen(false);
          load();
        }}
      />
    </div>
  );
}
