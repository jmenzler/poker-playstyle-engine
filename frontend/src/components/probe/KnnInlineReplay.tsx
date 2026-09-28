import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import InlineReplayer, { type ReplayStep } from '@/components/InlineReplayer';

interface KnnNeighbor {
  cluster_key: string;
  decision_id: string | null;
  distance: number;
  action_dist: Record<string, number> | null;
  obs_id: string | null;
  hand_id: string | null;
  n_obs: number;
}

interface Props {
  neighbor: KnnNeighbor;
}

type LoadState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'done'; steps: ReplayStep[] }
  | { status: 'error'; message: string };

export default function KnnInlineReplay({ neighbor }: Props) {
  const [load, setLoad] = useState<LoadState>({ status: 'idle' });

  useEffect(() => {
    if (!neighbor.obs_id && !neighbor.hand_id) return;
    const path = neighbor.hand_id
      ? `/api/hands/by-hand/${encodeURIComponent(neighbor.hand_id)}/replay`
      : `/api/hands/${encodeURIComponent(neighbor.obs_id!)}/replay`;
    let cancelled = false;
    Promise.resolve()
      .then(() => { if (!cancelled) setLoad({ status: 'loading' }); })
      .then(() => api.get<ReplayStep[]>(path))
      .then((steps) => { if (!cancelled) setLoad({ status: 'done', steps }); })
      .catch((e: unknown) => {
        if (!cancelled) setLoad({ status: 'error', message: e instanceof Error ? e.message : String(e) });
      });
    return () => { cancelled = true; };
  }, [neighbor.hand_id, neighbor.obs_id]);

  const label = neighbor.hand_id ?? neighbor.obs_id ?? '—';

  return (
    <div className="knn-replay">
      <div className="knn-replay-head">
        <span className="cap">representative replay</span>
        <span className="dim knn-replay-meta">
          hand_id {label} · 1 of {neighbor.n_obs.toLocaleString()}
        </span>
      </div>
      {!neighbor.obs_id && !neighbor.hand_id && (
        <div className="knn-replay-none">no representative observation</div>
      )}
      {load.status === 'loading' && <div className="knn-replay-loading">loading replay…</div>}
      {load.status === 'error' && <div className="knn-replay-err">error: {load.message}</div>}
      {load.status === 'done' && <InlineReplayer steps={load.steps} rx={220} ry={130} />}
    </div>
  );
}
