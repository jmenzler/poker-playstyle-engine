import { useEffect, useRef, useState } from 'react';
import { api, subscribeJob } from '@/lib/api';

interface SolverTruth {
  action_dist: Record<string, number>;
  kl_actual_vs_solver?: number | null;
}

interface Props {
  clusterKey: string;
  engineDist: Record<string, number> | null;
  solverTruth: SolverTruth | null;
  onRefresh: () => void;
}

type QueueState = { etaSec: number | null } | null;

export default function SolverTruthBlock({ clusterKey, engineDist, solverTruth, onRefresh }: Props) {
  const [queue, setQueue] = useState<QueueState>(null);
  const [err, setErr] = useState<string | null>(null);
  const unsubRef = useRef<(() => void) | null>(null);

  // Tear down any in-flight verify stream when the cluster changes (not just on
  // unmount): a stale subscription would fire onRefresh for the wrong cluster.
  useEffect(() => () => {
    unsubRef.current?.();
    unsubRef.current = null;
    setQueue(null);
    setErr(null);
  }, [clusterKey]);

  function enqueue() {
    setErr(null);
    setQueue({ etaSec: null });
    api
      .post<{ job_id: string; status: string }>('/api/verify', { cluster_key: clusterKey, force: false })
      .then(({ job_id }) => {
        unsubRef.current = subscribeJob(job_id, '/api/verify/sse', {
          onProgress: (data) => {
            const d = data as { eta_sec?: number };
            setQueue({ etaSec: d.eta_sec ?? null });
          },
          onDone: () => {
            unsubRef.current = null;
            setQueue(null);
            onRefresh();
          },
          onError: (msg) => {
            unsubRef.current = null;
            setQueue(null);
            setErr(msg);
          },
        });
      })
      .catch((e: unknown) => {
        setQueue(null);
        setErr(e instanceof Error ? e.message : String(e));
      });
  }

  return (
    <div className="solver-truth-block">
      <div className="solver-truth-head">
        <span className="cap">solver truth</span>
        {solverTruth && (
          <span className="dim solver-truth-kl">
            · KL = {solverTruth.kl_actual_vs_solver?.toFixed(4) ?? '—'}
          </span>
        )}
        {queue && !solverTruth && (
          <span className="solver-truth-queued">
            · queued{queue.etaSec != null ? ` · eta ~${queue.etaSec}s` : ''}
          </span>
        )}
        {!solverTruth && !queue && <span className="dim">· no cached result</span>}
        {!solverTruth && !queue && (
          <button className="btn btn-small solver-truth-btn" onClick={enqueue}>
            [ solve spot ▸ ]
          </button>
        )}
      </div>
      {err && <div className="solver-truth-err">error: {err}</div>}
      {solverTruth &&
        Object.entries(solverTruth.action_dist).map(([action, freq]) => {
          const eng = engineDist?.[action] ?? 0;
          const dpp = (freq - eng) * 100;
          const deltaClass = dpp > 2 ? 'up' : dpp < -2 ? 'down' : '';
          return (
            <div key={action} className="dist-row">
              <span className="dist-row-act solver">{action}</span>
              <span className="dist-row-freq">{(freq * 100).toFixed(1)}%</span>
              <div className="dist-row-bar">
                <div className="dist-row-bar-fill solver" style={{ width: `${freq * 100}%` }}></div>
              </div>
              <span className={`dist-row-delta ${deltaClass}`}>
                {dpp > 0 ? '+' : ''}{dpp.toFixed(1)}pp
              </span>
            </div>
          );
        })}
      {queue && !solverTruth && (
        <div className="queue-indicator">
          <span className="qind-tag">computing</span>
          <span>solver verification running</span>
          {queue.etaSec != null && <span className="qind-eta">eta ~{queue.etaSec}s</span>}
        </div>
      )}
    </div>
  );
}
