import { Fragment, useState } from 'react';
import KnnInlineReplay from './KnnInlineReplay';

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
  neighbors: KnnNeighbor[];
  onQuery: (clusterKey: string) => void;
  querying?: boolean;
}

export default function KnnPanel({ neighbors, onQuery, querying }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [open, setOpen] = useState(true);

  if (querying) {
    return (
      <div className="section">
        <div className="section-head">
          <div className="section-head-l">
            <span className="section-head-title">kNN NEIGHBORS</span>
            <span className="section-head-meta">· searching milvus…</span>
          </div>
          <div className="section-head-r">
            <span className="chev">▼</span>
          </div>
        </div>
        <div className="section-body">
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '1fr 80px 60px 60px', gap: 10, padding: '4px 0' }}>
              <span className="skel skel-bar skel-w-90" style={{ display: 'block' }}></span>
              <span className="skel skel-bar skel-w-70" style={{ display: 'block' }}></span>
              <span className="skel skel-bar skel-w-60" style={{ display: 'block' }}></span>
              <span className="skel skel-bar skel-w-40" style={{ display: 'block' }}></span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!neighbors || neighbors.length === 0) {
    return (
      <div className="section">
        <div className="section-head">
          <div className="section-head-l">
            <span className="section-head-title">kNN NEIGHBORS</span>
            <span className="section-head-meta">· none</span>
          </div>
        </div>
        <div className="section-body muted" style={{ fontSize: 11 }}>
          no neighbors returned for this spot — the decision index has no resolvable matches yet
        </div>
      </div>
    );
  }

  return (
    <div className={'section' + (!open ? ' collapsed' : '')}>
      <div className="section-head">
        <div className="section-head-l">
          <span className="section-head-title">kNN NEIGHBORS</span>
          <span className="section-head-meta">· top {neighbors.length}</span>
        </div>
        <div className="section-head-r">
          <span className="chev" onClick={() => setOpen((o) => !o)} title="toggle">
            {open ? '▼' : '▶'}
          </span>
        </div>
      </div>
      <div className="section-body">
        <table className="knn-table">
          <colgroup>
            <col className="knn-col-key" />
            <col className="knn-col-dist" />
            <col className="knn-col-nobs" />
            <col className="knn-col-act" />
          </colgroup>
          <thead>
            <tr>
              <th>hand_id</th>
              <th style={{ textAlign: 'right' }}>distance</th>
              <th style={{ textAlign: 'right' }}>n_obs</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {neighbors.map((n, idx) => {
              // Hard-filtered neighbors share one cluster_key, so the row/expand
              // identity must be the per-hit decision_id (fall back to index).
              const rowId = n.decision_id ?? String(idx);
              const isOpen = expanded === rowId;
              const canReplay = !!(n.hand_id || n.obs_id);
              return (
                <Fragment key={rowId}>
                  <tr>
                    <td className="knn-key">{n.hand_id ?? n.cluster_key}</td>
                    <td className="knn-dist">{n.distance.toFixed(4)}</td>
                    <td className="knn-nobs">{n.n_obs.toLocaleString()}</td>
                    <td className="knn-pick">
                      <div className="knn-pick-row">
                        <button
                          className="btn btn-small btn-ghost"
                          disabled={!canReplay}
                          title={canReplay ? 'inline replay of this hand' : 'no replayable hand'}
                          onClick={() => setExpanded(isOpen ? null : rowId)}
                          type="button"
                        >
                          {isOpen ? '▼ hide' : '▶ replay'}
                        </button>
                        <button
                          className="btn btn-small btn-ghost"
                          onClick={() => onQuery(n.cluster_key)}
                          title="re-run engine query for this neighbor"
                          type="button"
                        >
                          [ query ▸ ]
                        </button>
                      </div>
                    </td>
                  </tr>
                  {isOpen && (
                    <tr>
                      <td colSpan={4} style={{ padding: '8px 0' }}>
                        <KnnInlineReplay neighbor={n} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
