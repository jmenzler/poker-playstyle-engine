import SolverTruthBlock from './SolverTruthBlock';

interface EngineResponse {
  source: string | null;
  n_obs: number;
  action_dist: Record<string, number> | null;
  ev_loss: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  flagged_sparse?: boolean | null;
  max_neighbor_dist?: number | null;
  empty_reason?: string;
}

interface SolverTruth {
  action_dist: Record<string, number>;
  kl_actual_vs_solver?: number | null;
}

interface ProbeResult {
  engine_response: EngineResponse;
  solver_truth: SolverTruth | null;
  cluster_key?: string;
}

interface Props {
  clusterKey: string;
  result: ProbeResult | null;
  onRefreshResult: () => void;
  querying?: boolean;
}

export default function EngineResponsePanel({ clusterKey, result, onRefreshResult, querying }: Props) {
  if (querying) {
    return (
      <div className="section">
        <div className="section-head">
          <div className="section-head-l">
            <span className="section-head-title">ENGINE RESPONSE</span>
            <span className="section-head-meta">· querying…</span>
          </div>
          <div className="section-head-r">
            <span className="cluster-status querying">
              <span className="cluster-status-dot"></span>
              lookup
            </span>
          </div>
        </div>
        <div className="section-body">
          <div className="meta-grid">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="meta-cell">
                <div className="meta-cell-label">
                  <span className="skel skel-bar-sm skel-w-60" style={{ display: 'block' }}></span>
                </div>
                <div style={{ marginTop: 4 }}>
                  <span className="skel skel-bar skel-w-80" style={{ display: 'block' }}></span>
                </div>
              </div>
            ))}
          </div>
          <div className="cap" style={{ marginBottom: 6 }}>action distribution</div>
          {[0, 1, 2].map((i) => (
            <div key={i} className="dist-row">
              <span className="skel skel-bar skel-w-90" style={{ display: 'block' }}></span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="section">
        <div className="section-head">
          <div className="section-head-l">
            <span className="section-head-title">ENGINE RESPONSE</span>
            <span className="section-head-meta">· awaiting query</span>
          </div>
        </div>
        <div className="section-body muted">
          build a legal spot and press <span className="kbd">[ run query ]</span> · the engine response will appear here
        </div>
      </div>
    );
  }

  const eng = result.engine_response;
  const isEmpty = !!eng.empty_reason;

  return (
    <div className="section">
      <div className="section-head">
        <div className="section-head-l">
          <span className="section-head-title">ENGINE RESPONSE</span>
          <span className="section-head-meta">· source: {eng.source ?? '—'}</span>
        </div>
        <div className="section-head-r">
          <span className={'cluster-status' + (eng.flagged_sparse ? ' fresh' : '')}>
            <span className="cluster-status-dot"></span>
            {eng.flagged_sparse ? 'sparse' : 'stable'}
          </span>
        </div>
      </div>
      <div className="section-body">
        <div className="meta-grid">
          <div className="meta-cell">
            <div className="meta-cell-label">source</div>
            <div className="meta-cell-value">{eng.source ?? '—'}</div>
          </div>
          <div className="meta-cell">
            <div className="meta-cell-label">n_obs</div>
            <div className="meta-cell-value mono-num">{eng.n_obs.toLocaleString()}</div>
          </div>
          <div className="meta-cell">
            <div className="meta-cell-label">ev_loss</div>
            <div className={'meta-cell-value mono-num' + ((eng.ev_loss ?? 0) > 0.2 ? ' warn' : '')}>
              {eng.ev_loss != null ? eng.ev_loss.toFixed(3) : '—'}
              <span className="dim" style={{ fontSize: 9 }}> bb/h</span>
            </div>
          </div>
          <div className="meta-cell">
            <div className="meta-cell-label">CI 95%</div>
            <div className="meta-cell-value muted mono-num">
              {eng.ci_high != null ? `±${eng.ci_high.toFixed(3)}` : '—'}
            </div>
          </div>
        </div>

        {eng.action_dist && (
          <>
            <div className="cap" style={{ marginBottom: 6 }}>action distribution</div>
            {Object.entries(eng.action_dist).map(([action, freq]) => {
              const deltaClass = freq >= 0.5 ? 'majority' : freq >= 0.25 ? 'mixed' : 'minor';
              return (
                <div key={action} className="dist-row">
                  <span className="dist-row-act">{action}</span>
                  <span className="dist-row-freq">{(freq * 100).toFixed(1)}%</span>
                  <div className="dist-row-bar">
                    <div className="dist-row-bar-fill" style={{ width: `${freq * 100}%` }}></div>
                  </div>
                  <span className="dist-row-delta">{deltaClass}</span>
                </div>
              );
            })}
          </>
        )}

        {isEmpty && (
          <div className="empty-banner">
            <span className="empty-banner-tag">empty</span>
            <span className="empty-banner-body">
              <b>no strategy node for this spot</b> — {eng.empty_reason}{' '}
              <span className="dim">(informational · not an error)</span>
            </span>
          </div>
        )}
      </div>

      <SolverTruthBlock
        clusterKey={clusterKey}
        engineDist={eng.action_dist}
        solverTruth={result.solver_truth}
        onRefresh={onRefreshResult}
      />
    </div>
  );
}
