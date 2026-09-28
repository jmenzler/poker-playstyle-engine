import { Fragment } from 'react';

interface ActionSpec {
  action_dist: Record<string, number> | null;
  cluster_key: string;
}

interface Props {
  specA: ActionSpec;
  specB: ActionSpec;
  onDismiss: () => void;
}

export default function ABComparePanel({ specA, specB, onDismiss }: Props) {
  const distA = specA.action_dist ?? {};
  const distB = specB.action_dist ?? {};

  if (!specA.action_dist || !specB.action_dist) return null;

  const keys: string[] = [];
  for (const k of Object.keys(distA)) keys.push(k);
  for (const k of Object.keys(distB)) if (!keys.includes(k)) keys.push(k);

  return (
    <div className="section">
      <div className="section-head">
        <div className="section-head-l">
          <span className="section-head-title">A vs B · ENGINE COMPARE</span>
        </div>
        <div className="section-head-r">
          <button className="btn btn-small btn-ghost" onClick={onDismiss} type="button">[ × dismiss B ]</button>
        </div>
      </div>
      <div className="section-body">
        <div className="ab-keys">
          <div><span className="cap">A</span> <span style={{ color: 'var(--color-accent)', fontSize: 11 }}>{specA.cluster_key}</span></div>
          <div><span className="cap">B</span> <span style={{ color: 'var(--color-success)', fontSize: 11 }}>{specB.cluster_key}</span></div>
        </div>
        <div className="ab-grid">
          <span className="cap">action</span>
          <span className="cap" style={{ textAlign: 'right' }}>A</span>
          <span className="cap">A bar</span>
          <span className="cap" style={{ textAlign: 'right' }}>B</span>
          <span className="cap">B bar</span>
          <span className="cap" style={{ textAlign: 'right' }}>Δ</span>
          {keys.map((a) => {
            const fa = distA[a] ?? 0;
            const fb = distB[a] ?? 0;
            const delta = +(fb - fa).toFixed(3);
            const deltaClass = delta > 0.02 ? 'up' : delta < -0.02 ? 'down' : '';
            return (
              <Fragment key={a}>
                <span className="ab-act">{a}</span>
                <span className="ab-pct">{(fa * 100).toFixed(1)}%</span>
                <div className="dist-row-bar">
                  <div className="dist-row-bar-fill" style={{ width: `${fa * 100}%` }}></div>
                </div>
                <span className="ab-pct">{(fb * 100).toFixed(1)}%</span>
                <div className="dist-row-bar">
                  <div className="dist-row-bar-fill" style={{ width: `${fb * 100}%`, background: 'var(--color-success)' }}></div>
                </div>
                <span className={'ab-delta ' + deltaClass}>
                  {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}pp
                </span>
              </Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );
}
