interface Props {
  clusterKey: string | null;
  onEdit?: () => void;
}

export default function ActionsPanel({ clusterKey, onEdit }: Props) {
  return (
    <div className="section">
      <div className="section-head">
        <div className="section-head-l">
          <span className="section-head-title">ACTIONS</span>
          {clusterKey && <span className="section-head-meta">· operator-level</span>}
        </div>
      </div>
      <div className="section-body">
        <div className="row" style={{ flexWrap: 'wrap' }}>
          <button
            className="btn"
            onClick={onEdit}
            disabled={!clusterKey}
            type="button"
          >[ edit this cluster ]</button>
          <button className="btn" disabled={!clusterKey} type="button">[ inspect history ]</button>
          <button className="btn" disabled={!clusterKey} type="button">[ enqueue rerun ]</button>
          <span className="spacer"></span>
          <button className="btn btn-destructive" disabled={!clusterKey} type="button">
            [ flag as leak ]
          </button>
        </div>
        <div className="dim" style={{ marginTop: 8, fontSize: 10 }}>
          flagging adds a suppression row and excludes this cluster from autoloop training
        </div>
      </div>
    </div>
  );
}
