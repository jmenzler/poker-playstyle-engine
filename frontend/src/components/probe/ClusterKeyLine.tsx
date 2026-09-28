import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '@/lib/api';

type PillState = 'fresh' | 'querying' | 'queried' | 'invalid';

interface ClusterKeyLineProps {
  previewKey: string | null;
  valid: boolean;
  querying: boolean;
  queried: boolean;
  structural: Array<{ field: string; msg: string }>;
  onLoad: (key: string) => void;
  onLoadB?: (key: string) => void;
}

export default function ClusterKeyLine({
  previewKey,
  valid,
  querying,
  queried,
  structural,
  onLoad,
  onLoadB,
}: ClusterKeyLineProps) {
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState('');
  const [showOpts, setShowOpts] = useState(false);
  const [hl, setHl] = useState(0);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api
      .get<string[]>('/api/hands/distinct?field=cluster_key')
      .then(setSuggestions)
      .catch((e) => console.error('cluster_key suggestions fetch failed', e));
  }, []);

  const tokens = previewKey ? previewKey.split('|') : null;

  let pillState: PillState;
  if (querying) pillState = 'querying';
  else if (editing) pillState = 'fresh';
  else if (!valid) pillState = 'invalid';
  else if (queried) pillState = 'queried';
  else pillState = 'fresh';

  let pillText: string;
  if (querying) pillText = 'querying…';
  else if (editing) pillText = 'edit mode';
  else if (!valid) pillText = 'invalid spot';
  else if (queried) pillText = 'queried';
  else pillText = 'fresh · not run';

  const matches = suggestions.filter((s) =>
    !editValue || s.toLowerCase().includes(editValue.toLowerCase())
  ).slice(0, 8);

  const handleEditValueChange = useCallback((val: string) => {
    setEditValue(val);
    setHl(0);
  }, []);

  function startEdit() {
    setEditValue(previewKey || '');
    setEditing(true);
    setShowOpts(true);
    setTimeout(() => inputRef.current?.focus(), 0);
  }

  function cancel() { setEditing(false); setShowOpts(false); }

  function submit(key: string) {
    if (!key) { cancel(); return; }
    onLoad(key);
    cancel();
  }

  const tooltipText = !valid && !querying && structural.length > 0
    ? structural.map((e) => `• ${e.msg}`).join('\n')
    : undefined;

  if (editing) {
    return (
      <div className="cluster-line cluster-line-editing">
        <span className="cluster-line-label">cluster_key</span>
        <div className="cluster-line-edit-wrap">
          <input
            ref={inputRef}
            className="cluster-line-input"
            value={editValue}
            onChange={(e) => { handleEditValueChange(e.target.value); setShowOpts(true); }}
            onFocus={() => setShowOpts(true)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') { e.preventDefault(); setHl((h) => Math.min(h + 1, matches.length - 1)); }
              if (e.key === 'ArrowUp') { e.preventDefault(); setHl((h) => Math.max(h - 1, 0)); }
              if (e.key === 'Enter') { e.preventDefault(); submit(matches[hl] || editValue); }
              if (e.key === 'Escape') { e.preventDefault(); cancel(); }
            }}
            placeholder="flop|SRP|IP|Axx-two-tone|x|AKs"
          />
          {showOpts && matches.length > 0 && (
            <div className="combobox-options cluster-line-options">
              {matches.map((m, i) => (
                <div
                  key={m}
                  className={'combobox-opt' + (i === hl ? ' hl' : '')}
                  onMouseDown={(e) => { e.preventDefault(); submit(m); }}
                >
                  <span>{m}</span>
                </div>
              ))}
            </div>
          )}
        </div>
        <span className="dim" style={{ fontSize: 10, whiteSpace: 'nowrap' }}>
          <span className="kbd">↵</span> load · <span className="kbd">esc</span> cancel
          {onLoadB && (
            <>
              {' · '}
              <button
                className="cluster-line-edit-btn"
                style={{ marginLeft: 4 }}
                onMouseDown={(e) => { e.preventDefault(); if (editValue) { onLoadB(editValue); cancel(); } }}
                disabled={!editValue}
                title="load this cluster_key as spot B for side-by-side comparison"
                type="button"
              >load as B ▸</button>
            </>
          )}
        </span>
        <span className={'cluster-line-status ' + pillState}>
          <span className="cluster-status-dot"></span>
          {pillText}
        </span>
      </div>
    );
  }

  return (
    <div className="cluster-line">
      <span className="cluster-line-label">cluster_key</span>
      <span className={'cluster-line-value' + (tokens ? '' : ' placeholder')}>
        {tokens
          ? tokens.map((tok, i) => (
              <span key={i}>
                {i > 0 && <span className="ck-pipe">|</span>}
                <span className="ck-tok">{tok}</span>
              </span>
            ))
          : <>— build spot to encode —</>}
      </span>
      <button
        className="cluster-line-edit-btn"
        onClick={startEdit}
        title="paste / look up a cluster_key directly"
        type="button"
      >[ edit ▾ ]</button>
      <span
        className={
          'cluster-line-status ' + pillState +
          (!valid && !querying && structural.length > 0 ? ' has-tip' : '')
        }
        title={tooltipText}
      >
        <span className="cluster-status-dot"></span>
        {pillText}
        {!valid && !querying && structural.length > 0 && (
          <span className="cluster-line-status-tip">
            {structural.map((e, i) => (
              <span key={i} className="cluster-line-status-tip-row">• {e.msg}</span>
            ))}
          </span>
        )}
      </span>
    </div>
  );
}
