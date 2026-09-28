import { useState, useEffect } from 'react';
import { api } from '@/lib/api';

interface AdvancedClusterKeyInputProps {
  onLoad: (key: string) => void;
}

export default function AdvancedClusterKeyInput({ onLoad }: AdvancedClusterKeyInputProps) {
  const [expanded, setExpanded] = useState(false);
  const [value, setValue] = useState('');
  const [showOpts, setShowOpts] = useState(false);
  const [hl, setHl] = useState(0);
  const [suggestions, setSuggestions] = useState<string[]>([]);

  useEffect(() => {
    api
      .get<string[]>('/api/hands/distinct?field=cluster_key')
      .then(setSuggestions)
      .catch((e) => console.error('cluster_key suggestions fetch failed', e));
  }, []);

  const matches = suggestions.filter((s) =>
    !value || s.toLowerCase().includes(value.toLowerCase())
  ).slice(0, 8);

  function submit(key: string) {
    if (!key) return;
    onLoad(key);
    setShowOpts(false);
  }

  return (
    <div className="advanced">
      <div className="advanced-head" onClick={() => setExpanded((e) => !e)}>
        <span>
          <span className="ah-chev">{expanded ? '▼' : '▶'}</span>{' '}
          advanced · paste cluster_key (power-user path)
        </span>
        <span className="dim">{expanded ? '' : 'click to expand'}</span>
      </div>
      {expanded && (
        <>
          <div className="advanced-body">
            <div className="combobox">
              <input
                placeholder="flop|SRP|IP|Axx-two-tone|x|AKs"
                value={value}
                onChange={(e) => { setValue(e.target.value); setShowOpts(true); setHl(0); }}
                onFocus={() => setShowOpts(true)}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowDown') { e.preventDefault(); setHl((h) => Math.min(h + 1, matches.length - 1)); }
                  if (e.key === 'ArrowUp') { e.preventDefault(); setHl((h) => Math.max(h - 1, 0)); }
                  if (e.key === 'Enter') { e.preventDefault(); submit(matches[hl] || value); }
                  if (e.key === 'Escape') setShowOpts(false);
                }}
              />
              {showOpts && matches.length > 0 && (
                <div className="combobox-options">
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
            <button
              className="btn"
              onClick={() => submit(value)}
              disabled={!value}
              type="button"
            >[ run query ]</button>
          </div>
          <div className="advanced-hint">
            known cluster_keys · type to filter · ↑↓ navigate · ↵ run
          </div>
        </>
      )}
    </div>
  );
}
