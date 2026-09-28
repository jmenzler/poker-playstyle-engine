import { useState } from 'react';
import { truncateClusterKey } from '@/lib/format';

interface GapActionButtonsProps {
  legalActions: string[];
  isMultiway: boolean;
  fastMode: boolean;
  prevNodeId?: string | null;
  clusterKey?: string;
  resolving?: boolean;
  error?: string | null;
  onResolveGap: (actionDist: Record<string, number>) => void;
  onSendToSolver?: () => void;
}

export default function GapActionButtons({
  legalActions,
  isMultiway: _isMultiway,
  fastMode,
  prevNodeId,
  clusterKey,
  resolving = false,
  error = null,
  onResolveGap,
}: GapActionButtonsProps) {
  const [selected, setSelected] = useState<string | null>(null);
  const [splitOpen, setSplitOpen] = useState(false);
  const [splitPcts, setSplitPcts] = useState<Record<string, number>>({});
  const [showPreview, setShowPreview] = useState(false);

  function initSplitPcts(action: string): Record<string, number> {
    const initial: Record<string, number> = {};
    for (const a of legalActions) {
      initial[a] = a === action ? 100 : 0;
    }
    return initial;
  }

  function handleActionClick(action: string) {
    if (resolving) return;
    setSelected(action);
    setSplitOpen(false);
    setSplitPcts(initSplitPcts(action));
    if (fastMode) {
      onResolveGap({ [action]: 1.0 });
    } else {
      setShowPreview(true);
    }
  }

  function handleSave() {
    if (resolving || !selected) return;
    let dist: Record<string, number>;
    if (splitOpen) {
      const total = Object.values(splitPcts).reduce((a, b) => a + b, 0);
      if (total !== 100) return;
      dist = Object.fromEntries(
        Object.entries(splitPcts)
          .filter(([, pct]) => pct > 0)
          .map(([a, pct]) => [a, pct / 100]),
      );
    } else {
      dist = { [selected]: 1.0 };
    }
    onResolveGap(dist);
  }

  function handleCancel() {
    setSelected(null);
    setSplitOpen(false);
    setSplitPcts({});
    setShowPreview(false);
  }

  const splitTotal = Object.values(splitPcts).reduce((a, b) => a + b, 0);
  const splitValid = splitTotal === 100;

  const previewDist: Record<string, number> = splitOpen
    ? Object.fromEntries(
        Object.entries(splitPcts)
          .filter(([, pct]) => pct > 0)
          .map(([a, pct]) => [a, pct / 100]),
      )
    : selected
    ? { [selected]: 1.0 }
    : {};

  return (
    <div className="flex flex-col gap-3 font-mono">
      {prevNodeId && (
        <div className="text-[10px] text-warning">
          previously authored — saving will supersede
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        {legalActions.map((action) => {
          const isSelected = selected === action && !splitOpen;
          return (
            <button
              key={action}
              aria-label={`${action} 100%`}
              disabled={resolving}
              onClick={() => handleActionClick(action)}
              className={`border border-border px-3 py-1 text-[11px] ${
                isSelected
                  ? 'bg-[#d4d4d4] text-[#000000]'
                  : 'text-fg hover:bg-surface-2'
              }`}
            >
              {action}
            </button>
          );
        })}
      </div>

      {resolving && (
        <div className="text-[11px] text-fg-muted">saving…</div>
      )}

      {error && !resolving && (
        <div className="text-[12px] text-destructive">save failed: {error}</div>
      )}

      {!fastMode && selected && showPreview && !resolving && (
        <>
          <button
            onClick={() => setSplitOpen((o) => !o)}
            className="text-[11px] text-fg w-fit"
          >
            [ {splitOpen ? '- split' : '+ split'} ]
          </button>

          {splitOpen && (
            <div className="flex items-center gap-2 flex-wrap">
              {legalActions.map((action) => (
                <div key={action} className="flex items-center gap-1">
                  <span className="text-[11px] text-fg">{action}:</span>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={1}
                    value={splitPcts[action] ?? 0}
                    onChange={(e) => {
                      const val = parseInt(e.target.value, 10) || 0;
                      setSplitPcts((prev) => ({ ...prev, [action]: val }));
                    }}
                    className="border border-border bg-transparent px-1 text-[12px] w-14"
                  />
                  <span className="text-[11px] text-fg-muted">%</span>
                </div>
              ))}
              <span
                className={`text-[11px] ${splitValid ? 'text-fg-muted' : 'text-destructive'}`}
              >
                total: {splitTotal}%
              </span>
            </div>
          )}

          <div className="flex flex-col gap-1">
            <div className="text-[10px] text-fg-caption">PREVIEW</div>
            {Object.entries(previewDist).map(([action, freq]) => (
              <div key={action} className="text-[12px] text-fg">
                {action}: {Math.round(freq * 100)}%
              </div>
            ))}
            {clusterKey && (
              <div className="text-[12px] text-fg-muted">
                cluster_key: {truncateClusterKey(clusterKey)}
              </div>
            )}
            <div className="flex gap-2 mt-1">
              <button
                onClick={handleSave}
                disabled={resolving || (splitOpen && !splitValid)}
                className="border border-border px-2 py-0.5 text-[11px] text-fg"
              >
                [ save ]
              </button>
              <button
                onClick={handleCancel}
                disabled={resolving}
                className="border border-border px-2 py-0.5 text-[11px] text-fg"
              >
                [ cancel ]
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
