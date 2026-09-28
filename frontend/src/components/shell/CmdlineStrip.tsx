import { useState } from 'react';

// 24px bottom cmdline strip per UI-SPEC v2.2 §Bottom cmdline strip.
// Vim-prompt leading colon + filter/sort/view chips + ⌘K palette chip +
// READY/BUSY status indicator on the right edge.

export default function CmdlineStrip() {
  // TODO Plan 09: wire to JobRegistry inflight count via /api/health.n_jobs
  const [busy] = useState(false);

  return (
    <footer
      className="bg-surface flex items-center px-4 text-[12px] border-t border-border"
      style={{ height: 24 }}
    >
      <span className="text-fg-muted">: filter all</span>
      <span className="ml-auto flex gap-4">
        <span className="text-fg-muted">filter:all</span>
        <span className="text-fg-muted">sort:time</span>
        <span className="text-fg-muted">view:dense</span>
        <span className="text-accent">⌘K palette</span>
        <span className={busy ? 'text-warning' : 'text-success'}>
          {'>_ '}
          {busy ? 'BUSY' : 'READY'}
        </span>
      </span>
    </footer>
  );
}
