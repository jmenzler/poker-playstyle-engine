import { NavLink } from 'react-router';
import { useEffect, useState } from 'react';
import { api } from '@/lib/api';

// 7-row sidebar per UI-SPEC v2.2 §Sidebar (D-NEW-23 + D-NEW-30).
// Order: dashboard, probe, strategy, coverage, hands, patches, eval.
// Count badges show on strategy / coverage / patches / eval.
// Active row inverts colours (bg #d4d4d4, fg #000) per UI-SPEC.

interface NavRow {
  num: string;
  path: string;
  label: string;
  countKey?: 'strategy' | 'coverage' | 'patches' | 'eval';
}

const NAV: NavRow[] = [
  { num: '1', path: '/dashboard', label: 'dashboard' },
  { num: '2', path: '/probe', label: 'probe' },
  { num: '3', path: '/strategy', label: 'strategy', countKey: 'strategy' },
  { num: '4', path: '/coverage', label: 'coverage', countKey: 'coverage' },
  { num: '5', path: '/hands', label: 'hands' },
  { num: '6', path: '/patches', label: 'patches', countKey: 'patches' },
  { num: '7', path: '/eval', label: 'eval', countKey: 'eval' },
  { num: '8', path: '/decisions', label: 'decisions' },
  { num: '9', path: '/gap-resolver', label: 'gap-resolver' },
];

interface HealthShape {
  counts?: Partial<Record<NavRow['countKey'] & string, number>>;
}

export default function Sidebar() {
  const [counts, setCounts] = useState<Record<string, number>>({});

  useEffect(() => {
    // D-NEW-21 #5: poll sidebar count badges every 30s. /api/health is the
    // lightest aggregate; if it doesn't include counts (current Plan 06-07
    // returns {status, n_jobs}), we render zero badges silently.
    let stop = false;
    async function tick() {
      try {
        const data = await api.get<HealthShape>('/api/health');
        if (!stop) setCounts((data.counts ?? {}) as Record<string, number>);
      } catch {
        // backend unreachable — leave badges empty
      }
    }
    tick();
    const id = setInterval(tick, 30000);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, []);

  return (
    <aside
      className="bg-surface flex flex-col border-r border-border"
      style={{ width: 240 }}
    >
      {/* App-brand row */}
      <div className="px-4 py-3 border-b border-border">
        <div className="text-[14px] font-bold text-fg">{'>_ tauri'}</div>
        <div className="text-[10px] text-fg-caption">v0.2.0 · phase 6</div>
      </div>

      {/* Nav block */}
      <nav className="flex-1 px-2 py-2 space-y-0.5">
        {NAV.map(({ num, path, label, countKey }) => {
          const count = countKey ? counts[countKey] ?? 0 : 0;
          return (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) =>
                `flex items-center justify-between px-3 h-[32px] text-[13px] ${
                  isActive ? 'bg-[#d4d4d4] text-[#000000]' : 'text-fg hover:bg-surface-2'
                }`
              }
            >
              <span>
                {num} {label}
              </span>
              {countKey && count > 0 && (
                <span className="text-[10px] text-fg-muted">[{count}]</span>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* KEY legend block */}
      <div className="px-4 py-3 border-t border-border">
        <div className="text-[10px] text-fg-caption mb-2">KEY</div>
        <div className="space-y-1 text-[11px]">
          <div>
            <span className="text-success">■</span> VERIFIED
          </div>
          <div>
            <span className="text-warning">■</span> VERIFYING
          </div>
          <div>
            <span className="text-fg-muted">□</span> PENDING
          </div>
          <div>
            <span className="text-destructive">■</span> REJECTED
          </div>
        </div>
      </div>

      {/* FASTAPI footer */}
      <div className="px-4 py-2 border-t border-border text-[10px]">
        <div className="text-fg-muted">FASTAPI · 12ms p50</div>
        <div className="h-[2px] mt-1 bg-success" style={{ width: '40%' }} />
      </div>
    </aside>
  );
}
