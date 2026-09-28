import { useEffect, useRef, useState } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';
import { api } from '@/lib/api';
import GapPanel from '@/components/GapPanel';

// Dashboard panel per UI-SPEC v2.2 + D-NEW-27 (verdict-led vertical stack).
// Sections (top to bottom):
//   1. LOOP HEALTH verdict bar
//   2. EV_LOSS TREND hero chart (full-width line, recharts)
//   3. RECENT ACTIVITY + SUBSYSTEM HEALTH (2-col grid)
//   4. SESSION + KB GROWTH (2-col grid)

interface LoopHealth {
  verdict: 'improving' | 'stalled' | 'mixed' | string;
  narrative: string;
}

interface EvLossPoint {
  session_id: string;
  ts: string;
  ev_loss: number;
}

interface RecentActivity {
  ts: string;
  event_type: string;
  summary: string;
}

interface SessionInfo {
  session_id: string;
  n_obs: number;
  start_ts: string;
}

interface KBGrowth {
  total_nodes: number;
  by_source: Record<string, number>;
  last_7d_delta: Record<string, number>;
}

interface DashboardData {
  loop_health: LoopHealth;
  ev_loss_trend: EvLossPoint[];
  recent_activity: RecentActivity[];
  health: Record<string, string>;
  session: SessionInfo | null;
  kb_growth: KBGrowth;
}

interface AutoloopStatus {
  run_id: string;
  state: 'running' | 'stopped' | 'stopping' | 'idle' | string;
  cycles_completed: number;
  patches_accepted: number;
  patches_rejected: number;
  ev_loss_trend: number[];
}

const verdictColors: Record<string, string> = {
  improving: 'text-success',
  stalled: 'text-destructive',
  mixed: 'text-warning',
};

const verdictGlyphs: Record<string, string> = {
  improving: '✓',
  stalled: '✗',
  mixed: '~',
};

export default function Dashboard() {
  const [d, setD] = useState<DashboardData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [alRunId, setAlRunId] = useState(() => {
    const saved = localStorage.getItem('autoloop_run_id');
    return saved ?? `dash-${Date.now()}`;
  });
  const [alBaseSeed, setAlBaseSeed] = useState(0);
  const [alMaxCycles, setAlMaxCycles] = useState<string>('');
  const [alStatus, setAlStatus] = useState<AutoloopStatus | null>(null);
  const [alErr, setAlErr] = useState<string | null>(null);
  const alIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    api
      .get<DashboardData>('/api/dashboard')
      .then(setD)
      .catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (alIntervalRef.current) clearInterval(alIntervalRef.current);
    const poll = () => {
      api
        .get<AutoloopStatus>(`/api/autoloop/status?run_id=${encodeURIComponent(alRunId)}`)
        .then(setAlStatus)
        .catch((e) => setAlErr(String(e)));
    };
    poll();
    alIntervalRef.current = setInterval(poll, 7000);
    return () => {
      if (alIntervalRef.current) clearInterval(alIntervalRef.current);
    };
  }, [alRunId]);

  // Persist run_id so leaving the dashboard route and returning keeps tracking
  // the active run instead of minting a fresh, untracked id (which polls a
  // nonexistent run and shows it as idle/stopped).
  useEffect(() => {
    localStorage.setItem('autoloop_run_id', alRunId);
  }, [alRunId]);

  function alStart() {
    setAlErr(null);
    const body: Record<string, unknown> = { run_id: alRunId, base_seed: alBaseSeed };
    if (alMaxCycles !== '') body.max_cycles = parseInt(alMaxCycles, 10);
    api
      .post<AutoloopStatus>('/api/autoloop/run', body)
      .then(setAlStatus)
      .catch((e) => setAlErr(String(e)));
  }

  function alStop() {
    setAlErr(null);
    api
      .post<AutoloopStatus>('/api/autoloop/stop', { run_id: alRunId })
      .then(setAlStatus)
      .catch((e) => setAlErr(String(e)));
  }

  function alResume() {
    setAlErr(null);
    api
      .post<AutoloopStatus>('/api/autoloop/resume', { run_id: alRunId })
      .then(setAlStatus)
      .catch((e) => setAlErr(String(e)));
  }

  if (err) return <div className="text-destructive text-[12px]">ERROR: {err}</div>;
  if (!d) return <div className="text-fg-muted text-[12px]">loading…</div>;

  const verdictColor = verdictColors[d.loop_health.verdict] ?? 'text-fg';
  const verdictGlyph = verdictGlyphs[d.loop_health.verdict] ?? '?';

  const alState = alStatus?.state ?? 'idle';

  return (
    <div className="space-y-6">
      <h1 className="text-[14px] font-bold">:: DASHBOARD :: control panel</h1>

      {/* AUTOLOOP CONTROL */}
      <section className="border border-border p-4">
        <div className="text-[10px] text-fg-caption mb-2">AUTOLOOP CONTROL</div>
        <div className="space-y-2 text-[12px]">
          <div className="flex gap-2 flex-wrap items-center">
            <label className="text-fg-muted">run_id</label>
            <input
              className="border border-border bg-transparent px-1 text-[12px] w-40"
              value={alRunId}
              onChange={(e) => setAlRunId(e.target.value)}
            />
            <label className="text-fg-muted">seed</label>
            <input
              className="border border-border bg-transparent px-1 text-[12px] w-16"
              type="number"
              value={alBaseSeed}
              onChange={(e) => setAlBaseSeed(parseInt(e.target.value, 10) || 0)}
            />
            <label className="text-fg-muted">max_cycles</label>
            <input
              className="border border-border bg-transparent px-1 text-[12px] w-16"
              type="number"
              value={alMaxCycles}
              placeholder="∞"
              onChange={(e) => setAlMaxCycles(e.target.value)}
            />
          </div>
          <div className="flex gap-2">
            <button
              className="border border-border px-2 py-0.5 text-[11px] disabled:opacity-40"
              onClick={alStart}
              disabled={alState === 'running'}
            >
              START
            </button>
            <button
              className="border border-border px-2 py-0.5 text-[11px] disabled:opacity-40"
              onClick={alStop}
              disabled={alState !== 'running'}
            >
              STOP
            </button>
            <button
              className="border border-border px-2 py-0.5 text-[11px] disabled:opacity-40"
              onClick={alResume}
              disabled={alState === 'running'}
            >
              RESUME
            </button>
          </div>
          {alErr && <div className="text-destructive text-[11px]">{alErr}</div>}
          {alStatus && (
            <div className="space-y-0.5 text-fg-muted">
              <div>
                state: <span className="text-fg">{alStatus.state}</span>
              </div>
              <div>cycles: {alStatus.cycles_completed}</div>
              <div>
                patches accepted: {alStatus.patches_accepted} / rejected:{' '}
                {alStatus.patches_rejected}
              </div>
              {alStatus.ev_loss_trend.length > 0 && (
                <div>
                  ev_loss trend:{' '}
                  {alStatus.ev_loss_trend
                    .slice()
                    .reverse()
                    .map((v) => v.toFixed(3))
                    .join(' → ')}
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      {/* LOOP HEALTH verdict bar */}
      <section className="border border-border p-4">
        <div className="text-[10px] text-fg-caption mb-1">LOOP HEALTH</div>
        <div className={`text-[14px] ${verdictColor}`}>
          {verdictGlyph} {d.loop_health.verdict}
        </div>
        <div className="text-[12px] text-fg-muted mt-1">{d.loop_health.narrative}</div>
      </section>

      {/* EV_LOSS TREND hero — Pitfall 9: explicit height to avoid 0-height SVG */}
      <section className="border border-border p-4">
        <div className="text-[10px] text-fg-caption mb-2">
          EV_LOSS TREND · last {d.ev_loss_trend.length} sessions
        </div>
        <div style={{ width: '100%', height: 240 }}>
          <ResponsiveContainer>
            <LineChart data={d.ev_loss_trend}>
              <XAxis dataKey="session_id" tick={{ fontSize: 10, fill: '#888' }} />
              <YAxis tick={{ fontSize: 10, fill: '#888' }} />
              <Tooltip
                contentStyle={{
                  background: '#0a0a0a',
                  border: '1px solid #262626',
                  color: '#d4d4d4',
                }}
              />
              <Line
                type="monotone"
                dataKey="ev_loss"
                stroke="#3b82f6"
                dot={false}
                strokeWidth={1.5}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* RECENT ACTIVITY + HEALTH */}
      <div className="grid grid-cols-2 gap-4">
        <section className="border border-border p-4">
          <div className="text-[10px] text-fg-caption mb-2">RECENT ACTIVITY · last 10</div>
          <ul className="space-y-1 text-[12px]">
            {d.recent_activity.map((r, i) => (
              <li key={i}>
                <span className="text-fg-muted">{r.ts.slice(0, 16)}</span> {r.event_type} ·{' '}
                {r.summary}
              </li>
            ))}
          </ul>
        </section>
        <section className="border border-border p-4">
          <div className="text-[10px] text-fg-caption mb-2">SUBSYSTEM HEALTH</div>
          <ul className="space-y-1 text-[12px]">
            {Object.entries(d.health).map(([k, v]) => (
              <li key={k}>
                <span
                  className={
                    v === 'healthy'
                      ? 'text-success'
                      : v === 'down' || v === 'unhealthy'
                        ? 'text-destructive'
                        : 'text-fg-muted'
                  }
                >
                  ■
                </span>{' '}
                {k.padEnd(10)} {v}
              </li>
            ))}
          </ul>
        </section>
      </div>

      {/* GAP QUEUE */}
      <GapPanel />

      {/* SESSION + KB GROWTH */}
      <div className="grid grid-cols-2 gap-4">
        <section className="border border-border p-4">
          <div className="text-[10px] text-fg-caption mb-2">SESSION</div>
          {d.session ? (
            <>
              <div className="text-[12px]">
                session: <span className="text-fg">{d.session.session_id}</span>
              </div>
              <div className="text-[12px]">n_obs: {d.session.n_obs}</div>
              <div className="text-[12px] text-fg-muted">started: {d.session.start_ts}</div>
            </>
          ) : (
            <div className="text-fg-muted text-[12px]">(no session in flight)</div>
          )}
        </section>
        <section className="border border-border p-4">
          <div className="text-[10px] text-fg-caption mb-2">KB GROWTH</div>
          <div className="text-[12px]">
            total: <span className="text-fg">{d.kb_growth.total_nodes}</span>
          </div>
          {Object.entries(d.kb_growth.by_source).map(([src, n]) => (
            <div key={src} className="text-[12px] text-fg-muted">
              {src.padEnd(10)} {n}{' '}
              {d.kb_growth.last_7d_delta[src] != null && (
                <span className="text-success">+{d.kb_growth.last_7d_delta[src]}</span>
              )}
            </div>
          ))}
        </section>
      </div>
    </div>
  );
}
