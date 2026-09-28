import { Fragment, useEffect, useState } from 'react';
import { api } from '@/lib/api';
import NewMatchDrawer from '@/components/drawer/NewMatchDrawer';
import { useJobSse } from '@/hooks/useJobSse';
import { useEvalRuns, type EvalRun } from '@/state/evalRuns';

// Eval panel per D-NEW-30 + UI-SPEC v2.2 §Eval panel.
//
// Sections:
//  - LEADERBOARD: GET /api/eval/leaderboard — per-opponent rollup
//  - MATCH HISTORY: GET /api/eval/matches?limit=50 — reverse chronological
//  - [+ new match] opens NewMatchDrawer; queued match returns to panel.

interface LeaderRow {
  opponent: string;
  n_matches: number;
  total_hands: number;
  avg_bb_per_100: number | null;
  avg_ci_low: number | null;
  avg_ci_high: number | null;
  wins: number;
  losses: number;
  ties: number;
  regressions: number;
}

interface MatchRow {
  match_id: string;
  opponent: string;
  hands: number;
  seed: number;
  bb_per_100: number | null;
  ci_low: number | null;
  ci_high: number | null;
  status: string;
  started_at: string;
  finished_at?: string | null;
  engine_version?: string;
}

function verdictText(m: MatchRow): string {
  switch (m.status) {
    case 'inconclusive':
      return 'CI brackets 0 — engine performance not distinguishable from baseline at this sample size. Run more hands for tighter bounds.';
    case 'won':
      return `Lower CI bound > 0 — engine outperforms ${m.opponent} with statistical significance.`;
    case 'lost':
      return `Upper CI bound < 0 — engine underperforms ${m.opponent} with statistical significance.`;
    case 'regression':
      return 'Engine version regressed vs the prior snapshot.';
    case 'running':
    case 'queued':
      return 'Match still in progress.';
    default:
      return '—';
  }
}

function durationDisplay(m: MatchRow): string {
  if (!m.finished_at) return '—';
  const s =
    (new Date(m.finished_at).getTime() - new Date(m.started_at).getTime()) /
    1000;
  if (!Number.isFinite(s) || s < 0) return '—';
  return `${s.toFixed(1)}s`;
}

const STATUS_COLORS: Record<string, string> = {
  won: 'text-success',
  lost: 'text-destructive',
  regression: 'text-destructive',
  inconclusive: 'text-fg-muted',
  running: 'text-warning',
  queued: 'text-fg-muted',
  failed: 'text-destructive',
};

// One in-flight match row, driven by the real job SSE stream. On a terminal
// event it removes itself from the durable store and triggers a history reload
// so the persisted match row appears in its place.
function RunningRow({
  run,
  onResolved,
}: {
  run: EvalRun;
  onResolved: () => void;
}) {
  const { removeRun } = useEvalRuns();
  const job = useJobSse<unknown>(run.job_id, 'eval');

  const done = job.status === 'done';
  const failed = job.status === 'failed';

  // Auto-resolve only on genuine completion. A `failed` status — which a
  // transient SSE transport drop can surface — must NOT erase a live run;
  // keep the row visible (FAILED) and let the user dismiss it.
  useEffect(() => {
    if (!done) return;
    const id = window.setTimeout(() => {
      removeRun(run.job_id);
      onResolved();
    }, 600);
    return () => window.clearTimeout(id);
  }, [done, removeRun, run.job_id, onResolved]);

  const handsDone = Number(job.progress?.hands_done ?? 0);
  const handsTotal = Number(job.progress?.hands_total ?? run.hands) || run.hands;
  const denom = handsTotal > 0 ? handsTotal : 1;
  const pct = failed
    ? 100
    : done
      ? 100
      : Math.min(100, Math.max(2, (handsDone / denom) * 100));
  const label = failed ? 'FAILED' : done ? 'DONE' : 'RUNNING';
  const tone = failed ? 'text-destructive' : 'text-warning';
  const barTone = failed ? 'bg-destructive' : 'bg-warning';

  return (
    <tr className="border-b border-border-faint">
      <td className={`p-2 ${tone}`}>(running)</td>
      <td className="p-2">{run.opponent}</td>
      <td className="p-2 text-right">{run.hands}</td>
      <td className="p-2 text-right text-fg-muted">
        {handsDone > 0 ? `${handsDone}/${handsTotal}` : '—'}
      </td>
      <td className="p-2 text-right">{run.seed}</td>
      <td className={`p-2 ${tone}`}>
        <div>{label}</div>
        <div className="mt-1 h-[2px] bg-border w-full">
          <div className={`h-[2px] ${barTone}`} style={{ width: `${pct}%` }} />
        </div>
      </td>
      <td className="p-2 text-fg-muted">{run.started_at.slice(0, 19)}</td>
      <td className="p-2">
        {failed && (
          <button
            className="text-[11px] text-fg-muted hover:text-fg"
            onClick={() => {
              removeRun(run.job_id);
              onResolved();
            }}
          >
            [ ✕ ]
          </button>
        )}
      </td>
    </tr>
  );
}

function MatchDetail({
  match,
  onClose,
}: {
  match: MatchRow;
  onClose: () => void;
}) {
  return (
    <div className="border border-success p-4 my-1">
      <div className="flex items-center justify-between mb-3">
        <div className="text-[12px] font-bold">
          :: MATCH DETAIL :: {match.match_id}
        </div>
        <button
          onClick={onClose}
          className="text-[12px] border border-border px-2"
          aria-label="close detail panel"
        >
          [ × ]
        </button>
      </div>

      <div className="grid grid-cols-2 gap-6 text-[12px]">
        <div>
          <div className="text-fg-caption text-[10px] mb-1">CONFIG</div>
          <div className="flex justify-between">
            <span className="text-fg-muted">opponent</span>
            <span>{match.opponent}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-fg-muted">hands</span>
            <span>{match.hands.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-fg-muted">seed</span>
            <span>{match.seed}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-fg-muted">engine_version</span>
            <span>{match.engine_version ?? '—'}</span>
          </div>
        </div>

        <div>
          <div className="text-fg-caption text-[10px] mb-1">OUTCOME</div>
          <div className="flex justify-between">
            <span className="text-fg-muted">status</span>
            <span className={STATUS_COLORS[match.status] ?? ''}>
              {match.status.toUpperCase()}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-fg-muted">bb/100</span>
            <span>{match.bb_per_100?.toFixed(2) ?? '—'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-fg-muted">CI</span>
            <span>
              [{match.ci_low?.toFixed(2) ?? '—'},{' '}
              {match.ci_high?.toFixed(2) ?? '—'}]
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-fg-muted">CI width</span>
            <span>
              {match.ci_low != null && match.ci_high != null
                ? (match.ci_high - match.ci_low).toFixed(2)
                : '—'}
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6 text-[12px] mt-4 pt-3 border-t border-border-faint">
        <div className="flex justify-between">
          <span className="text-fg-muted">started</span>
          <span>{match.started_at?.slice(0, 19)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-fg-muted">finished</span>
          <span>{match.finished_at?.slice(0, 19) ?? '—'}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-fg-muted">duration</span>
          <span>{durationDisplay(match)}</span>
        </div>
      </div>

      <div className="text-[12px] mt-4 pt-3 border-t border-border-faint">
        <div className="text-fg-caption text-[10px] mb-1">VERDICT</div>
        <div className="text-fg">{verdictText(match)}</div>
      </div>

      <div className="text-[10px] text-fg-muted italic mt-4 pt-3 border-t border-border-faint">
        per-hand breakdown not yet available — matches table stores aggregate
        stats only. Tracked as v2 backlog item.
      </div>
    </div>
  );
}

export default function Eval() {
  const [leaders, setLeaders] = useState<LeaderRow[]>([]);
  const [matches, setMatches] = useState<MatchRow[]>([]);
  const { runs, addRun } = useEvalRuns();
  const [drawer, setDrawer] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [selectedMatch, setSelectedMatch] = useState<MatchRow | null>(null);

  useEffect(() => {
    reload();
  }, []);

  // Esc clears the detail panel.
  useEffect(() => {
    if (!selectedMatch) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setSelectedMatch(null);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectedMatch]);

  function reload() {
    setErr(null);
    api
      .get<LeaderRow[]>('/api/eval/leaderboard')
      .then(setLeaders)
      .catch((e) => setErr(String(e)));
    api
      .get<MatchRow[]>('/api/eval/matches?limit=50')
      .then(setMatches)
      .catch((e) => setErr(String(e)));
  }

  function onQueued(
    jobId: string,
    cfg: { opponent: string; hands: number; seed: number },
  ) {
    addRun({
      job_id: jobId,
      opponent: cfg.opponent,
      hands: cfg.hands,
      seed: cfg.seed,
      started_at: new Date().toISOString(),
    });
    reload();
  }

  async function deleteMatch(matchId: string) {
    const prev = matches;
    setMatches((m) => m.filter((row) => row.match_id !== matchId));
    if (selectedMatch?.match_id === matchId) setSelectedMatch(null);
    try {
      await api.del(`/api/eval/matches/${matchId}`);
      reload();
    } catch (e) {
      setErr(String(e));
      setMatches(prev); // restore on failure
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-[14px] font-bold">
          :: EVALUATION :: head-to-head :: {matches.length + runs.length} matches
          {runs.length > 0 && (
            <span className="text-warning ml-2">({runs.length} running)</span>
          )}
        </h1>
        <button
          onClick={() => setDrawer(true)}
          className="border border-border px-2 py-1 text-[12px]"
        >
          [ + new match ]
        </button>
      </div>

      {err && <div className="text-destructive text-[12px]">ERROR: {err}</div>}

      {/* LEADERBOARD */}
      <section className="border border-border p-4">
        <div className="text-[10px] text-fg-caption mb-2">
          LEADERBOARD · engine vs each opponent
        </div>
        <table className="w-full text-[12px]">
          <thead className="text-fg-muted">
            <tr>
              <th className="text-left p-2">opponent</th>
              <th className="text-right p-2">matches</th>
              <th className="text-right p-2">total_hands</th>
              <th className="text-right p-2">avg bb/100 [CI]</th>
              <th className="text-right p-2">W/L/T/R</th>
            </tr>
          </thead>
          <tbody>
            {leaders.map((l) => (
              <tr
                key={l.opponent}
                className="border-b border-border-faint"
              >
                <td className="p-2">{l.opponent}</td>
                <td className="p-2 text-right">{l.n_matches}</td>
                <td className="p-2 text-right">{l.total_hands}</td>
                <td className="p-2 text-right">
                  <span
                    className={
                      (l.avg_bb_per_100 ?? 0) > 0
                        ? 'text-success'
                        : (l.avg_bb_per_100 ?? 0) < 0
                        ? 'text-destructive'
                        : ''
                    }
                  >
                    {l.avg_bb_per_100?.toFixed(2) ?? '—'}
                  </span>{' '}
                  <span className="text-fg-muted text-[10px]">
                    [{l.avg_ci_low?.toFixed(1) ?? '—'},{' '}
                    {l.avg_ci_high?.toFixed(1) ?? '—'}]
                  </span>
                </td>
                <td className="p-2 text-right">
                  <span className="text-success">{l.wins}</span>/
                  <span className="text-destructive">{l.losses}</span>/
                  <span className="text-fg-muted">{l.ties}</span>/
                  <span className="text-destructive">{l.regressions}</span>
                </td>
              </tr>
            ))}
            {leaders.length === 0 && (
              <tr>
                <td
                  colSpan={5}
                  className="p-4 text-fg-muted text-center"
                >
                  no leaderboard data yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      {/* MATCH HISTORY */}
      <section className="border border-border p-4">
        <div className="text-[10px] text-fg-caption mb-2">
          MATCH HISTORY · reverse chronological
        </div>
        <table className="w-full text-[12px]">
          <thead className="text-fg-muted">
            <tr>
              <th className="text-left p-2">match_id</th>
              <th className="text-left p-2">opponent</th>
              <th className="text-right p-2">hands</th>
              <th className="text-right p-2">bb/100 [CI]</th>
              <th className="text-right p-2">seed</th>
              <th className="text-left p-2">status</th>
              <th className="text-left p-2">when</th>
              <th className="p-2" />
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <RunningRow key={`run-${r.job_id}`} run={r} onResolved={reload} />
            ))}
            {matches.map((m) => {
              const isOpen = selectedMatch?.match_id === m.match_id;
              return (
                <Fragment key={m.match_id}>
                  <tr
                    onClick={() =>
                      setSelectedMatch((cur) =>
                        cur?.match_id === m.match_id ? null : m,
                      )
                    }
                    className={`border-b border-border-faint hover:bg-surface-2 cursor-pointer ${
                      isOpen ? 'bg-surface-2' : ''
                    }`}
                  >
                    <td className="p-2">{m.match_id?.slice(0, 8)}</td>
                    <td className="p-2">{m.opponent}</td>
                    <td className="p-2 text-right">{m.hands}</td>
                    <td className="p-2 text-right">
                      {m.bb_per_100?.toFixed(2) ?? '—'} [
                      {m.ci_low?.toFixed(1) ?? '—'},{' '}
                      {m.ci_high?.toFixed(1) ?? '—'}]
                    </td>
                    <td className="p-2 text-right">{m.seed}</td>
                    <td className={`p-2 ${STATUS_COLORS[m.status] ?? ''}`}>
                      {m.status.toUpperCase()}
                    </td>
                    <td className="p-2 text-fg-muted">
                      {m.started_at?.slice(0, 19)}
                    </td>
                    <td className="p-2 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteMatch(m.match_id);
                        }}
                        className="text-fg-muted hover:text-destructive px-1"
                        aria-label={`delete match ${m.match_id}`}
                        title="delete match"
                      >
                        [ × ]
                      </button>
                    </td>
                  </tr>
                  {isOpen && (
                    <tr>
                      <td colSpan={8} className="p-0">
                        <MatchDetail
                          match={m}
                          onClose={() => setSelectedMatch(null)}
                        />
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
            {matches.length === 0 && runs.length === 0 && (
              <tr>
                <td colSpan={8} className="p-4 text-fg-muted text-center">
                  no matches recorded yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      <NewMatchDrawer
        open={drawer}
        onClose={() => setDrawer(false)}
        onQueued={(jobId, cfg) => {
          onQueued(jobId, cfg);
        }}
      />
    </div>
  );
}
