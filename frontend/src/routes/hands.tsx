import { Fragment, useEffect, useState } from 'react';
import { api } from '@/lib/api';
import InlineReplayer, { ReplayStep } from '@/components/InlineReplayer';
import PostflopRangePanel from '@/components/PostflopRangePanel';
import { getRanges, type RangesContract } from '@/lib/rangeClient';

interface Hand {
  hand_id: string;
  source: string;
  hero_position?: string | null;
  stake?: string | null;
  n_decisions: number;
  street_reached?: string | null;
  played_ts?: string | null;
  solved: boolean;
  corpus_solved: boolean;
}

type SourceFilter = 'all' | 'hh' | 'sim';
const PAGE = 100;

export default function Hands() {
  const [rows, setRows] = useState<Hand[]>([]);
  const [source, setSource] = useState<SourceFilter>('all');
  const [solvedOnly, setSolvedOnly] = useState(false);
  const [corpusOnly, setCorpusOnly] = useState(false);
  const [offset, setOffset] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [reindexing, setReindexing] = useState(false);
  const [expandedHand, setExpandedHand] = useState<string | null>(null);
  const [replaySteps, setReplaySteps] = useState<Record<string, ReplayStep[]>>({});
  const [replayLoading, setReplayLoading] = useState<Record<string, boolean>>({});
  const [replayError, setReplayError] = useState<Record<string, string>>({});
  const [rangesPayload, setRangesPayload] = useState<Record<string, RangesContract[]>>({});
  const [rangesLoading, setRangesLoading] = useState<Record<string, boolean>>({});
  const [rangesError, setRangesError] = useState<Record<string, string>>({});
  const [exploitabilityPct, setExploitabilityPct] = useState<Record<string, number | null>>({});
  const [currentStep, setCurrentStep] = useState<Record<string, ReplayStep | null>>({});

  useEffect(() => {
    const params = new URLSearchParams();
    params.set('source', source);
    if (solvedOnly) params.set('solved', 'true');
    if (corpusOnly) params.set('corpus_solved', 'true');
    params.set('limit', String(PAGE));
    params.set('offset', String(offset));
    api
      .get<Hand[]>(`/api/hands?${params}`)
      .then((r) => {
        setRows(r);
        setErr(null);
      })
      .catch((e) => {
        setRows([]);
        setErr(e instanceof Error ? e.message : String(e));
      });
  }, [source, solvedOnly, corpusOnly, offset]);

  function setSourceReset(s: SourceFilter) { setSource(s); setOffset(0); }
  function setSolvedReset(v: boolean) { setSolvedOnly(v); setOffset(0); }
  function setCorpusReset(v: boolean) { setCorpusOnly(v); setOffset(0); }

  function reindex() {
    setReindexing(true);
    setErr(null);
    api
      .post('/api/hands/reindex', {})
      .catch((e) => setErr(`reindex failed: ${e instanceof Error ? e.message : String(e)}`))
      .finally(() => { setReindexing(false); setOffset(0); setSource((s) => s); });
  }

  function loadRanges(handId: string) {
    if (rangesLoading[handId]) return;
    setRangesLoading((l) => ({ ...l, [handId]: true }));
    getRanges(handId)
      .then((data) => setRangesPayload((s) => ({ ...s, [handId]: data })))
      .catch((e) => setRangesError((err) => ({ ...err, [handId]: e instanceof Error ? e.message : String(e) })))
      .finally(() => setRangesLoading((l) => ({ ...l, [handId]: false })));
  }

  function reloadRanges(handId: string) {
    setRangesPayload((s) => ({ ...s, [handId]: [] }));
    loadRanges(handId);
  }

  function toggleExpand(handId: string) {
    setExpandedHand((prev) => (prev === handId ? null : handId));
  }

  useEffect(() => {
    if (!expandedHand) return;
    const handId = expandedHand;
    if (!replaySteps[handId] && !replayLoading[handId]) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setReplayLoading((l) => ({ ...l, [handId]: true }));
      api
        .get<ReplayStep[]>(`/api/hands/by-hand/${encodeURIComponent(handId)}/replay`)
        .then((steps) => setReplaySteps((s) => ({ ...s, [handId]: steps })))
        .catch((e) => setReplayError((err) => ({ ...err, [handId]: e instanceof Error ? e.message : String(e) })))
        .finally(() => setReplayLoading((l) => ({ ...l, [handId]: false })));
    }
    if (rangesPayload[handId] === undefined && !rangesLoading[handId]) {
      loadRanges(handId);
    }
    // fetches are guarded by their own loading/loaded flags; effect re-runs only on expansion change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expandedHand]);

  const seg = (s: SourceFilter, label: string) => (
    <button
      onClick={() => setSourceReset(s)}
      className={`px-3 py-1 text-[11px] border border-border${source === s ? ' bg-accent text-bg' : ''}`}
    >
      {label}
    </button>
  );

  return (
    <div className="space-y-4">
      <h1 className="text-[14px] font-bold">:: HANDS :: all hands — real + sim</h1>
      {err && <div className="text-destructive text-[12px]">ERROR: {err}</div>}
      <div className="flex gap-3 flex-wrap items-center">
        <div className="flex">{seg('all', 'All')}{seg('hh', 'Real')}{seg('sim', 'SIM')}</div>
        <label className="flex items-center gap-1 text-[11px]">
          <input type="checkbox" checked={solvedOnly} onChange={(e) => setSolvedReset(e.target.checked)} />
          solved only
        </label>
        <label className="flex items-center gap-1 text-[11px]">
          <input type="checkbox" checked={corpusOnly} onChange={(e) => setCorpusReset(e.target.checked)} />
          corpus ◆ only
        </label>
        <button onClick={reindex} disabled={reindexing} className="text-[11px] border border-border px-2 py-1">
          {reindexing ? '[ reindexing… ]' : '[ reindex ]'}
        </button>
        <div className="ml-auto flex items-center gap-2 text-[11px]">
          <button onClick={() => setOffset((o) => Math.max(0, o - PAGE))} disabled={offset === 0}>[ ◀ prev ]</button>
          <span className="text-fg-muted">{offset + 1}–{offset + rows.length}</span>
          <button onClick={() => setOffset((o) => o + PAGE)} disabled={rows.length < PAGE}>[ next ▶ ]</button>
        </div>
      </div>
      <table className="w-full text-[12px]">
        <thead className="text-fg-muted border-b border-border">
          <tr>
            <th className="text-left p-2">played</th>
            <th className="text-left p-2">hand_id</th>
            <th className="text-left p-2">src</th>
            <th className="text-left p-2">stake</th>
            <th className="text-left p-2">hero</th>
            <th className="text-left p-2">dps</th>
            <th className="text-left p-2">street</th>
            <th className="text-left p-2">solved</th>
            <th className="text-left p-2">corpus</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isExpanded = expandedHand === r.hand_id;
            return (
              <Fragment key={r.hand_id}>
                <tr
                  className={`border-b border-border-faint hover:bg-surface-2 cursor-pointer${isExpanded ? ' border-l-2 border-accent' : ''}`}
                  onClick={() => toggleExpand(r.hand_id)}
                  tabIndex={0}
                  onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && toggleExpand(r.hand_id)}
                >
                  <td className="p-2 text-fg-muted">{r.played_ts?.slice(0, 19) ?? '—'}</td>
                  <td className="p-2">{r.hand_id}</td>
                  <td className="p-2">
                    <span className={r.source === 'hh' ? 'text-accent' : 'text-fg-muted'}>
                      {r.source === 'hh' ? 'REAL' : 'SIM'}
                    </span>
                  </td>
                  <td className="p-2">{r.stake ?? '—'}</td>
                  <td className="p-2">{r.hero_position ?? '—'}</td>
                  <td className="p-2">{r.n_decisions}</td>
                  <td className="p-2">{r.street_reached ?? '—'}</td>
                  <td className="p-2">{r.solved ? <span className="text-accent">✓</span> : '—'}</td>
                  <td className="p-2">{r.corpus_solved ? <span className="text-accent" title="covered by the solver_cache kNN corpus">◆</span> : '—'}</td>
                </tr>
                {isExpanded && (
                  <tr key={r.hand_id + '-expanded'}>
                    <td colSpan={9} className="bg-surface p-4 border-b border-border">
                      <div className="text-[10px] text-fg-caption mb-2 flex items-center gap-2">
                        :: REPLAY :: {r.hand_id}
                        <button
                          className="ml-4 border border-border px-2 text-[10px]"
                          aria-label="collapse hand replay"
                          onClick={(e) => { e.stopPropagation(); setExpandedHand(null); }}
                        >
                          [ ▲ collapse ]
                        </button>
                      </div>
                      {replayError[r.hand_id] ? (
                        <div className="text-destructive p-6 text-[12px]">ERROR: {replayError[r.hand_id]}</div>
                      ) : replayLoading[r.hand_id] || !replaySteps[r.hand_id] ? (
                        <div className="text-fg-muted text-[12px]">loading hand {r.hand_id}…</div>
                      ) : (
                        <InlineReplayer
                          steps={replaySteps[r.hand_id]}
                          rx={380}
                          ry={210}
                          heroPosition={r.hero_position ?? null}
                          onStepChange={(step) => setCurrentStep((s) => ({ ...s, [r.hand_id]: step }))}
                        />
                      )}
                      {rangesError[r.hand_id] && (
                        <div className="text-destructive text-[12px] mt-2">ranges error: {rangesError[r.hand_id]}</div>
                      )}
                      <PostflopRangePanel
                        handId={r.hand_id}
                        ranges={rangesPayload[r.hand_id]}
                        currentStep={currentStep[r.hand_id] ?? null}
                        heroPosition={r.hero_position ?? null}
                        exploitabilityPct={exploitabilityPct[r.hand_id] ?? null}
                        onSolve={() => {
                          reloadRanges(r.hand_id);
                          setExploitabilityPct((s) => ({ ...s, [r.hand_id]: null }));
                        }}
                      />
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
          {rows.length === 0 && (
            <tr>
              <td colSpan={9} className="p-4 text-fg-muted text-center">no hands match filters</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
