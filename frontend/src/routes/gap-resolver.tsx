import { useEffect, useState, useCallback, useMemo } from 'react';
import { useSearchParams, useNavigate } from 'react-router';
import { api } from '@/lib/api';
import InlineReplayer, { ReplayStep } from '@/components/InlineReplayer';
import GapPanel from '@/components/GapPanel';
import { truncateClusterKey } from '@/lib/format';
import './replayer-felt.css';

// :: GAP RESOLVER :: — rapid-authoring surface for flagged_sparse coverage holes.
// Mirrors replayer.tsx structure; uses useSearchParams for decision_id + hand_id.

interface GapMeta {
  decision_id: string;
  hand_id: string;
  cluster_key: string;
  legal_actions: string[];
  is_multiway: boolean;
  max_neighbor_distance: number;
  n_players_active: number;
  street: string;
  prev_node_id: string | null;
  send_to_solver_disabled: boolean;
}

interface GapQueueItem {
  decision_id: string;
  hand_id: string;
  max_neighbor_distance: number;
}

export default function GapResolver() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const decisionId = searchParams.get('decision_id');
  const handId = searchParams.get('hand_id');

  const [steps, setSteps] = useState<ReplayStep[]>([]);
  const [gapMeta, setGapMeta] = useState<GapMeta | null>(null);
  const [queue, setQueue] = useState<GapQueueItem[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [fastMode, setFastMode] = useState(false);
  const [resolving, setResolving] = useState(false);
  const [resolveError, setResolveError] = useState<string | null>(null);
  const [resolved, setResolved] = useState(false);
  const [solverEnqueued, setSolverEnqueued] = useState(false);
  const [solverError, setSolverError] = useState<string | null>(null);

  useEffect(() => {
    if (!handId || !decisionId) return;
    setErr(null);
    setSteps([]);
    setGapMeta(null);
    setResolved(false);
    setResolveError(null);
    setSolverEnqueued(false);
    setSolverError(null);
    Promise.all([
      api.get<ReplayStep[]>(`/api/hands/by-hand/${encodeURIComponent(handId)}/replay`),
      api.get<GapMeta>(`/api/gaps/${encodeURIComponent(decisionId)}`),
    ])
      .then(([s, meta]) => {
        setSteps(s);
        setGapMeta(meta);
      })
      .catch((e) => setErr(String(e)));
  }, [handId, decisionId]);

  useEffect(() => {
    api.get<GapQueueItem[]>('/api/gaps?limit=200')
      .then(setQueue)
      .catch(() => {});
  }, []);

  const nextGap = (() => {
    if (!decisionId || queue.length === 0) return null;
    const idx = queue.findIndex((q) => q.decision_id === decisionId);
    if (idx < 0 || idx >= queue.length - 1) return null;
    return queue[idx + 1];
  })();

  // Cut the hand off AT the decision point so the author can't see what was
  // actually done. Truncate to the gap step, strip the hero's own action +
  // chips from it (the answer), and surface the hero position so InlineReplayer
  // hides villain cards and anchors the felt on the seat-to-decide. Fail open
  // (full hand, no hero) if the decision_id isn't in the timeline.
  const { gapSteps, heroPosition, maskError } = useMemo<{
    gapSteps: ReplayStep[];
    heroPosition: string | null;
    maskError: string | null;
  }>(() => {
    if (!steps.length || !decisionId)
      return { gapSteps: steps, heroPosition: null, maskError: null };
    const gapIdx = steps.findIndex((s) => s.decision_id === decisionId);
    if (gapIdx < 0) return { gapSteps: steps, heroPosition: null, maskError: null };
    const hero = steps[gapIdx].actor ?? null;
    const truncated = steps.slice(0, gapIdx + 1);
    const gapStep = truncated[gapIdx];
    // Invariant #7: a decision view must show the PRE-action pot, never the
    // post-action pot — the latter leaks the action the author must guess.
    // pot_before is the only faithful source; refuse the step if it's absent
    // rather than silently falling back to the post-action pot.
    const preActionPot = gapStep.pot_before;
    if (preActionPot == null) {
      return {
        gapSteps: steps,
        heroPosition: null,
        maskError: `gap step ${decisionId} has no pot_before — cannot mask the decision without leaking the action`,
      };
    }
    const committed = { ...(gapStep.street_committed ?? {}) };
    if (hero) delete committed[hero];
    const delta = (gapStep.pot ?? 0) - preActionPot;
    const seatMapPre = { ...(gapStep.seat_map ?? {}) };
    if (hero && delta > 0) {
      const entry = Object.entries(seatMapPre).find(([, s]) => s.name === hero);
      if (entry) {
        const [k, s] = entry;
        seatMapPre[k] = { ...s, stack: Math.round((s.stack + delta) * 100) / 100 };
      }
    }
    truncated[gapIdx] = {
      ...gapStep,
      pot: preActionPot,
      action_taken: undefined,
      bet_amount: undefined,
      street_committed: committed,
      seat_map: seatMapPre,
    };
    return { gapSteps: truncated, heroPosition: hero, maskError: null };
  }, [steps, decisionId]);

  const handleResolve = useCallback(
    async (actionDist: Record<string, number>) => {
      if (!decisionId) return;
      setResolving(true);
      setResolveError(null);
      try {
        await api.post(`/api/gaps/${encodeURIComponent(decisionId)}/resolve`, {
          action_dist: actionDist,
        });
        setResolved(true);
        setResolving(false);
        setQueue((q) => q.filter((x) => x.decision_id !== decisionId));
        if (fastMode && nextGap) {
          navigate(
            `/gap-resolver?decision_id=${encodeURIComponent(nextGap.decision_id)}&hand_id=${encodeURIComponent(nextGap.hand_id)}`,
          );
        }
      } catch (e) {
        setResolveError(e instanceof Error ? e.message : String(e));
        setResolving(false);
      }
    },
    [decisionId, fastMode, nextGap, navigate],
  );

  const handleSendToSolver = useCallback(async () => {
    if (!decisionId || gapMeta?.send_to_solver_disabled) return;
    setSolverError(null);
    try {
      await api.post(`/api/gaps/${encodeURIComponent(decisionId)}/send-to-solver`);
      setSolverEnqueued(true);
    } catch (e) {
      setSolverError(e instanceof Error ? e.message : String(e));
    }
  }, [decisionId, gapMeta]);

  const handleNextGap = useCallback(() => {
    if (!nextGap) return;
    navigate(
      `/gap-resolver?decision_id=${encodeURIComponent(nextGap.decision_id)}&hand_id=${encodeURIComponent(nextGap.hand_id)}`,
    );
  }, [nextGap, navigate]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName ?? '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || target?.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 'n' || e.key === 'N') {
        handleNextGap();
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleNextGap]);

  if (!decisionId || !handId) {
    // A single param present is a malformed deep-link — keep the error.
    if (decisionId || handId) {
      return (
        <div className="min-h-screen bg-page text-fg font-mono p-6">
          <div className="text-destructive text-[12px]">
            ERROR: missing {decisionId ? 'hand_id' : 'decision_id'} query parameter
          </div>
        </div>
      );
    }
    // Bare /gap-resolver (nav tab / hotkey) — show the queue to pick a gap from.
    return (
      <div className="min-h-screen bg-page text-fg font-mono p-4 space-y-3">
        <h1 className="text-[14px] font-bold">:: GAP RESOLVER :: pick a gap to resolve</h1>
        <GapPanel />
      </div>
    );
  }

  if (err) {
    return (
      <div className="min-h-screen bg-page text-fg font-mono p-6">
        <div className="text-destructive text-[12px]">ERROR: {err}</div>
      </div>
    );
  }

  if (maskError) {
    return (
      <div className="min-h-screen bg-page text-fg font-mono p-6">
        <div className="text-destructive text-[12px]">ERROR: {maskError}</div>
      </div>
    );
  }

  if (!steps.length || !gapMeta) {
    return (
      <div className="min-h-screen bg-page text-fg font-mono p-6">
        <div className="text-fg-muted text-[12px]">loading gap {decisionId}…</div>
      </div>
    );
  }

  const idShort = decisionId.slice(-8);
  const clusterKeyTruncated = truncateClusterKey(gapMeta.cluster_key, 40);
  const distance = gapMeta.max_neighbor_distance;

  return (
    <div className="min-h-screen bg-page text-fg font-mono p-4 space-y-2">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <h1 className="text-[14px] font-bold truncate">
            :: GAP RESOLVER :: decision_{idShort} · {clusterKeyTruncated}
          </h1>
          <div className="text-[12px] text-fg-muted mt-1">
            hand_{handId} · street: {gapMeta.street} · distance:{' '}
            {distance.toFixed(3)}
            {gapMeta.is_multiway && (
              <span className="text-warning ml-2">[MULTIWAY]</span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3 flex-shrink-0">
          <button
            aria-pressed={fastMode}
            onClick={() => setFastMode((f) => !f)}
            className="text-[11px] text-fg border border-border px-2 py-0.5"
          >
            FAST [
            <span className={fastMode ? 'text-fg-muted' : 'text-accent'}> off </span>|
            <span className={fastMode ? 'text-accent' : 'text-fg-muted'}> ON </span>]
          </button>

          {nextGap ? (
            <button
              onClick={handleNextGap}
              className="border border-border px-2 py-0.5 text-[11px] text-fg"
            >
              [ next gap → ]
            </button>
          ) : (
            <button
              disabled
              className="border border-border px-2 py-0.5 text-[11px] text-fg opacity-40"
            >
              [ no more gaps ]
            </button>
          )}
        </div>
      </div>

      {resolved && !fastMode && (
        <div className="text-[12px] text-success">resolved — click next gap to continue</div>
      )}

      <InlineReplayer
        steps={gapSteps}
        rx={360}
        ry={215}
        hideVillainCards
        heroPosition={heroPosition}
        gapDecisionId={decisionId}
        gapLegalActions={gapMeta.legal_actions}
        isMultiway={gapMeta.is_multiway}
        fastMode={fastMode}
        onResolveGap={handleResolve}
        onSendToSolver={handleSendToSolver}
        resolving={resolving}
        resolveError={resolveError}
        prevNodeId={gapMeta.prev_node_id}
        clusterKey={gapMeta.cluster_key}
      />

      <div className="flex justify-end">
        {gapMeta.send_to_solver_disabled ? (
          <div className="flex flex-col items-end gap-1">
            <button
              disabled
              className="border border-border px-2 py-0.5 text-[11px] text-fg opacity-40 cursor-not-allowed"
            >
              [ send to solver ]
            </button>
            <div className="text-[10px] text-fg-caption">
              solver is HU-only · multiway gaps cannot be solver-resolved
            </div>
          </div>
        ) : solverEnqueued ? (
          <button
            disabled
            className="border border-border px-2 py-0.5 text-[11px] text-fg opacity-40"
          >
            [ solver enqueued ]
          </button>
        ) : (
          <div className="flex flex-col items-end gap-1">
            <button
              onClick={handleSendToSolver}
              className="border border-border px-2 py-0.5 text-[11px] text-fg"
            >
              [ send to solver ]
            </button>
            {solverError && (
              <div className="text-[10px] text-destructive">
                send to solver failed: {solverError}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
