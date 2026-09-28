import { useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router';
import { api } from '@/lib/api';
import EditNodeDrawer from '@/components/drawer/EditNodeDrawer';
import CardPicker from '@/components/CardPicker';
import RangeViewer from '@/components/RangeViewer';
import HandHistoryRail from '@/components/probe/HandHistoryRail';
import StepEditorPanel from '@/components/probe/StepEditorPanel';
import ClusterKeyLine from '@/components/probe/ClusterKeyLine';
import SpotPreview from '@/components/probe/SpotPreview';
import KnnPanel from '@/components/probe/KnnPanel';
import EngineResponsePanel from '@/components/probe/EngineResponsePanel';
import ABComparePanel from '@/components/probe/ABComparePanel';
import ActionsPanel from '@/components/probe/ActionsPanel';
import { encodeSpot, type EncodeForm } from '@/lib/probeEncoder';
import type { RangeViewerTab } from '@/components/RangeViewer';
import './probe.css';

interface ProbeNeighbor {
  cluster_key: string;
  decision_id: string | null;
  distance: number;
  action_dist: Record<string, number> | null;
  obs_id: string | null;
  hand_id: string | null;
  n_obs: number;
}

interface ProbeEngineResponse {
  source: string | null;
  n_obs: number;
  action_dist: Record<string, number> | null;
  ev_loss: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  flagged_sparse?: boolean | null;
  max_neighbor_dist?: number | null;
  empty_reason?: string;
}

interface ProbeResponse {
  engine_response: ProbeEngineResponse;
  knn_neighbors: ProbeNeighbor[];
  solver_truth: {
    action_dist: Record<string, number>;
    kl_actual_vs_solver?: number | null;
  } | null;
}

interface RangeBatchResponse {
  hands: Array<{
    hand_class: string;
    action_dist: Record<string, number>;
    combos: number;
  }>;
  aggregate_range_pct: number;
  legend: Record<string, string>;
}

function emptyForm(): EncodeForm {
  return {
    heroPos: null,
    preflop: [],
    flop: { board: [null, null, null], actions: [] },
    turn: { board: [null], actions: [] },
    river: { board: [null], actions: [] },
    heroHole: [null, null],
  };
}

type ActiveStep = 'preflop' | 'flop' | 'turn' | 'river';
type PickerTarget = { street: 'hole' } | { street: 'flop' | 'turn' | 'river'; slot: number };

export default function Probe() {
  const [params, setParams] = useSearchParams();

  const [form, setForm] = useState<EncodeForm>(emptyForm);
  const [picker, setPicker] = useState<PickerTarget | null>(null);
  const [activeStep, setActiveStep] = useState<ActiveStep>('preflop');
  const [showPreview, setShowPreview] = useState(false);

  const [response, setResponse] = useState<ProbeResponse | null>(null);
  const [serverClusterKey, setServerClusterKey] = useState<string | null>(null);
  const [queriedKey, setQueriedKey] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [querying, setQuerying] = useState(false);

  const [rangeData, setRangeData] = useState<RangeBatchResponse | null>(null);
  const [spotB, setSpotB] = useState<{ cluster_key: string; action_dist: Record<string, number> | null } | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const encodeSeq = useRef(0);
  const querySeq = useRef(0);
  const spotBSeq = useRef(0);

  const enc = useMemo(() => encodeSpot(form), [form]);

  // client preview encoder — PREVIEW ONLY for instant keystroke feedback
  const previewKey = enc.key;

  // authoritative server encode: debounced on form changes; serverClusterKey is used for
  // all actual engine queries and range-batch calls (D-14-08 boundary)
  useEffect(() => {
    const seq = ++encodeSeq.current;
    const isValid = enc.valid;
    const spotDict = buildSpotDictFromForm(form);
    const delay = isValid ? 300 : 0;
    const handle = setTimeout(async () => {
      if (seq !== encodeSeq.current) return;
      if (!isValid) { setServerClusterKey(null); return; }
      try {
        const r = await api.post<{ cluster_key: string; empty_reason: string | null }>(
          '/api/probe/encode',
          { spot: spotDict },
        );
        if (seq !== encodeSeq.current) return;
        setServerClusterKey(r.cluster_key);
      } catch {
        if (seq !== encodeSeq.current) return;
        setServerClusterKey(null);
      }
    }, delay);
    return () => clearTimeout(handle);
  }, [form, enc.valid]);

  const reachedMap = useMemo(() => {
    const flopReached = form.preflop.length >= 2 || form.preflop.some((t) => t.verb !== 'fold');
    const turnReached = flopReached && form.flop.board.filter(Boolean).length === 3;
    const riverReached = turnReached && form.turn.board.filter(Boolean).length === 1;
    return { preflop: true, flop: flopReached, turn: turnReached, river: riverReached };
  }, [form]);

  const queried = !!queriedKey && queriedKey === serverClusterKey;

  const actionKey = queriedKey ?? '';

  async function runQuery() {
    if (!enc.valid || !serverClusterKey) return;
    const seq = ++querySeq.current;
    setQuerying(true);
    setLoading(true);
    setErr(null);
    try {
      const r = await api.post<ProbeResponse>('/api/probe/spot', {
        spot: buildSpotDictFromForm(form),
        k: 5,
      });
      if (seq !== querySeq.current) return;
      setResponse(r);
      setQueriedKey(serverClusterKey);
      await fetchRangeBatch(seq);
    } catch (e) {
      if (seq !== querySeq.current) return;
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      if (seq === querySeq.current) {
        setLoading(false);
        setQuerying(false);
      }
    }
  }

  // Range-batch needs the built spot dict + actor (endpoint = {spot, actor_position}).
  // Only runnable when a form-built spot exists; key-only loads skip it.
  async function fetchRangeBatch(seq: number) {
    if (!form.heroPos) { setRangeData(null); return; }
    try {
      const r = await api.post<RangeBatchResponse>('/api/probe/range-batch', {
        spot: buildSpotDictFromForm(form),
        actor_position: form.heroPos,
      });
      if (seq !== querySeq.current) return;
      setRangeData(r);
    } catch {
      if (seq !== querySeq.current) return;
      setRangeData(null);
    }
  }

  function loadFromKey(key: string) {
    const ck = key.trim();
    if (!ck) return;
    const seq = ++querySeq.current;
    // A key-load supersedes any in-flight form query, whose spinner teardown is
    // now skipped (stale seq); clear it here so querying can't stick true.
    setQuerying(false);
    setLoading(true);
    setErr(null);
    api
      .get<ProbeResponse>(`/api/probe?cluster_key=${encodeURIComponent(ck)}&k=5`)
      .then((r) => {
        if (seq !== querySeq.current) return;
        setResponse(r);
        setQueriedKey(ck);
        setParams({ cluster_key: ck });
      })
      .catch((e) => {
        if (seq !== querySeq.current) return;
        setErr(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (seq === querySeq.current) setLoading(false);
      });
  }

  function loadSpotB(key: string) {
    const ck = key.trim();
    if (!ck) return;
    const seq = ++spotBSeq.current;
    api
      .get<ProbeResponse>(`/api/probe?cluster_key=${encodeURIComponent(ck)}&k=5`)
      .then((r) => {
        if (seq !== spotBSeq.current) return;
        setSpotB({ cluster_key: ck, action_dist: r.engine_response.action_dist });
      })
      .catch(() => {
        if (seq !== spotBSeq.current) return;
        setSpotB(null);
      });
  }

  const deepLinkRan = useRef(false);
  useEffect(() => {
    if (deepLinkRan.current) return;
    const ck = params.get('cluster_key');
    if (!ck) return;
    deepLinkRan.current = true;
    const handle = setTimeout(() => loadFromKey(ck), 0);
    return () => clearTimeout(handle);
    // deep-link fires once at mount only; deepLinkRan ref guarantees idempotence
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const rangeViewerTabs = useMemo((): RangeViewerTab[] => {
    if (!rangeData) return [];
    const freqMap: Record<string, { call: number; raise: number; fold: number }> = {};
    const comboCounts: Record<string, number> = {};
    for (const h of rangeData.hands) {
      freqMap[h.hand_class] = {
        call: h.action_dist?.call ?? 0,
        raise: h.action_dist?.raise ?? 0,
        fold: h.action_dist?.fold ?? 0,
      };
      comboCounts[h.hand_class] = h.combos;
    }
    return [{
      label: 'action',
      actor: form.heroPos || 'hero',
      freqMap,
      comboCounts,
    }];
  }, [rangeData, form.heroPos]);

  const heroHand = useMemo(() => {
    const cards = form.heroHole.filter(Boolean) as string[];
    if (cards.length !== 2) return null;
    const [a, b] = cards;
    const ra = a[0], rb = b[0];
    const sa = a[1], sb = b[1];
    if (ra === rb) return `${ra}${rb}`;
    const order = 'AKQJT98765432';
    const hi = order.indexOf(ra) < order.indexOf(rb) ? ra : rb;
    const lo = hi === ra ? rb : ra;
    return `${hi}${lo}${sa === sb ? 's' : 'o'}`;
  }, [form.heroHole]);

  function clearForm() {
    setForm(emptyForm());
    setResponse(null);
    setQueriedKey(null);
    setServerClusterKey(null);
    setRangeData(null);
    setSpotB(null);
    setErr(null);
  }

  function openHolePicker() { setPicker({ street: 'hole' }); }
  function openBoardPicker(street: 'flop' | 'turn' | 'river', slot: number) {
    setPicker({ street, slot });
  }

  function onPickHole(cards: string[]) {
    setForm((f) => ({ ...f, heroHole: cards.length === 2 ? cards : [cards[0] ?? null, cards[1] ?? null] }));
    if (cards.length >= 2) setPicker(null);
  }

  function onPickBoard(cards: string[]) {
    if (!picker || picker.street === 'hole') return;
    const street = picker.street;
    const slotsNeeded = { flop: 3, turn: 1, river: 1 }[street];
    const newBoard = Array.from({ length: slotsNeeded }, (_, i) => cards[i] ?? null);
    setForm((f) => ({ ...f, [street]: { ...f[street], board: newBoard } }));
    if (cards.length >= slotsNeeded) setPicker(null);
  }

  const pickerSlots = picker?.street === 'hole' ? 2 : picker ? { flop: 3, turn: 1, river: 1 }[picker.street] : 0;
  const pickerSelected = picker?.street === 'hole'
    ? (form.heroHole.filter(Boolean) as string[])
    : picker
    ? (form[picker.street as 'flop' | 'turn' | 'river'].board.filter(Boolean) as string[])
    : [];

  const pickerUsedElsewhere = useMemo(() => {
    if (!picker) return new Set<string>();
    const used = new Set<string>([
      ...(form.heroHole.filter(Boolean) as string[]),
      ...(form.flop.board.filter(Boolean) as string[]),
      ...(form.turn.board.filter(Boolean) as string[]),
      ...(form.river.board.filter(Boolean) as string[]),
    ]);
    const own = picker.street === 'hole'
      ? (form.heroHole.filter(Boolean) as string[])
      : (form[picker.street as 'flop' | 'turn' | 'river'].board.filter(Boolean) as string[]);
    for (const c of own) used.delete(c);
    return used;
  }, [picker, form]);

  return (
    <div className="probe-v2">
      {/* ============= LEFT: RAIL + STEP EDITOR ============= */}
      <div className="probe-v2-left">
        <h1 className="page-title">
          <span className="pt-tag">::</span> PROBE <span className="pt-sep">::</span> spot builder v2
        </h1>
        <div className="page-sub">
          4-chip rail · one active step editor · server encode authoritative
        </div>

        <ClusterKeyLine
          previewKey={previewKey}
          valid={enc.valid}
          querying={querying}
          queried={queried}
          structural={enc.structural}
          onLoad={loadFromKey}
          onLoadB={loadSpotB}
        />

        <HandHistoryRail
          form={form}
          enc={enc}
          focusedStreet={activeStep}
          onFocus={(s) => {
            const step = s as ActiveStep;
            setActiveStep(step);
            if (step === 'preflop') {
              if (!form.heroHole.every(Boolean)) openHolePicker();
            } else if (step === 'flop' || step === 'turn' || step === 'river') {
              const need = { flop: 3, turn: 1, river: 1 }[step];
              if (reachedMap[step] && form[step].board.filter(Boolean).length < need) openBoardPicker(step, 0);
            }
          }}
          reachedMap={reachedMap}
          onEditBoard={(street) => openBoardPicker(street as 'flop' | 'turn' | 'river', 0)}
        />

        <StepEditorPanel
          activeStep={activeStep}
          setActiveStep={setActiveStep}
          form={form}
          setForm={setForm}
          reachedMap={reachedMap}
          enc={enc}
        />

        {/* Run row */}
        <div className="runrow">
          <span className={'runrow-status ' + (enc.valid ? 'ok' : 'bad')}>
            {querying
              ? <>● querying engine · kNN search over cluster_keys</>
              : enc.valid
                ? <>● spot is legal · {enc.parts.potType} {enc.parts.position} · decision @ {enc.parts.street}</>
                : <>● {enc.structural.length} issue{enc.structural.length === 1 ? '' : 's'}</>}
          </span>
          <button type="button" className="btn btn-ghost" onClick={clearForm} disabled={querying}>[ clear ]</button>
          <button
            type="button"
            className={'btn' + (showPreview ? ' btn-active' : '')}
            disabled={!enc.valid}
            onClick={() => setShowPreview((v) => !v)}
          >
            {showPreview ? '[ hide preview ▲ ]' : '[ visualize spot ▶ ]'}
          </button>
          <button
            type="button"
            className={'btn btn-primary' + (querying ? ' querying' : '')}
            disabled={!enc.valid || !serverClusterKey || loading}
            onClick={runQuery}
          >
            {querying ? '[ querying… ]' : '[ run query ]'}
          </button>
        </div>

        {/* Cmdline */}
        <div className="cmdline">
          <span className="cmdline-prompt">$</span>
          <span className="muted">
            {querying
              ? 'querying engine · kNN search · ~600ms'
              : picker
              ? 'card picker open · esc to dismiss · 1-4 filter suits'
              : !enc.valid
              ? `incomplete spot — ${enc.structural[0]?.msg ?? '...'}`
              : `cluster_key ready · ${enc.key}`}
          </span>
          <span className="cmdline-right">
            <span className="kbd">⌘K</span> palette &nbsp;
            <span className="kbd">/</span> focus &nbsp;
            <span className="kbd">?</span> help
          </span>
        </div>
      </div>

      {/* ============= RIGHT: RESULT COLUMN ============= */}
      <div className="probe-v2-right">
        <div className="cap" style={{ marginBottom: 10 }}>
          query results · {querying ? 'querying…' : (response ? 'loaded' : 'awaiting query')}
        </div>

        {querying && <div className="query-progress"></div>}

        {showPreview && <SpotPreview form={form} enc={enc} />}

        {err && (
          <div className="section">
            <div className="section-body" style={{ color: 'var(--color-destructive)' }}>
              ERROR: {err}
            </div>
          </div>
        )}

        {(querying || response) && (
          <EngineResponsePanel
            clusterKey={queriedKey ?? ''}
            result={response}
            onRefreshResult={() => { if (queriedKey) loadFromKey(queriedKey); }}
            querying={querying}
          />
        )}

        {spotB && response && queriedKey && (
          <ABComparePanel
            specA={{ cluster_key: queriedKey, action_dist: response.engine_response.action_dist }}
            specB={spotB}
            onDismiss={() => setSpotB(null)}
          />
        )}

        {(querying || response) && (
          <KnnPanel
            neighbors={response?.knn_neighbors ?? []}
            onQuery={loadFromKey}
            querying={querying}
          />
        )}

        <RangeViewer
          tabs={rangeViewerTabs}
          heroHand={heroHand}
        />

        {queriedKey && (
          <ActionsPanel
            clusterKey={queriedKey}
            onEdit={() => setDrawerOpen(true)}
          />
        )}

        {drawerOpen && (
          <EditNodeDrawer
            open={drawerOpen}
            onClose={() => setDrawerOpen(false)}
            clusterKey={actionKey || null}
            onSaved={() => {
              setDrawerOpen(false);
              if (actionKey) loadFromKey(actionKey);
            }}
          />
        )}
      </div>

      {/* card picker modal */}
      {picker && (
        <div
          className="picker-backdrop"
          onClick={(e) => { if (e.target === e.currentTarget) setPicker(null); }}
        >
          <div className="picker-modal" role="dialog" aria-label="card picker">
            <div className="picker-modal-head">
              <span>
                <span className="muted">pick </span>
                <span className="picker-modal-target">
                  {picker.street === 'hole' ? 'hero hole' : `${picker.street} board`}
                </span>
              </span>
              <span className="row">
                <span className="picker-modal-meta">esc to dismiss</span>
                <button type="button" className="btn btn-small" onClick={() => setPicker(null)}>[ x ]</button>
              </span>
            </div>
            <div className="picker-modal-body">
              <CardPicker
                target={picker.street === 'hole' ? 'hole' : 'board'}
                slots={pickerSlots}
                selected={pickerSelected}
                usedElsewhere={pickerUsedElsewhere}
                onChange={picker.street === 'hole' ? onPickHole : onPickBoard}
                onClose={() => setPicker(null)}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function buildSpotDictFromForm(form: EncodeForm): Record<string, unknown> {
  const preflopSeq = form.preflop
    .filter((t) => t.verb !== 'fold')
    .map((t) => `${t.pos}:${t.verb}${t.size != null ? `_${t.size}` : ''}`);

  const flopActions = form.flop.actions.map((a) => {
    const verb = a.verb === 'check' ? 'check'
      : a.verb === 'call' ? 'call'
      : a.verb === 'fold' ? 'fold'
      : a.verb === 'bet' ? `bet_${a.size ?? 50}`
      : a.verb === 'raise' ? `raise_${a.size ?? '3x'}`
      : a.verb;
    return a.pos ? `${a.pos}:${verb}` : verb;
  });
  const turnActions = form.turn.actions.map((a) => {
    const verb = a.verb === 'check' ? 'check'
      : a.verb === 'call' ? 'call'
      : a.verb === 'fold' ? 'fold'
      : a.verb === 'bet' ? `bet_${a.size ?? 50}`
      : a.verb === 'raise' ? `raise_${a.size ?? '3x'}`
      : a.verb;
    return a.pos ? `${a.pos}:${verb}` : verb;
  });
  const riverActions = form.river.actions.map((a) => {
    const verb = a.verb === 'check' ? 'check'
      : a.verb === 'call' ? 'call'
      : a.verb === 'fold' ? 'fold'
      : a.verb === 'bet' ? `bet_${a.size ?? 50}`
      : a.verb === 'raise' ? `raise_${a.size ?? '3x'}`
      : a.verb;
    return a.pos ? `${a.pos}:${verb}` : verb;
  });

  const board = [
    ...form.flop.board.filter(Boolean),
    ...form.turn.board.filter(Boolean),
    ...form.river.board.filter(Boolean),
  ];

  const street = (() => {
    if (form.river.board.filter(Boolean).length > 0 || form.river.actions.length > 0) return 'river';
    if (form.turn.board.filter(Boolean).length > 0 || form.turn.actions.length > 0) return 'turn';
    if (form.flop.board.filter(Boolean).length > 0 || form.flop.actions.length > 0) return 'flop';
    return 'preflop';
  })();

  return {
    street,
    hero_position: form.heroPos,
    hero_hole: form.heroHole.filter(Boolean),
    board,
    action_sequence: [...preflopSeq, ...flopActions, ...turnActions, ...riverActions],
  };
}
