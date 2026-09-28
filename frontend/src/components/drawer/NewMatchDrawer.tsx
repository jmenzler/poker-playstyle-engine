import { useEffect, useState } from 'react';
import Drawer from './Drawer';
import { api } from '@/lib/api';

// New Match drawer per D-NEW-30 + UI-SPEC v2.2 §New Match drawer body.
//
// Two modes:
// - vs baseline: engine vs baseline opponents (table 2..6) -> POST /api/eval/run
// - vs version:  the v_new engine vs the v_old engine (head-to-head) -> POST /api/eval/compare
// Both surface a job_id via onQueued and flow through the Eval tab identically.

const OPPONENTS = [
  { name: 'random', desc: 'uniform random over legal actions' },
  { name: 'always-call', desc: 'call any bet ≤ pot; check otherwise' },
  { name: 'always-raise', desc: 'min-raise everything; never call' },
  { name: 'tight-passive', desc: 'top 12% RFI; fold to most aggression' },
  { name: 'LAG-profile', desc: 'wide opens, frequent 3-bets, aggressive cbet' },
  { name: 'TAG-profile', desc: 'solid 18% RFI, balanced 3-bet, smart cbet' },
] as const;

const DEFAULT_OPP = 'TAG-profile';
const TABLE_SIZES = [2, 3, 4, 5, 6] as const;
const HANDS_PRESETS = [1000, 10000, 50000, 100000] as const;

interface Version {
  version: number;
  label: string;
  source: string;
  cutoff_ts: number;
  created_at: string;
}

export interface QueuedConfig {
  opponent: string;
  hands: number;
  seed: number;
}

interface NewMatchDrawerProps {
  open: boolean;
  onClose: () => void;
  onQueued?: (jobId: string, config: QueuedConfig) => void;
}

interface QueuedState {
  jobId: string;
  opponent: string;
  hands: number;
  minutesEstimate: number;
}

function resizeSeats(prev: string[], tableSize: number): string[] {
  const need = tableSize - 1;
  const next = prev.slice(0, need);
  while (next.length < need) next.push(DEFAULT_OPP);
  return next;
}

export default function NewMatchDrawer({
  open,
  onClose,
  onQueued,
}: NewMatchDrawerProps) {
  const [mode, setMode] = useState<'baseline' | 'version'>('baseline');
  const [tableSize, setTableSize] = useState<number>(6);
  const [seats, setSeats] = useState<string[]>(() => resizeSeats([], 6));
  const [hands, setHands] = useState<number>(10000);
  const [customHands, setCustomHands] = useState<string>('');
  const [seed, setSeed] = useState<number>(42);
  const [err, setErr] = useState<string | null>(null);
  const [queueing, setQueueing] = useState(false);
  const [queued, setQueued] = useState<QueuedState | null>(null);
  const [versions, setVersions] = useState<Version[]>([]);
  const [vOld, setVOld] = useState<number | null>(null);
  const [vNew, setVNew] = useState<number | null>(null);

  useEffect(() => {
    if (!open) return;
    api
      .get<Version[]>('/api/eval/versions')
      .then((vs) => {
        setVersions(vs);
        if (vs.length) {
          const nums = vs.map((v) => v.version);
          setVOld((o) => (o == null ? Math.min(...nums) : o));
          setVNew((n) => (n == null ? Math.max(...nums) : n));
        }
      })
      .catch(() => {});
  }, [open]);

  const usingCustom = customHands !== '';
  const finalHands = usingCustom ? parseInt(customHands, 10) || 0 : hands;
  const label = seats.join('+');
  const versionLabel = `v${vNew}_vs_v${vOld}`;

  const baselineValid =
    seats.length === tableSize - 1 &&
    seats.every((s) => s !== '') &&
    finalHands >= 100 &&
    Number.isInteger(seed);
  const versionValid =
    vOld != null &&
    vNew != null &&
    vOld !== vNew &&
    finalHands >= 100 &&
    Number.isInteger(seed);
  const valid = mode === 'version' ? versionValid : baselineValid;

  // ~12.5k hands/min v1 estimate per UI-SPEC; clamp so tiny counts read >=1m.
  const minutesEstimate = Math.max(1, Math.ceil(finalHands / 12500));

  function changeTableSize(ts: number) {
    setTableSize(ts);
    setSeats((prev) => resizeSeats(prev, ts));
  }

  async function queue() {
    if (!valid) return;
    setQueueing(true);
    setErr(null);
    try {
      let jobId: string;
      let opponent: string;
      if (mode === 'version') {
        const r = await api.post<{ job_id: string }>('/api/eval/compare', {
          v_old: vOld,
          v_new: vNew,
          hands: finalHands,
          seed,
        });
        jobId = r.job_id;
        opponent = versionLabel;
      } else {
        const r = await api.post<{ job_id: string }>('/api/eval/run', {
          opponents: seats,
          hands: finalHands,
          seed,
          table_size: tableSize,
        });
        jobId = r.job_id;
        opponent = label;
      }
      onQueued?.(jobId, { opponent, hands: finalHands, seed });
      setQueued({ jobId, opponent, hands: finalHands, minutesEstimate });
      setTimeout(() => {
        setQueued(null);
        onClose();
      }, 1500);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(msg);
    } finally {
      setQueueing(false);
    }
  }

  const selectCls =
    'bg-page border border-border px-2 py-1 text-[12px] text-fg flex-1 min-w-0';

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title=":: NEW MATCH :: configure benchmark"
      footer={
        <div className="flex gap-2">
          <button
            onClick={onClose}
            disabled={queued !== null}
            className={`text-fg-muted border border-border px-2 py-1 text-[12px] ${
              queued !== null ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            [ cancel ]
          </button>
          <button
            onClick={queue}
            disabled={!valid || queueing || queued !== null}
            className={`border px-2 py-1 text-[12px] ${
              valid && !queueing && queued === null
                ? 'text-success border-success'
                : 'text-fg-muted border-border opacity-50'
            }`}
          >
            [ queue match ]
          </button>
        </div>
      }
    >
      {err && (
        <div className="text-destructive mb-2 text-[12px]">ERROR: {err}</div>
      )}

      {queued && (
        <div className="border border-success text-success p-3 mb-4 text-[12px]">
          <div className="font-bold mb-1">◆ QUEUED</div>
          <div className="text-fg">
            job: {queued.jobId.slice(0, 8)} · {queued.opponent} ·{' '}
            {queued.hands.toLocaleString()} hands
          </div>
          <div className="text-fg-muted">est. {queued.minutesEstimate}m</div>
        </div>
      )}

      {/* MODE section */}
      <div className="text-[10px] text-fg-caption mb-2 pb-1 border-b border-border-faint">
        MODE
      </div>
      <div className="flex gap-2 mb-6">
        {(['baseline', 'version'] as const).map((m) => {
          const isSelected = mode === m;
          return (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`border px-2 py-1 text-[12px] ${
                isSelected
                  ? 'border-success text-success'
                  : 'border-border hover:border-fg-muted'
              }`}
            >
              [ {isSelected ? `▶ vs ${m}` : `vs ${m}`} ]
            </button>
          );
        })}
      </div>

      {mode === 'baseline' && (
        <>
          {/* TABLE SIZE section */}
          <div className="text-[10px] text-fg-caption mb-2 pb-1 border-b border-border-faint">
            TABLE SIZE
          </div>
          <div className="flex gap-2 mb-1">
            {TABLE_SIZES.map((ts) => {
              const isSelected = tableSize === ts;
              const tsLabel = ts === 2 ? 'HU' : `${ts}-max`;
              return (
                <button
                  key={ts}
                  onClick={() => changeTableSize(ts)}
                  className={`border px-2 py-1 text-[12px] ${
                    isSelected
                      ? 'border-success text-success'
                      : 'border-border hover:border-fg-muted'
                  }`}
                >
                  [ {isSelected ? `▶ ${tsLabel}` : tsLabel} ]
                </button>
              );
            })}
          </div>
          <div className="text-[10px] text-fg-caption mt-1 mb-6">
            engine sits in seat 1. corpus is 6-max; HU (2) retrieves
            blind-vs-blind analogs — usable, but 6-max is the native distribution.
          </div>

          {/* OPPONENTS section */}
          <div className="text-[10px] text-fg-caption mb-2 pb-1 border-b border-border-faint flex items-center justify-between">
            <span>OPPONENTS · {tableSize - 1} seat{tableSize - 1 > 1 ? 's' : ''}</span>
            <span className="flex items-center gap-1">
              <span className="text-fg-muted">fill all:</span>
              <select
                value=""
                onChange={(e) => {
                  if (e.target.value) setSeats(Array(tableSize - 1).fill(e.target.value));
                }}
                className="bg-page border border-border px-1 py-0.5 text-[11px] text-fg"
              >
                <option value="">—</option>
                {OPPONENTS.map((o) => (
                  <option key={o.name} value={o.name}>
                    {o.name}
                  </option>
                ))}
              </select>
            </span>
          </div>
          <div className="space-y-2 mb-6">
            {seats.map((seatOpp, i) => {
              const desc = OPPONENTS.find((o) => o.name === seatOpp)?.desc ?? '';
              return (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="text-fg-muted shrink-0 w-12">seat {i + 2}</span>
                  <select
                    value={seatOpp}
                    onChange={(e) =>
                      setSeats((prev) =>
                        prev.map((s, idx) => (idx === i ? e.target.value : s)),
                      )
                    }
                    className={selectCls}
                    title={desc}
                  >
                    {OPPONENTS.map((o) => (
                      <option key={o.name} value={o.name}>
                        {o.name}
                      </option>
                    ))}
                  </select>
                </div>
              );
            })}
          </div>
        </>
      )}

      {mode === 'version' && (
        <>
          {/* VERSIONS section */}
          <div className="text-[10px] text-fg-caption mb-2 pb-1 border-b border-border-faint">
            VERSIONS · head-to-head (HU)
          </div>
          <div className="flex items-end gap-3 mb-1 text-[12px]">
            <label className="flex flex-col gap-0.5">
              <span className="text-[10px] text-fg-muted">old</span>
              <select
                value={vOld ?? ''}
                onChange={(e) => setVOld(Number(e.target.value))}
                className="bg-page border border-border px-2 py-1 text-fg"
              >
                {versions.map((v) => (
                  <option key={v.version} value={v.version}>
                    v{v.version} · {v.label}
                  </option>
                ))}
              </select>
            </label>
            <span className="pb-1.5 text-fg-muted">vs</span>
            <label className="flex flex-col gap-0.5">
              <span className="text-[10px] text-fg-muted">new</span>
              <select
                value={vNew ?? ''}
                onChange={(e) => setVNew(Number(e.target.value))}
                className="bg-page border border-border px-2 py-1 text-fg"
              >
                {versions.map((v) => (
                  <option key={v.version} value={v.version}>
                    v{v.version} · {v.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {versionValid ? null : (
            <div className="text-[10px] text-warning mt-1">
              pick two different versions
            </div>
          )}
          <div className="text-[10px] text-fg-caption mt-1 mb-6">
            the v_new engine plays the v_old engine head-to-head (paired seed);
            bb/100 is v_new's edge. persists like any match.
          </div>
        </>
      )}

      {/* HANDS section */}
      <div className="text-[10px] text-fg-caption mb-2 pb-1 border-b border-border-faint">
        HANDS
      </div>
      <div className="flex gap-2 mb-1">
        {HANDS_PRESETS.map((p) => {
          const isSelected = !usingCustom && hands === p;
          const presetLabel = p >= 1000 ? `${p / 1000}k` : String(p);
          return (
            <button
              key={p}
              onClick={() => {
                setHands(p);
                setCustomHands('');
              }}
              className={`border px-2 py-1 text-[12px] ${
                isSelected
                  ? 'border-success text-success'
                  : 'border-border hover:border-fg-muted'
              }`}
            >
              [ {isSelected ? `▶ ${presetLabel}` : presetLabel} ]
            </button>
          );
        })}
        <button
          onClick={() => setCustomHands(String(hands))}
          className={`border px-2 py-1 text-[12px] ${
            usingCustom
              ? 'border-success text-success'
              : 'border-border hover:border-fg-muted'
          }`}
        >
          [ {usingCustom ? '▶ custom' : 'custom'} ]
        </button>
      </div>
      {usingCustom && (
        <input
          type="number"
          value={customHands}
          onChange={(e) => setCustomHands(e.target.value)}
          className="w-32 mt-1 bg-page border border-border px-2 py-1 text-[12px]"
          min={100}
        />
      )}
      <div className="text-[10px] text-fg-caption mt-1 mb-6">
        larger counts give tighter CI but take longer. small counts (e.g. 1k)
        run fast but usually land inconclusive. 100k ≈ 8 min.
      </div>

      {/* SEED section */}
      <div className="text-[10px] text-fg-caption mb-2 pb-1 border-b border-border-faint">
        SEED
      </div>
      <div className="flex gap-2 mb-1">
        <input
          type="number"
          value={seed}
          onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
          className="w-32 bg-page border border-border px-2 py-1 text-[12px]"
        />
        <button
          onClick={() => setSeed(Math.floor(Math.random() * 2 ** 31))}
          className="border border-border px-2 py-1 text-[12px]"
        >
          [ randomize ]
        </button>
      </div>
      <div className="text-[10px] text-fg-caption mb-6">
        seed enables deterministic replay. same seed + same opponents + same
        engine version = identical hand sequence.
      </div>

      {/* Summary preview */}
      <div className="text-[12px] text-fg-muted pt-2 border-t border-border-faint">
        {mode === 'version'
          ? versionValid
            ? `v${vNew} vs v${vOld} · ${finalHands.toLocaleString()} hands · seed ${seed} · est. ${minutesEstimate}m`
            : 'pick two different versions and a hands count'
          : valid
            ? `engine vs ${label} · ${finalHands.toLocaleString()} hands · seed ${seed} · est. ${minutesEstimate}m`
            : 'configure table, opponents and hands count'}
      </div>
    </Drawer>
  );
}
