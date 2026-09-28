import { useState, useEffect, useRef } from 'react';
import ComboBreakdownPanel from './ComboBreakdownPanel';
import StrategyGrid from './StrategyGrid';
import ActionLegend from './ActionLegend';
import { buildGridModel } from '@/lib/gridModel';
import type { FillMode } from '@/lib/gridModel';
import {
  postSolveRanges,
  subscribeRangeSolveJob,
  type RangesContract,
  type RangeSeatPayload,
} from '@/lib/rangeClient';
import type { ReplayStep } from './InlineReplayer';
import '@/routes/probe.css';

const DEFAULT_EXPLOITABILITY = 1.0;

interface PostflopRangePanelProps {
  handId: string;
  ranges: RangesContract[] | undefined;
  currentStep: ReplayStep | null;
  heroPosition: string | null;
  exploitabilityPct?: number | null;
  onSolve: () => void;
}

interface Selected {
  seat: 'oop' | 'ip';
  handClass: string;
}

// Follow the replayer cursor by street: show the DP for whatever street the
// current step is on. Falls back to the first (earliest) DP off-street/preflop.
function resolveCurrentDp(
  ranges: RangesContract[],
  currentStep: ReplayStep | null,
): RangesContract | null {
  if (ranges.length === 0) return null;

  const street = currentStep?.street;
  if (street && street !== 'preflop') {
    const onStreet = ranges.filter((dp) => dp.street === street);
    if (onStreet.length > 0) return onStreet[0];
  }

  return ranges[0];
}

const HERO_ONLY_MODES: ReadonlySet<FillMode> = new Set(['action', 'ev-loss']);

function SeatToolbar({
  fillMode,
  isHeroGrid,
  onFillModeChange,
}: {
  fillMode: FillMode;
  isHeroGrid: boolean;
  onFillModeChange: (m: FillMode) => void;
}) {
  const modes: { v: FillMode; label: string }[] = [
    { v: 'range', label: 'range' },
    { v: 'action', label: 'action' },
    { v: 'ev-loss', label: 'ev-loss' },
    { v: 'equity', label: 'equity' },
  ];
  return (
    <div className="range-toolbar">
      <span className="cap" style={{ fontSize: 9 }}>fill</span>
      <div className="range-fill-seg">
        {modes.map((m) => {
          const heroOnly = m.v === 'action' || m.v === 'ev-loss';
          const disabled = heroOnly && !isHeroGrid;
          return (
            <button
              key={m.v}
              className={'range-fill-btn' + (fillMode === m.v ? ' active' : '')}
              onClick={() => !disabled && onFillModeChange(m.v)}
              type="button"
              disabled={disabled}
            >
              {m.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default function PostflopRangePanel({
  handId,
  ranges,
  currentStep,
  exploitabilityPct,
  onSolve,
}: PostflopRangePanelProps) {
  const [fillModeOop, setFillModeOop] = useState<FillMode>('action');
  const [fillModeIp, setFillModeIp] = useState<FillMode>('action');
  const [selected, setSelected] = useState<Selected | null>(null);
  const [solving, setSolving] = useState(false);
  const [solveProgress, setSolveProgress] = useState<number>(0);
  const [solveError, setSolveError] = useState<string | null>(null);
  const closeStreamRef = useRef<(() => void) | null>(null);
  const [cachedExploitability, setCachedExploitability] = useState<number | null>(null);

  useEffect(() => {
    return () => {
      closeStreamRef.current?.();
    };
  }, []);

  if (ranges === undefined) {
    return (
      <div className="section">
        <div className="section-head">
          <div className="section-head-l">
            <span className="section-head-title">RANGE VIEWER</span>
            <span className="section-head-meta">· loading…</span>
          </div>
        </div>
        <div className="section-body muted">loading ranges…</div>
      </div>
    );
  }

  if (ranges.length === 0) {
    async function handleSolve() {
      setSolving(true);
      setSolveError(null);
      setSolveProgress(0);
      try {
        const { job_id } = await postSolveRanges(handId);
        const close = subscribeRangeSolveJob(handId, job_id, {
          onProgress: (data: unknown) => {
            const d = data as Record<string, unknown>;
            if (typeof d.pct === 'number') setSolveProgress(d.pct);
          },
          onDone: (data: unknown) => {
            const d = data as Record<string, unknown>;
            const expPct =
              typeof d.exploitability_pct === 'number'
                ? d.exploitability_pct
                : DEFAULT_EXPLOITABILITY;
            setCachedExploitability(expPct);
            setSolving(false);
            closeStreamRef.current = null;
            onSolve();
          },
          onError: (msg: string) => {
            setSolveError(msg);
            setSolving(false);
            closeStreamRef.current = null;
          },
        });
        closeStreamRef.current = close;
      } catch (err) {
        setSolveError(String(err));
        setSolving(false);
      }
    }

    return (
      <div className="section">
        <div className="section-head">
          <div className="section-head-l">
            <span className="section-head-title">RANGE VIEWER</span>
            <span className="section-head-meta">· unsolved</span>
          </div>
        </div>
        <div className="section-body" style={{ padding: 16 }}>
          {solveError && (
            <div className="muted" style={{ marginBottom: 8, color: '#ef4444' }}>
              Error: {solveError}
            </div>
          )}
          {solving ? (
            <div>
              <div className="muted" style={{ marginBottom: 6 }}>Solving…</div>
              <div style={{ height: 2, background: '#222', width: '100%' }}>
                <div
                  style={{
                    height: 2,
                    background: '#f59e0b',
                    width: `${Math.max(2, solveProgress)}%`,
                    transition: 'width 0.3s',
                  }}
                />
              </div>
            </div>
          ) : (
            <button
              className="btn btn-primary"
              type="button"
              onClick={handleSolve}
            >
              Solve ranges
            </button>
          )}
        </div>
      </div>
    );
  }

  const dp = resolveCurrentDp(ranges, currentStep);

  if (!dp) {
    return (
      <div className="section">
        <div className="section-head">
          <div className="section-head-l">
            <span className="section-head-title">RANGE VIEWER</span>
          </div>
        </div>
        <div className="section-body muted">step to a hero flop/turn/river decision</div>
      </div>
    );
  }

  if (dp.narrowing === 'multiway_hu_unsupported') {
    return (
      <div className="section">
        <div className="section-head">
          <div className="section-head-l">
            <span className="section-head-title">RANGE VIEWER</span>
            <span className="section-head-meta">· {dp.street}</span>
          </div>
        </div>
        <div className="section-body muted" style={{ padding: 16 }}>
          narrowing unavailable (multiway)
        </div>
      </div>
    );
  }

  if (dp.narrowing === 'nav_failed') {
    return (
      <div className="section">
        <div className="section-head">
          <div className="section-head-l">
            <span className="section-head-title">RANGE VIEWER</span>
            <span className="section-head-meta">· {dp.street}</span>
          </div>
        </div>
        <div className="section-body" style={{ padding: 16, color: '#ef4444' }}>
          ERROR: navigation failed
        </div>
      </div>
    );
  }

  const oopSeat = dp.oop!;
  const ipSeat = dp.ip!;
  const isOopHero = dp.hero_seat === 0;
  const resolvedExploitability =
    cachedExploitability ?? exploitabilityPct ?? DEFAULT_EXPLOITABILITY;

  // Villain has no strategy → hero-only modes (action/ev-loss) coerce to range for that grid.
  const oopMode: FillMode = isOopHero || !HERO_ONLY_MODES.has(fillModeOop) ? fillModeOop : 'range';
  const ipMode: FillMode = !isOopHero || !HERO_ONLY_MODES.has(fillModeIp) ? fillModeIp : 'range';

  const oopModel = buildGridModel(oopSeat, oopMode, {
    isHero: isOopHero,
    exploitabilityPct: resolvedExploitability,
  });
  const ipModel = buildGridModel(ipSeat, ipMode, {
    isHero: !isOopHero,
    exploitabilityPct: resolvedExploitability,
  });

  const selectedSeatPayload: RangeSeatPayload | null =
    selected?.seat === 'oop' ? oopSeat : selected?.seat === 'ip' ? ipSeat : null;
  const selectedIsHero =
    selected?.seat === 'oop' ? isOopHero : selected?.seat === 'ip' ? !isOopHero : false;

  return (
    <div className="section">
      <div className="section-head">
        <div className="section-head-l">
          <span className="section-head-title">RANGE VIEWER</span>
          <span className="section-head-meta">· {dp.street} · dual seat</span>
        </div>
      </div>
      <div className="section-body" style={{ display: 'flex', gap: 16, flexWrap: 'wrap', padding: 14 }}>
        <div style={{ flex: '1 1 300px' }}>
          <div className="cap" style={{ marginBottom: 4 }}>OOP{isOopHero ? ' (hero)' : ' (villain)'}</div>
          <SeatToolbar
            fillMode={oopMode}
            isHeroGrid={isOopHero}
            onFillModeChange={setFillModeOop}
          />
          <StrategyGrid
            model={oopModel}
            onCellClick={(cls) => setSelected({ seat: 'oop', handClass: cls })}
          />
          <ActionLegend legend={oopModel.legend} />
        </div>
        <div style={{ flex: '1 1 300px' }}>
          <div className="cap" style={{ marginBottom: 4 }}>IP{!isOopHero ? ' (hero)' : ' (villain)'}</div>
          <SeatToolbar
            fillMode={ipMode}
            isHeroGrid={!isOopHero}
            onFillModeChange={setFillModeIp}
          />
          <StrategyGrid
            model={ipModel}
            onCellClick={(cls) => setSelected({ seat: 'ip', handClass: cls })}
          />
          <ActionLegend legend={ipModel.legend} />
        </div>
      </div>
      {selected && selectedSeatPayload && (
        <ComboBreakdownPanel
          handClass={selected.handClass}
          combos={selectedSeatPayload.combos}
          weights={selectedSeatPayload.weights}
          equity={selectedSeatPayload.equity}
          strategy={selectedIsHero ? selectedSeatPayload.strategy : undefined}
          ev_detail={selectedIsHero ? selectedSeatPayload.ev_detail : undefined}
          actions={selectedIsHero ? selectedSeatPayload.actions : undefined}
          hero_action={selectedIsHero ? (selectedSeatPayload.hero_action ?? null) : undefined}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
