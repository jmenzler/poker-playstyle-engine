import ActionTape, { type ActionTapeConfig } from '@/components/ActionTape';
import type { EncodeForm, EncodeResult } from '@/lib/probeEncoder';
import PlayingCard from '../PlayingCard';
import { SEAT_SUB } from '@/lib/probeEncoder';
import type { TapeStep } from '@/lib/spotBuilder';

const POSITION_ORDER = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB'];
const POSTFLOP_ORDER = ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN'];
const RAISE_VERBS_PREFLOP = new Set(['open', '3bet', '4bet', '5bet']);

function classifyHole(heroHole: Array<string | null>): string | null {
  const cards = heroHole.filter(Boolean) as string[];
  if (cards.length !== 2) return null;
  const [a, b] = cards;
  const ra = a[0], rb = b[0];
  const sa = a[1], sb = b[1];
  if (ra === rb) return `${ra}${rb}`;
  const order = 'AKQJT98765432';
  const hi = order.indexOf(ra) < order.indexOf(rb) ? ra : rb;
  const lo = hi === ra ? rb : ra;
  return `${hi}${lo}${sa === sb ? 's' : 'o'}`;
}

function legalPreflopVerbs(tape: EncodeForm['preflop']): string[] {
  if (!tape || tape.length === 0) return ['open', 'limp', 'fold'];
  const has5 = tape.some((t) => t.verb === '5bet');
  const has4 = tape.some((t) => t.verb === '4bet');
  const has3 = tape.some((t) => t.verb === '3bet');
  const hasOpen = tape.some((t) => t.verb === 'open');
  if (has5) return ['call', 'fold'];
  if (has4) return ['call', '5bet', 'fold'];
  if (has3) return ['call', '4bet', 'fold'];
  if (hasOpen) return ['call', '3bet', 'fold'];
  return ['limp', 'open', 'fold'];
}

function legalPostflopVerbs(streetTape: Array<{ verb: string }>): string[] {
  if (!streetTape || streetTape.length === 0) return ['check', 'bet', 'fold'];
  const last = streetTape[streetTape.length - 1];
  if (last.verb === 'bet' || last.verb === 'raise') return ['call', 'raise', 'fold'];
  return ['check', 'bet', 'fold'];
}

function inPotPositions(form: EncodeForm): string[] {
  const tape = form.preflop;
  if (!tape || tape.length === 0) return POSTFLOP_ORDER;
  const folded = new Set<string>();
  const acted = new Set<string>();
  for (const t of tape) {
    if (t.verb === 'fold') folded.add(t.pos);
    else acted.add(t.pos);
  }
  if (form.heroPos && !folded.has(form.heroPos)) acted.add(form.heroPos);
  return POSTFLOP_ORDER.filter((p) => acted.has(p) && !folded.has(p));
}

function nextPreflopActor(tape: EncodeForm['preflop']): string {
  if (tape.length === 0) return 'UTG';
  const folded = new Set(tape.filter((t) => t.verb === 'fold').map((t) => t.pos));
  const lastIdx = POSITION_ORDER.indexOf(tape[tape.length - 1].pos);
  for (let i = 1; i <= POSITION_ORDER.length; i++) {
    const cand = POSITION_ORDER[(lastIdx + i) % POSITION_ORDER.length];
    if (!folded.has(cand)) return cand;
  }
  return 'UTG';
}

type ActiveStep = 'preflop' | 'flop' | 'turn' | 'river';

interface StepEditorPanelProps {
  activeStep: ActiveStep;
  setActiveStep: (s: ActiveStep) => void;
  form: EncodeForm;
  setForm: (form: EncodeForm | ((f: EncodeForm) => EncodeForm)) => void;
  reachedMap: Record<string, boolean>;
  enc?: EncodeResult;
  onOpenRangeFor?: (stepIdx: number) => void;
}

const STEP_ORDER: ActiveStep[] = ['preflop', 'flop', 'turn', 'river'];

export default function StepEditorPanel({
  activeStep,
  setActiveStep,
  form,
  setForm,
  reachedMap,
  enc,
  onOpenRangeFor,
}: StepEditorPanelProps) {
  const errMap = enc
    ? Object.fromEntries(enc.structural.map((e) => [e.field, e.msg]))
    : {};
  const stepIdx = STEP_ORDER.indexOf(activeStep);
  const goPrev = stepIdx > 0 ? () => setActiveStep(STEP_ORDER[stepIdx - 1]) : null;
  const goNext = stepIdx < STEP_ORDER.length - 1 ? () => setActiveStep(STEP_ORDER[stepIdx + 1]) : null;

  const villainSeat = (() => {
    const aggressors = form.preflop.filter((t) => RAISE_VERBS_PREFLOP.has(t.verb));
    const lastAgg = aggressors.length > 0 ? aggressors[aggressors.length - 1].pos : null;
    if (lastAgg && lastAgg !== form.heroPos) return lastAgg;
    const inPot = inPotPositions(form);
    return inPot.find((p) => p !== form.heroPos) || 'BB';
  })();

  function renderBody() {
    if (activeStep === 'preflop') {
      const tapeSteps: TapeStep[] = form.preflop.map((s) => ({ pos: s.pos, action: s.verb, amount: s.size }));
      const nextActor = nextPreflopActor(form.preflop);
      const legalVerbs = legalPreflopVerbs(form.preflop);
      const config: ActionTapeConfig = {
        street: 'preflop',
        emitsPosition: true,
        verbSet: legalVerbs,
        heroPos: form.heroPos || '',
        defaultPos: nextActor,
        positionOptions: POSITION_ORDER,
      };

      return (
        <div>
          <div className="hero-dd-row" style={{ marginBottom: 10, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <div className="hero-dd-pos-pills">
              {POSITION_ORDER.map((p) => (
                <button
                  key={p}
                  className={'hero-dd-pos-pill' + (form.heroPos === p ? ' active' : '')}
                  onClick={() => setForm({ ...form, heroPos: p })}
                  type="button"
                >{p}</button>
              ))}
            </div>
            {form.heroPos && (
              <span className="dim" style={{ fontSize: 10, marginLeft: 'auto' }}>
                {SEAT_SUB[form.heroPos]}
              </span>
            )}
            <span className="dim" style={{ fontSize: 10, marginLeft: form.heroPos ? 0 : 'auto' }}>
              {form.heroHole.every(Boolean)
                ? <>holding <b style={{ color: 'var(--color-accent)' }}>{classifyHole(form.heroHole)}</b></>
                : 'no hole cards picked yet'}
            </span>
          </div>
          {errMap.positions && (
            <div className="step-error">{errMap.positions}</div>
          )}
          <ActionTape
            config={config}
            steps={tapeSteps}
            onChange={(newSteps) => {
              setForm({
                ...form,
                preflop: newSteps.map((s) => ({
                  pos: s.pos || nextActor,
                  verb: s.action,
                  size: s.amount,
                })),
              });
            }}
            onOpenRangeFor={onOpenRangeFor}
          />
        </div>
      );
    }

    if (activeStep === 'flop' || activeStep === 'turn' || activeStep === 'river') {
      if (!reachedMap[activeStep]) {
        return (
          <div className="step-unreached">
            <div className="step-unreached-title">{activeStep} not reached yet</div>
            <div className="step-unreached-body">finish the earlier street to build this one</div>
          </div>
        );
      }
      const slotsNeeded = { flop: 3, turn: 1, river: 1 }[activeStep];
      const streetData = form[activeStep];
      const inPot = inPotPositions(form);
      const legalVerbs = legalPostflopVerbs(streetData.actions);

      const tapeSteps: TapeStep[] = streetData.actions.map((s) => ({
        pos: s.pos,
        action: s.verb,
        amount: s.size,
      }));

      const config: ActionTapeConfig = {
        street: activeStep,
        emitsPosition: true,
        verbSet: legalVerbs,
        heroPos: form.heroPos || '',
        villainSeat,
        positionOptions: inPot,
        defaultPos: inPot[0],
      };

      const usedCardSet = new Set<string>([
        ...(form.heroHole.filter(Boolean) as string[]),
        ...(form.flop.board.filter(Boolean) as string[]),
        ...(form.turn.board.filter(Boolean) as string[]),
        ...(form.river.board.filter(Boolean) as string[]),
      ]);
      const boardCards = (streetData.board.filter(Boolean) as string[]);
      for (const c of boardCards) usedCardSet.delete(c);

      return (
        <div>
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 10, color: 'var(--color-fg-caption)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
              {activeStep} board · {boardCards.length}/{slotsNeeded} cards
            </div>
            <div className="slot-row board-row" style={{ marginBottom: 8 }}>
              {Array.from({ length: slotsNeeded }).map((_, i) => {
                const c = streetData.board[i];
                return (
                  <div
                    key={i}
                    className={'slot' + (c ? ' filled' : ' required')}
                  >
                    {c ? (
                      <>
                        <PlayingCard card={c} />
                        <span
                          className="slot-x"
                          title="clear"
                          onClick={(e) => {
                            e.stopPropagation();
                            const newBoard = [...streetData.board];
                            newBoard[i] = null;
                            setForm({ ...form, [activeStep]: { ...streetData, board: newBoard } });
                          }}
                        >×</span>
                      </>
                    ) : (
                      <div className="slot-empty">
                        <span className="slot-empty-glyph">+</span>
                        <span className="slot-empty-label">card</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            <div style={{ fontSize: 10, color: 'var(--color-fg-dim)' }}>
              {reachedMap[activeStep] ? '' : 'not reached yet'}
            </div>
            {errMap[`${activeStep}-board`] && (
              <div className="step-error">{errMap[`${activeStep}-board`]}</div>
            )}
            {errMap.duplicate && (
              <div className="step-error">{errMap.duplicate}</div>
            )}
          </div>
          <ActionTape
            config={config}
            steps={tapeSteps}
            onChange={(newSteps) => {
              setForm({
                ...form,
                [activeStep]: {
                  ...streetData,
                  actions: newSteps.map((s) => ({
                    pos: s.pos,
                    verb: s.action,
                    size: s.amount,
                  })),
                },
              });
            }}
          />
        </div>
      );
    }

    return null;
  }

  const summaryText = (() => {
    if (activeStep === 'preflop') {
      if (form.preflop.length === 0) {
        return form.heroPos
          ? `hero @ ${form.heroPos} · build tape below`
          : 'pick hero seat · build the tree';
      }
      const heroHand = form.heroHole.every(Boolean) ? classifyHole(form.heroHole) : null;
      return `${form.preflop.length} step${form.preflop.length === 1 ? '' : 's'}${heroHand ? ` · hero ${heroHand}` : ''}`;
    }
    const data = form[activeStep];
    const boardFilled = data.board.filter(Boolean).length;
    const slotsNeeded = { flop: 3, turn: 1, river: 1 }[activeStep];
    if (boardFilled > 0) {
      return `${boardFilled}/${slotsNeeded} cards · ${data.actions.length} action${data.actions.length === 1 ? '' : 's'}`;
    }
    return reachedMap[activeStep]
      ? `${slotsNeeded} card${slotsNeeded === 1 ? '' : 's'} required`
      : 'not reached yet · earlier street incomplete';
  })();

  return (
    <div className="step-editor">
      <div className="step-editor-head">
        <div className="step-editor-head-l">
          <span className="step-editor-head-target">{activeStep}</span>
          <span className="dim">·</span>
          <span className="step-editor-head-summary">{summaryText}</span>
        </div>
        <div className="step-editor-head-r">
          <span className="dim">step {stepIdx + 1} of {STEP_ORDER.length}</span>
          <div className="step-editor-nav">
            <button className="step-editor-nav-btn" onClick={goPrev ?? undefined} disabled={!goPrev} type="button">‹ prev</button>
            <button className="step-editor-nav-btn" onClick={goNext ?? undefined} disabled={!goNext} type="button">next ›</button>
          </div>
        </div>
      </div>
      <div className="step-editor-body">
        {renderBody()}
      </div>
    </div>
  );
}
