import { useEffect, useState } from 'react';
import '../routes/replayer-felt.css';
import PlayingCard from './PlayingCard';
import GapActionButtons from './GapActionButtons';

export interface ReplayStep {
  step_idx: number;
  street: string;
  board: string[];
  pot: number | null;
  // Pre-action pot; a decision view shows this so the pot doesn't leak the action.
  pot_before?: number | null;
  big_blind: number | null;
  action_taken?: string;
  seat_map?: Record<string, { name: string; stack: number }> | null;
  // God-view: every seat's hole cards, known on every step (harvested across
  // the per-player decision points). position -> [card, card].
  seat_cards?: Record<string, string[]> | null;
  button_seat?: number | null;
  actor?: string | null;
  bet_amount?: number | null;
  // Positions folded up to and including this step.
  folded?: string[] | null;
  // Legacy HM-path fields (still emitted by the HM branch).
  hero_hole?: string[];
  hero_seat?: number | null;
  action_sequence?: string[];
  street_committed?: Record<string, number> | null;
  shown_cards?: Record<string, string[]> | null;
  decision_id?: string | null;
}

export interface InlineReplayerProps {
  steps: ReplayStep[];
  rx?: number;
  ry?: number;
  // Position label (e.g. "BTN", "SB") to anchor at the bottom of the felt. When
  // set, that seat sits bottom-center for the whole hand (god-view "hero").
  heroPosition?: string | null;
  onStepChange?: (step: ReplayStep) => void;
  gapDecisionId?: string | null;
  gapLegalActions?: string[];
  isMultiway?: boolean;
  fastMode?: boolean;
  onResolveGap?: (actionDist: Record<string, number>) => void;
  onSendToSolver?: () => void;
  resolving?: boolean;
  resolveError?: string | null;
  prevNodeId?: string | null;
  clusterKey?: string;
  hideVillainCards?: boolean;
}

const N_SLOTS = 6;
const BET_F = 0.5;
const BASE_ANGLE = Math.PI / 2;

function slotIndex(seatNo: number): number {
  return (seatNo - 1) % N_SLOTS;
}

function seatPos(slot: number, heroSlot: number, rx: number, ry: number): React.CSSProperties {
  const theta = BASE_ANGLE + ((slot - heroSlot) / N_SLOTS) * 2 * Math.PI;
  return {
    left: `calc(50% + ${rx * Math.cos(theta)}px)`,
    top: `calc(50% + ${ry * Math.sin(theta)}px)`,
    transform: 'translate(-50%, -50%)',
  };
}

function Card({ card }: { card?: string }) {
  return <PlayingCard card={card} className="felt-card" style={{ width: 56, height: 'auto' }} />;
}

export default function InlineReplayer({
  steps,
  rx = 620,
  ry = 360,
  heroPosition = null,
  onStepChange,
  gapDecisionId = null,
  gapLegalActions,
  isMultiway = false,
  fastMode = false,
  onResolveGap,
  onSendToSolver,
  resolving = false,
  resolveError = null,
  prevNodeId = null,
  clusterKey,
  hideVillainCards = false,
}: InlineReplayerProps) {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  // gap-resolver truncates the line so the decision is the last step; land there
  // instead of step 1 so the author isn't forced to click through the whole lead-up.
  const decisionIdx = gapDecisionId
    ? steps.findIndex((s) => s.decision_id === gapDecisionId)
    : -1;

  useEffect(() => {
    setIdx(decisionIdx >= 0 ? decisionIdx : 0);
    setPlaying(false);
  }, [steps]);

  useEffect(() => {
    if (!playing) return;
    const t = setTimeout(() => {
      if (idx < steps.length - 1) setIdx(idx + 1);
      else setPlaying(false);
    }, 1200);
    return () => clearTimeout(t);
  }, [playing, idx, steps.length]);

  useEffect(() => {
    if (steps[idx]) onStepChange?.(steps[idx]);
  }, [idx]);

  if (!steps.length) {
    return <div className="text-fg-muted text-[12px] p-2">—</div>;
  }

  const current = steps[idx];
  const board = current.board ?? [];
  const seatMap = current.seat_map ?? {};
  const seatNos = Object.keys(seatMap)
    .map(Number)
    .sort((a, b) => a - b);
  const hasSeats = seatNos.length > 0;

  // Hero seat: explicit position label (gap-resolver/god-view) wins, else the
  // HM-path hero_seat. SIM steps carry no hero_seat, so the label is the only
  // signal that hides villain cards + highlights the seat-to-decide.
  const heroPosSeatNo = heroPosition
    ? seatNos.find((n) => seatMap[String(n)]?.name === heroPosition)
    : undefined;
  const heroSeatNo = heroPosSeatNo ?? current.hero_seat ?? null;

  // God-view: seats are fixed for the whole hand; the acting seat + folded set
  // change per step. SIM steps carry `seat_cards` (all known) + `folded`.
  const seatCardsMap = current.seat_cards ?? {};
  const folded = new Set<string>(current.folded ?? []);
  const shownCards = current.shown_cards ?? {};
  const atShowdown = idx === steps.length - 1;

  const bb = current.big_blind && current.big_blind > 0 ? current.big_blind : null;
  const inBB = (v: number): string => (bb ? `${(v / bb).toFixed(1)}bb` : String(v));

  function seatCards(name: string, isHero: boolean, seatNo: number): (string | undefined)[] | null {
    if (folded.has(name)) return null;
    // Bias-free authoring (gap-resolver): show only the hero's cards, villains face-down.
    // Fail open if the hero seat is unknown, so we never blank the whole table.
    if (hideVillainCards && !isHero && heroSeatNo != null) return [undefined, undefined];
    // god-view per-seat cards (SIM), else hero_hole/showdown (HM). hero_hole belongs to
    // the card owner (hero_seat), not the anchor — which the gap-resolver overrides.
    if (seatCardsMap[name]?.length) return seatCardsMap[name];
    if (current.hero_seat != null && seatNo === current.hero_seat && current.hero_hole?.length)
      return current.hero_hole;
    if (atShowdown && shownCards[name]?.length) return shownCards[name];
    return [undefined, undefined];
  }

  const controls = steps.length > 1 ? (
    <div className="flex gap-2 items-center">
      <button onClick={() => setIdx(Math.max(0, idx - 1))}>[ ⏮ ]</button>
      <button onClick={() => setPlaying(!playing)}>[ {playing ? '⏸' : '⏯'} ]</button>
      <button onClick={() => setIdx(Math.min(steps.length - 1, idx + 1))}>[ ⏭ ]</button>
      {decisionIdx >= 0 && idx !== decisionIdx && (
        <button onClick={() => setIdx(decisionIdx)} className="text-accent">[ ▸ decision ]</button>
      )}
      <span className="text-fg-muted ml-4">
        step {idx + 1} / {steps.length}
      </span>
    </div>
  ) : null;

  if (!hasSeats) {
    return (
      <div data-felt className="text-fg font-mono space-y-4">
        <div className="text-[10px] text-fg-caption">
          seat data unavailable — showing board + hole cards
        </div>
        <div className="text-[12px] text-fg-muted">
          street: {current.street ?? '—'}
        </div>
        <div>
          <div className="text-[10px] text-fg-caption">BOARD</div>
          <div className="flex gap-2 mt-1">
            {board.length === 0 ? (
              <div className="text-fg-muted text-[12px]">—</div>
            ) : (
              board.map((card, i) => <Card key={i} card={card} />)
            )}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-fg-caption">HOLE CARDS (HERO)</div>
          <div className="flex gap-2 mt-1">
            {(current.hero_hole ?? []).length === 0 ? (
              <div className="text-fg-muted text-[12px]">—</div>
            ) : (
              (current.hero_hole ?? []).map((card, i) => <Card key={i} card={card} />)
            )}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-fg-caption">ACTION</div>
          <div className="text-[16px] text-accent">{current.action_taken ?? '—'}</div>
        </div>
        {controls}
      </div>
    );
  }

  // Anchor the felt on a FIXED seat for the whole hand so the table does not
  // rotate step-to-step. Priority: hero (position label or HM hero_seat), then
  // button, then first seat.
  const anchorSeatNo = heroSeatNo ?? current.button_seat ?? seatNos[0];
  const heroSlot = slotIndex(anchorSeatNo);

  return (
    <div data-felt className="text-fg font-mono space-y-4">
      <div className="flex justify-center py-4">
        <div className="felt-table" style={{ width: rx * 2, height: ry * 2 }}>
          <div className="felt-pot">
            <span className="felt-chip" />
            <span>{inBB(current.pot ?? 0)}</span>
          </div>

          <div
            style={{
              position: 'absolute',
              left: '50%',
              top: 'calc(50% - 32px)',
              transform: 'translate(-50%, -50%)',
              display: 'flex',
              gap: 4,
            }}
          >
            {board.map((card, i) => (
              <Card key={i} card={card} />
            ))}
          </div>

          {seatNos.map((seatNo) => {
            const seat = seatMap[String(seatNo)];
            const slot = slotIndex(seatNo);
            const isActor = !!current.actor && seat.name === current.actor;
            const isHero = heroSeatNo != null && seatNo === heroSeatNo;
            const pos = seatPos(slot, heroSlot, rx, ry);
            const cards = seatCards(seat.name, isHero, seatNo);
            return (
              <div key={seatNo} className="felt-seat" style={pos}>
                {isActor && current.action_taken && (
                  <div className="felt-pill-banner">
                    {current.action_taken?.split(': ').slice(1).join(': ') || current.action_taken}
                    {current.bet_amount != null ? ` · ${inBB(current.bet_amount)}` : ''}
                  </div>
                )}
                {cards && (
                  <div className="felt-seat-cards">
                    {cards.map((card, i) => (
                      <Card key={i} card={card} />
                    ))}
                  </div>
                )}
                <div
                  className={`felt-seat-box${isActor ? ' felt-seat-box--actor' : ''}${
                    cards === null ? ' felt-seat-box--folded' : ''
                  }${isHero && gapDecisionId && current.decision_id === gapDecisionId ? ' felt-seat-box--gap-hero' : ''}`}
                >
                  <div className="text-[11px]">
                    {seat.name}
                    {isHero && <span className="text-accent"> (Hero)</span>}
                  </div>
                  <div className="text-[10px] text-fg-muted flex items-center justify-center gap-1">
                    <span className="felt-chip" />
                    {inBB(seat.stack)}
                  </div>
                </div>
              </div>
            );
          })}

          {current.button_seat != null && (
            <div
              className="felt-dealer"
              style={seatPos(slotIndex(current.button_seat), heroSlot, rx - 24, ry - 14)}
            >
              D
            </div>
          )}

          {seatNos.map((seatNo) => {
            const seat = seatMap[String(seatNo)];
            const amt = current.street_committed?.[seat.name];
            if (!amt) return null;
            return (
              <div
                key={`bet-${seatNo}`}
                className="felt-bet"
                style={seatPos(slotIndex(seatNo), heroSlot, rx * BET_F, ry * BET_F)}
              >
                <span className="felt-chip" />
                {inBB(amt)}
              </div>
            );
          })}

          {idx > 0 &&
            steps[idx - 1].street !== current.street &&
            Object.keys(steps[idx - 1].street_committed ?? {}).map((name) => {
              const seatNo = seatNos.find((s) => seatMap[String(s)].name === name);
              if (seatNo === undefined) return null;
              const theta =
                BASE_ANGLE + ((slotIndex(seatNo) - heroSlot) / N_SLOTS) * 2 * Math.PI;
              return (
                <div
                  key={`sweep-${idx}-${name}`}
                  className="chip-slide"
                  style={
                    {
                      '--from-x': `${rx * BET_F * Math.cos(theta)}px`,
                      '--from-y': `${ry * BET_F * Math.sin(theta)}px`,
                    } as React.CSSProperties
                  }
                />
              );
            })}
        </div>
      </div>

      {gapDecisionId &&
        current.decision_id === gapDecisionId &&
        (gapLegalActions?.length ?? 0) > 0 &&
        onResolveGap && (
          <GapActionButtons
            legalActions={gapLegalActions!}
            isMultiway={isMultiway}
            fastMode={fastMode}
            onResolveGap={onResolveGap}
            onSendToSolver={onSendToSolver}
            resolving={resolving}
            error={resolveError}
            prevNodeId={prevNodeId}
            clusterKey={clusterKey}
          />
        )}

      {controls}
    </div>
  );
}
