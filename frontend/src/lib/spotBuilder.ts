// Guided spot-builder selections -> canonical action_sequence POS:verb tokens +
// spot dict + structural validation. Raise stems drive pot_type; last raiser -> hero_pos_rel.

export type Position = 'UTG' | 'MP' | 'CO' | 'BTN' | 'SB' | 'BB';
export type Street = 'preflop' | 'flop' | 'turn' | 'river';
export type PreflopAction = 'limp' | 'open' | '3bet' | '4bet';

export interface TapeStep {
  pos?: string;
  action: string;
  amount?: number;
}

export interface BuilderState {
  street: Street;
  heroPosition: Position;
  preflopAction: PreflopAction;
  raiserPosition: Position | '';
  heroHole: string[];
  board: string[];
  postflopTape?: TapeStep[];
}

export interface SpotValidation {
  ok: boolean;
  errors: string[];
}

export const POSITIONS: Position[] = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB'];

export const BOARD_SLOTS: Record<Street, number> = {
  preflop: 0,
  flop: 3,
  turn: 4,
  river: 5,
};

const CARD_RE = /^[2-9TJQKA][cdhs]$/;

// Deterministic seat for an unknown intermediate raiser. Safe because only the raise
// COUNT and the LAST raiser seat affect the cluster_key bucket, not intermediate seats.
function placeholderSeat(exclude: Array<Position | ''>): Position {
  const seat = POSITIONS.find((p) => !exclude.includes(p));
  return seat ?? 'UTG';
}

export function buildActionSequence(state: BuilderState): string[] {
  const { preflopAction, raiserPosition, heroPosition } = state;
  if (preflopAction === 'limp' || raiserPosition === '') return [];

  const raiser = raiserPosition;
  if (preflopAction === 'open') return [`${raiser}:open`];

  const opener = placeholderSeat([heroPosition, raiser]);
  if (preflopAction === '3bet') return [`${opener}:open`, `${raiser}:3bet`];

  const threeBettor = placeholderSeat([heroPosition, raiser, opener]);
  return [`${opener}:open`, `${threeBettor}:3bet`, `${raiser}:4bet`];
}

function verbToEncoderStem(action: string, amount?: number): string {
  if (action === 'check') return 'check';
  if (action === 'call') return 'call';
  if (action === 'fold') return 'fold';
  if (action === 'bet') return amount != null ? `bet_${amount}` : 'bet';
  if (action === 'raise') return amount != null ? `raise_${amount}x` : 'raise';
  return action;
}

export function villainSeatFromPreflop(state: BuilderState): string {
  const { preflopAction, raiserPosition, heroPosition } = state;
  if (preflopAction !== 'limp' && raiserPosition && raiserPosition !== heroPosition) {
    return raiserPosition;
  }
  const fallback = POSITIONS.find((p) => p !== heroPosition);
  return fallback ?? 'BB';
}

export function buildPostflopTokens(state: BuilderState): string[] {
  const tape = state.postflopTape;
  if (!tape || tape.length === 0) return [];
  const villainSeat = villainSeatFromPreflop(state);
  return tape.map((step) => {
    const seat = step.pos ?? (step.action === 'check' || step.action === 'call' || step.action === 'fold' ? villainSeat : state.heroPosition);
    const verb = verbToEncoderStem(step.action, step.amount);
    return `${seat}:${verb}`;
  });
}

export function buildSpotDict(state: BuilderState): Record<string, unknown> {
  const preflopSeq = buildActionSequence(state);
  const postflopSeq = buildPostflopTokens(state);
  return {
    street: state.street,
    hero_position: state.heroPosition,
    hero_hole: state.heroHole,
    board: state.board,
    action_sequence: [...preflopSeq, ...postflopSeq],
  };
}

export function usedCards(state: BuilderState): Set<string> {
  return new Set<string>([...state.heroHole, ...state.board]);
}

export function validateSpot(state: BuilderState): SpotValidation {
  const errors: string[] = [];
  const all = [...state.heroHole, ...state.board];

  const formatFailed = new Set<string>();
  for (const card of all) {
    if (!CARD_RE.test(card)) {
      errors.push(`illegal card: ${card || '(empty)'}`);
      formatFailed.add(card);
    }
  }

  if (state.heroHole.length !== 2) {
    errors.push(`hero hole must have 2 cards, got ${state.heroHole.length}`);
  }

  const expectedBoard = BOARD_SLOTS[state.street];
  if (state.board.length !== expectedBoard) {
    errors.push(`${state.street} board must have ${expectedBoard} cards, got ${state.board.length}`);
  }

  if (
    state.preflopAction !== 'limp' &&
    state.raiserPosition !== '' &&
    state.raiserPosition === state.heroPosition
  ) {
    errors.push('hero cannot be the last raiser (hero position equals raiser position)');
  }

  const seen = new Set<string>();
  for (const card of all) {
    if (formatFailed.has(card)) { seen.add(card); continue; }
    if (seen.has(card)) errors.push(`duplicate card: ${card}`);
    seen.add(card);
  }

  return { ok: errors.length === 0, errors };
}
