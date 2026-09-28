/**
 * Client-side PREVIEW encoder for instant keystroke feedback ONLY.
 * The authoritative query cluster_key comes from server POST /api/probe/encode.
 * Never the source of truth (encode-equivalence).
 */

const POSITION_ORDER = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB'];
const POSTFLOP_ORDER = ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN'];
const RAISE_VERBS = new Set(['open', '3bet', '4bet', '5bet']);

export const SEAT_SUB: Record<string, string> = {
  UTG: 'early',
  MP: 'early',
  CO: 'late',
  BTN: 'button',
  SB: 'blind',
  BB: 'blind',
};

export interface EncodeForm {
  heroPos: string | null;
  preflop: Array<{ pos: string; verb: string; size?: number }>;
  flop: { board: Array<string | null>; actions: Array<{ pos?: string; verb: string; size?: number }> };
  turn: { board: Array<string | null>; actions: Array<{ pos?: string; verb: string; size?: number }> };
  river: { board: Array<string | null>; actions: Array<{ pos?: string; verb: string; size?: number }> };
  heroHole: Array<string | null>;
}

export interface EncodeParts {
  street: string;
  potType: string;
  position: string | null;
  boardClass: string | null;
  streetSeq: string | null;
  holeClass: string | null;
  preflopSeq: string;
  heroPos: string | null;
  raises: number;
}

export interface EncodeResult {
  key: string | null;
  valid: boolean;
  tier1_errors: Array<{ field: string; msg: string }>;
  structural: Array<{ field: string; msg: string }>;
  soft: Array<{ field: string; msg: string }>;
  parts: EncodeParts;
  seq: string | null;
}

function classifyPreflop(tape: EncodeForm['preflop']): { potType: string; seq: string } {
  const has5 = tape.some((t) => t.verb === '5bet');
  const has4 = tape.some((t) => t.verb === '4bet');
  const has3 = tape.some((t) => t.verb === '3bet');
  const hasOpen = tape.some((t) => t.verb === 'open');
  const hasLimp = tape.some((t) => t.verb === 'limp');
  let potType = 'WALK';
  if (has5) potType = '5BP+';
  else if (has4) potType = '4BP';
  else if (has3) potType = '3BP';
  else if (hasOpen) potType = 'SRP';
  else if (hasLimp) potType = 'LIMP';

  const seq = tape.map((t) => {
    const v =
      t.verb === 'open' ? 'o' :
      t.verb === 'limp' ? 'l' :
      t.verb === 'call' ? 'c' :
      t.verb === '3bet' ? '3' :
      t.verb === '4bet' ? '4' :
      t.verb === '5bet' ? '5' : 'f';
    return `${t.pos}.${v}${t.size != null ? t.size : ''}`;
  }).join('-');

  return { potType, seq };
}

function classifyRelativePosition(tape: EncodeForm['preflop'], heroPos: string): string | null {
  if (!heroPos) return null;
  const aggressors = tape.filter((t) => RAISE_VERBS.has(t.verb));
  const lastAgg = aggressors.length > 0 ? aggressors[aggressors.length - 1].pos : null;
  if (!lastAgg) return heroPos === 'SB' || heroPos === 'BB' ? 'OOP' : 'IP';
  const heroIdx = POSTFLOP_ORDER.indexOf(heroPos);
  const aggIdx = POSTFLOP_ORDER.indexOf(lastAgg);
  if (heroPos === lastAgg) {
    const stillIn = tape.filter((t) => t.verb !== 'fold' && t.pos !== heroPos).map((t) => t.pos);
    const others = [...new Set(stillIn)];
    if (others.length === 0) return 'IP';
    const maxOther = Math.max(...others.map((p) => POSTFLOP_ORDER.indexOf(p)));
    return heroIdx > maxOther ? 'IP' : 'OOP';
  }
  return heroIdx > aggIdx ? 'IP' : 'OOP';
}

function classifyBoard(boardCards: Array<string | null>): string | null {
  const cards = boardCards.filter(Boolean) as string[];
  if (cards.length < 3) return null;
  const flop = cards.slice(0, 3);
  const rankMap = '23456789TJQKA';
  const rankVals = flop.map((c) => rankMap.indexOf(c[0]));
  const suits = flop.map((c) => c[1]);
  const sortedR = [...rankVals].sort((a, b) => b - a);
  const counts: Record<number, number> = {};
  for (const r of rankVals) counts[r] = (counts[r] || 0) + 1;
  const max = Math.max(...Object.values(counts));
  let pairing = 'unpaired';
  if (max === 3) pairing = 'trips';
  else if (max === 2) pairing = 'paired';
  const sCounts: Record<string, number> = {};
  for (const s of suits) sCounts[s] = (sCounts[s] || 0) + 1;
  const maxSuit = Math.max(...Object.values(sCounts));
  let suited = 'rainbow';
  if (maxSuit === 3) suited = 'mono';
  else if (maxSuit === 2) suited = 'two-tone';
  const high = sortedR[0];
  let highClass = 'low';
  if (high === 12) highClass = 'Axx';
  else if (high === 11) highClass = 'Kxx';
  else if (high === 10) highClass = 'Qxx';
  else if (high === 9) highClass = 'Jxx';
  else if (high === 8) highClass = 'Txx';
  else if (high >= 5) highClass = 'mid';
  let conn = '';
  if (pairing === 'unpaired') {
    const gaps: number[] = [];
    for (let i = 0; i < sortedR.length - 1; i++) gaps.push(sortedR[i] - sortedR[i + 1]);
    const minGap = Math.min(...gaps);
    if (minGap === 1) conn = '-connected';
    else if (minGap === 2) conn = '-semi';
    else if (sortedR[0] - sortedR[sortedR.length - 1] <= 4) conn = '-dynamic';
    else conn = '-dry';
  }
  if (pairing === 'paired') return `${highClass}-paired-${suited}`;
  if (pairing === 'trips') return `${highClass}-trips`;
  return `${highClass}-${suited}${conn}`;
}

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

function classifyStreetSeq(tape: Array<{ verb: string; size?: number }>): string | null {
  if (!tape || tape.length === 0) return null;
  return tape.map((t) => {
    if (t.verb === 'check') return 'x';
    if (t.verb === 'call') return 'c';
    if (t.verb === 'fold') return 'f';
    if (t.verb === 'bet') return `b${t.size ?? ''}`;
    if (t.verb === 'raise') return `r${t.size ?? ''}`;
    return t.verb;
  }).join('-');
}

function decisionStreet(form: EncodeForm): string {
  const riverBoard = form.river.board.filter(Boolean).length;
  const turnBoard = form.turn.board.filter(Boolean).length;
  const flopBoard = form.flop.board.filter(Boolean).length;
  if (riverBoard > 0 || form.river.actions.length > 0) return 'river';
  if (turnBoard > 0 || form.turn.actions.length > 0) return 'turn';
  if (flopBoard > 0 || form.flop.actions.length > 0) return 'flop';
  return 'preflop';
}

export function encodeSpot(form: EncodeForm): EncodeResult {
  const structural: Array<{ field: string; msg: string }> = [];

  if (!form.heroPos) {
    structural.push({ field: 'positions', msg: 'select hero position' });
  }
  const heroHoleCount = form.heroHole.filter(Boolean).length;
  if (heroHoleCount < 2) {
    structural.push({ field: 'heroHole', msg: `missing hero hole cards — ${2 - heroHoleCount} of 2 needed` });
  }

  const street = decisionStreet(form);
  const flopHave = form.flop.board.filter(Boolean).length;
  const turnHave = form.turn.board.filter(Boolean).length;
  const riverHave = form.river.board.filter(Boolean).length;

  if (street === 'flop' && flopHave !== 3) {
    structural.push({ field: 'flop-board', msg: `flop needs 3 cards, ${flopHave} selected` });
  }
  if (street === 'turn') {
    if (flopHave !== 3) structural.push({ field: 'flop-board', msg: `flop needs 3 cards, ${flopHave} selected` });
    if (turnHave !== 1) structural.push({ field: 'turn-board', msg: `turn needs 1 card, ${turnHave} selected` });
  }
  if (street === 'river') {
    if (flopHave !== 3) structural.push({ field: 'flop-board', msg: `flop needs 3 cards, ${flopHave} selected` });
    if (turnHave !== 1) structural.push({ field: 'turn-board', msg: `turn needs 1 card, ${turnHave} selected` });
    if (riverHave !== 1) structural.push({ field: 'river-board', msg: `river needs 1 card, ${riverHave} selected` });
  }

  const allCards = [
    ...form.flop.board.filter(Boolean) as string[],
    ...form.turn.board.filter(Boolean) as string[],
    ...form.river.board.filter(Boolean) as string[],
    ...form.heroHole.filter(Boolean) as string[],
  ];
  const seen = new Set<string>();
  const dupes = new Set<string>();
  for (const c of allCards) {
    if (seen.has(c)) dupes.add(c);
    seen.add(c);
  }
  if (dupes.size > 0) {
    structural.push({ field: 'duplicate', msg: `duplicate card${dupes.size > 1 ? 's' : ''}: ${[...dupes].join(', ')}` });
  }

  const pre = classifyPreflop(form.preflop);
  const ipoop = classifyRelativePosition(form.preflop, form.heroPos!);
  const fullBoard = [
    ...form.flop.board,
    ...form.turn.board,
    ...form.river.board,
  ];
  const boardClass = classifyBoard(fullBoard);
  const holeClass = classifyHole(form.heroHole);

  const flopSeq = classifyStreetSeq(form.flop.actions);
  const turnSeq = classifyStreetSeq(form.turn.actions);
  const riverSeq = classifyStreetSeq(form.river.actions);
  const streetSeqByName: Record<string, string | null> = {
    preflop: pre.seq,
    flop: flopSeq,
    turn: turnSeq,
    river: riverSeq,
  };

  const isPreflop = street === 'preflop';
  const partsObj: EncodeParts = {
    street,
    potType: pre.potType,
    position: isPreflop ? form.heroPos : ipoop,
    boardClass,
    streetSeq: streetSeqByName[street] ?? null,
    holeClass,
    preflopSeq: pre.seq,
    heroPos: form.heroPos,
    raises: form.preflop.filter((t) => RAISE_VERBS.has(t.verb)).length,
  };

  if (structural.length > 0) {
    return {
      key: null,
      valid: false,
      tier1_errors: structural,
      structural,
      soft: [],
      parts: partsObj,
      seq: null,
    };
  }

  const keyParts = [
    street,
    pre.potType,
    isPreflop ? form.heroPos : ipoop,
    isPreflop ? '--' : (boardClass || '--'),
    streetSeqByName[street] || 'x',
    holeClass,
  ];
  const key = keyParts.join('|');

  return {
    key,
    valid: true,
    tier1_errors: [],
    structural: [],
    soft: [],
    parts: partsObj,
    seq: key,
  };
}

export { POSITION_ORDER };
