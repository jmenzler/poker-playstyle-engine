import { describe, it, expect } from 'vitest';
import { buildSpotDict, type BuilderState } from '../spotBuilder';

function postflopState(overrides: Partial<BuilderState> = {}): BuilderState {
  return {
    street: 'flop',
    heroPosition: 'BTN',
    preflopAction: 'open',
    raiserPosition: 'CO',
    heroHole: ['Ah', 'Kh'],
    board: ['Jc', 'Jd', 'Ad'],
    postflopTape: [
      { pos: 'BTN', action: 'bet', amount: 50 },
      { pos: 'CO', action: 'call' },
    ],
    ...overrides,
  };
}

describe('buildSpotDict postflop real-seat sink', () => {
  it('appends real-seat postflop tokens to action_sequence (no HERO/VILL literals)', () => {
    const dict = buildSpotDict(postflopState());
    const seq = dict.action_sequence as string[];
    const postflopTokens = seq.filter((t) => !t.includes(':open') && !t.includes(':3bet') && !t.includes(':4bet') && !t.includes(':limp'));
    expect(postflopTokens.length).toBeGreaterThan(0);
    for (const tok of seq) {
      expect(tok).not.toMatch(/^HERO:/);
      expect(tok).not.toMatch(/^VILL:/);
    }
  });

  it('uses heroPosition as the hero seat token (not HERO: literal)', () => {
    const dict = buildSpotDict(postflopState());
    const seq = dict.action_sequence as string[];
    const heroToken = seq.find((t) => t.startsWith('BTN:') && (t.includes('bet') || t.includes('check') || t.includes('call') || t.includes('raise') || t.includes('fold')));
    expect(heroToken).toBeDefined();
  });

  it('preflop action_sequence is unchanged when postflopTape is empty', () => {
    const base = postflopState({ postflopTape: [] });
    const dict = buildSpotDict(base);
    const seq = dict.action_sequence as string[];
    expect(seq).toEqual(['CO:open']);
  });

  it('buildSpotDict without postflopTape is backward-compatible (no postflop tokens appended)', () => {
    const legacyState: BuilderState = {
      street: 'flop',
      heroPosition: 'BTN',
      preflopAction: 'open',
      raiserPosition: 'CO',
      heroHole: ['Ah', 'Kh'],
      board: ['Jc', 'Jd', 'Ad'],
    };
    const dict = buildSpotDict(legacyState);
    const seq = dict.action_sequence as string[];
    expect(seq).toEqual(['CO:open']);
  });
});
