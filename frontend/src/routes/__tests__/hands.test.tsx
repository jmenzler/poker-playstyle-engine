import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';

const getRanges = vi.fn();
vi.mock('@/lib/rangeClient', () => ({
  getRanges: (...a: unknown[]) => getRanges(...a),
  postSolveRanges: vi.fn(),
  subscribeRangeSolveJob: vi.fn(() => () => undefined),
}));

vi.mock('@/lib/comboUtils', () => ({
  comboToClass: vi.fn((combo: string) => combo.slice(0, 2) + 's'),
  buildComboClassMap: vi.fn(() => ({ classToData: {} })),
  buildRangeFillProvider: vi.fn(() => () => 0),
  buildEquityFillProvider: vi.fn(() => () => 0),
  buildEvLossFillProvider: vi.fn(() => () => 0),
  buildClassActionMap: vi.fn(() => ({})),
}));

const apiGet = vi.fn();
vi.mock('@/lib/api', () => ({ api: { get: (...a: unknown[]) => apiGet(...a) } }));

import Hands from '../hands';

const HAND_ROW = {
  hand_id: 'h001',
  source: 'hh',
  hero_position: 'BTN',
  stake: '2NL',
  n_decisions: 4,
  street_reached: 'flop',
  played_ts: '2026-06-01T10:00:00',
  solved: true,
};

const REPLAY_STEPS = [
  {
    step_idx: 0,
    street: 'preflop',
    board: [],
    pot: 1.5,
    big_blind: 1,
    actor: 'BTN',
    action_taken: 'BTN: raises',
  },
];

describe('Hands route — PostflopRangePanel mount on expand', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiGet.mockResolvedValue([HAND_ROW]);
    getRanges.mockResolvedValue([]);
  });

  it('calls getRanges when a hand row is expanded', async () => {
    render(<Hands />);
    await waitFor(() => expect(screen.getByText('h001')).toBeInTheDocument());
    fireEvent.click(screen.getByText('h001'));
    await waitFor(() => expect(getRanges).toHaveBeenCalledWith('h001'));
  });

  it('mounts PostflopRangePanel in the expanded row', async () => {
    apiGet
      .mockResolvedValueOnce([HAND_ROW])
      .mockResolvedValueOnce(REPLAY_STEPS);
    render(<Hands />);
    await waitFor(() => expect(screen.getByText('h001')).toBeInTheDocument());
    fireEvent.click(screen.getByText('h001'));
    await waitFor(() =>
      expect(screen.getByText(/RANGE VIEWER/i)).toBeInTheDocument(),
    );
  });
});
