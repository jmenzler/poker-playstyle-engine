import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

vi.mock('react-router', () => ({
  useParams: () => ({ hand_id: 'RC123' }),
}));

const get = vi.fn();
vi.mock('@/lib/api', () => ({ api: { get: (...a: unknown[]) => get(...a) } }));

import Replayer from '../replayer';

const FELT_EVENT = {
  obs_id: 'o1',
  hand_id: 'RC123',
  ts: '2026-05-20',
  cluster_key: 'k',
  street: 'preflop',
  action_taken: 'Hero: raises 0.5 to 0.75',
  actor: 'Hero',
  bet_amount: 0.75,
  pot: 1.25,
  button_seat: 3,
  hero_seat: 1,
  seat_map: {
    '1': { name: 'Hero', stack: 100 },
    '3': { name: 'Villain1', stack: 95 },
    '5': { name: 'Villain2', stack: 80 },
  },
  spot_features: { board: ['Ah', 'Kd'], hero_hole: ['Qs', 'Jc'] },
};

describe('Replayer felt table', () => {
  beforeEach(() => {
    get.mockReset();
  });

  it('renders a validated card as owned rank and suit text', async () => {
    // /api/hands empty → /api/hm3/replay returns the felt event.
    get.mockResolvedValueOnce([]).mockResolvedValueOnce([FELT_EVENT]);
    render(<Replayer />);
    await waitFor(() => expect(screen.getByRole('img', { name: 'Ah' })).toBeInTheDocument());
    expect(screen.getByRole('img', { name: 'Ah' })).toHaveTextContent('A♥');
    expect(screen.getByRole('img', { name: 'Ah' })).not.toHaveAttribute('src');
  });

  it('falls through to /api/hm3/replay when /api/hands rows lack felt fields', async () => {
    // Regression: hm3 hands return non-empty /api/hands rows that carry no
    // spot_features/seat_map. The replayer must NOT short-circuit on them — it
    // must fall through to /api/hm3/replay, which has the felt data.
    const feltless = { obs_id: 'x', hand_id: 'RC123', ts: '', cluster_key: 'k', action_taken: 'fold' };
    get
      .mockResolvedValueOnce([feltless, feltless])
      .mockResolvedValueOnce([FELT_EVENT]);
    render(<Replayer />);
    await waitFor(() => expect(screen.getByRole('img', { name: 'Ah' })).toBeInTheDocument());
    // second call must target the hm3 replay endpoint
    expect(get.mock.calls[1][0]).toContain('/api/hm3/replay');
  });

  it('renders one .felt-seat per seat_map entry', async () => {
    get.mockResolvedValueOnce([]).mockResolvedValueOnce([FELT_EVENT]);
    const { container } = render(<Replayer />);
    await waitFor(() => expect(screen.getByText('Hero')).toBeInTheDocument());
    expect(container.querySelectorAll('.felt-seat')).toHaveLength(3);
    // actor seat carries the highlight class.
    expect(container.querySelectorAll('.felt-seat--actor')).toHaveLength(1);
  });

  it('renders a face-down card without an asset path for invalid input', async () => {
    const bad = {
      ...FELT_EVENT,
      spot_features: { board: ['../etc/passwd'], hero_hole: [] },
    };
    get.mockResolvedValueOnce([]).mockResolvedValueOnce([bad]);
    render(<Replayer />);
    await waitFor(() => expect(screen.getAllByRole('img', { name: 'back' }).length).toBeGreaterThan(0));
    for (const img of screen.getAllByRole('img', { name: 'back' })) {
      expect(img).not.toHaveAttribute('src');
      expect(img).toHaveClass('playing-card--back');
    }
  });
});
