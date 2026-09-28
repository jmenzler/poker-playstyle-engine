import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';

const apiGet = vi.fn();
const apiPost = vi.fn();
vi.mock('@/lib/api', () => ({
  api: {
    get: (...a: unknown[]) => apiGet(...a),
    post: (...a: unknown[]) => apiPost(...a),
  },
}));

import Patches from '../patches';

const PATCHES = [
  {
    patch_id: 'aaaaaaaa1111',
    ts: '2026-06-01T10:00:00',
    cluster_key: 'k1',
    source: 'manual',
    pre_ev_loss: 0.5,
    post_ev_loss: 0.1,
    status: 'applied',
  },
  {
    patch_id: 'bbbbbbbb2222',
    ts: '2026-06-01T11:00:00',
    cluster_key: 'k2',
    source: 'manual',
    pre_ev_loss: 0.4,
    post_ev_loss: 0.2,
    status: 'applied',
  },
];

describe('Patches bulk rollback', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    apiGet.mockReset();
    apiPost.mockReset();
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    apiGet.mockResolvedValue(PATCHES);
    apiPost.mockResolvedValue({});
  });

  it('bulk button posts a rollback for every selected patch', async () => {
    render(<Patches />);
    await waitFor(() => expect(screen.getAllByRole('checkbox')).toHaveLength(2));

    const boxes = screen.getAllByRole('checkbox');
    fireEvent.click(boxes[0]);
    fireEvent.click(boxes[1]);

    const bulk = await screen.findByText(/rollback selected 2/);
    fireEvent.click(bulk);

    await waitFor(() => {
      const rollbackCalls = apiPost.mock.calls.filter((c) => String(c[0]).includes('/rollback'));
      expect(rollbackCalls).toHaveLength(2);
    });
    const urls = apiPost.mock.calls.map((c) => String(c[0]));
    expect(urls).toContain('/api/patches/aaaaaaaa1111/rollback');
    expect(urls).toContain('/api/patches/bbbbbbbb2222/rollback');
  });

  it('bulk button does nothing when the confirm is declined', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(false);
    render(<Patches />);
    await waitFor(() => expect(screen.getAllByRole('checkbox')).toHaveLength(2));

    fireEvent.click(screen.getAllByRole('checkbox')[0]);
    fireEvent.click(await screen.findByText(/rollback selected 1/));

    await waitFor(() => {
      const rollbackCalls = apiPost.mock.calls.filter((c) => String(c[0]).includes('/rollback'));
      expect(rollbackCalls).toHaveLength(0);
    });
  });
});
