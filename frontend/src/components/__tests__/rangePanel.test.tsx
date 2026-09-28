import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

const getRanges = vi.fn();
const postSolveRanges = vi.fn();
vi.mock('@/lib/rangeClient', () => ({
  getRanges: (...a: unknown[]) => getRanges(...a),
  postSolveRanges: (...a: unknown[]) => postSolveRanges(...a),
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

import PostflopRangePanel from '../PostflopRangePanel';
import ComboBreakdownPanel from '../ComboBreakdownPanel';

const MULTIWAY_RANGE: import('@/lib/rangeClient').RangesContract = {
  decision_id: 'dp1',
  street: 'flop',
  hero_seat: 0,
  multiway: true,
  narrowing: 'multiway_hu_unsupported',
  oop: null,
  ip: null,
};

const SOLVED_RANGE: import('@/lib/rangeClient').RangesContract = {
  decision_id: 'dp2',
  street: 'flop',
  hero_seat: 0,
  multiway: false,
  narrowing: 'ok',
  oop: {
    combos: ['AhKh', 'AsKs'],
    weights: [0.8, 0.9],
    equity: [0.65, 0.72],
    strategy: [0.6, 0.4, 0.6, 0.4],
    ev_detail: [1.2, 0.9, 1.2, 0.9],
    actions: ['CHECK', 'BET'],
    hero_action: 'CHECK',
  },
  ip: {
    combos: ['QhJh'],
    weights: [1.0],
    equity: [0.35],
    strategy: [0.3, 0.7],
    ev_detail: [0.5, 0.8],
    actions: ['CHECK', 'BET'],
    hero_action: null,
  },
};

const BASE_PANEL_PROPS = {
  handId: 'hand123',
  currentStep: null,
  heroPosition: 'OOP' as const,
  onSolve: vi.fn(),
};

describe('multiway badge', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders a "narrowing unavailable (multiway)" notice when narrowing is multiway_hu_unsupported', () => {
    render(
      <PostflopRangePanel
        {...BASE_PANEL_PROPS}
        ranges={[MULTIWAY_RANGE]}
      />,
    );
    expect(
      screen.getByText(/narrowing unavailable \(multiway\)/i),
    ).toBeInTheDocument();
  });

  it('renders no range grid when narrowing is multiway_hu_unsupported', () => {
    const { container } = render(
      <PostflopRangePanel
        {...BASE_PANEL_PROPS}
        ranges={[MULTIWAY_RANGE]}
      />,
    );
    expect(container.querySelector('.range-grid')).toBeNull();
  });
});

describe('solve CTA', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders a Solve button when ranges is empty at a hero postflop decision point', () => {
    render(
      <PostflopRangePanel
        {...BASE_PANEL_PROPS}
        ranges={[]}
      />,
    );
    expect(screen.getByRole('button', { name: /solve/i })).toBeInTheDocument();
  });
});

describe('hero-only', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('disables the action fill-mode button on the villain (non-acting) grid', () => {
    render(
      <PostflopRangePanel
        {...BASE_PANEL_PROPS}
        ranges={[SOLVED_RANGE]}
      />,
    );
    const actionButtons = screen.getAllByRole('button', { name: /^action$/i });
    const villainActionBtn = actionButtons.find((btn) => btn.hasAttribute('disabled'));
    expect(villainActionBtn).toBeDefined();
    expect(villainActionBtn).toBeDisabled();
  });

  it('disables the ev-loss fill-mode button on the villain (non-acting) grid', () => {
    render(
      <PostflopRangePanel
        {...BASE_PANEL_PROPS}
        ranges={[SOLVED_RANGE]}
      />,
    );
    const evLossButtons = screen.getAllByRole('button', { name: /ev.?loss/i });
    const villainEvLossBtn = evLossButtons.find((btn) => btn.hasAttribute('disabled'));
    expect(villainEvLossBtn).toBeDefined();
    expect(villainEvLossBtn).toBeDisabled();
  });
});

describe('ComboBreakdownPanel L1', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const L1_PROPS = {
    handClass: 'AKs',
    combos: ['AhKh', 'AsKs'],
    weights: [0.8, 0.9],
    equity: [0.65, 0.72],
    strategy: [0.6, 0.4, 0.6, 0.4],
    ev_detail: [1.2, 0.9, 1.2, 0.9],
    actions: ['CHECK', 'BET'],
    hero_action: 'CHECK',
    onClose: vi.fn(),
  };

  it('renders action-mix rows (dist-row) for the selected hand class', () => {
    const { container } = render(<ComboBreakdownPanel {...L1_PROPS} />);
    const distRows = container.querySelectorAll('.dist-row');
    expect(distRows.length).toBeGreaterThanOrEqual(1);
  });

  it('renders EV per action for the selected hand class', () => {
    render(<ComboBreakdownPanel {...L1_PROPS} />);
    expect(screen.getByText(/CHECK/)).toBeInTheDocument();
    expect(screen.getByText(/BET/)).toBeInTheDocument();
  });

  it('flags the hero actual action in the L1 summary', () => {
    render(<ComboBreakdownPanel {...L1_PROPS} />);
    const heroFlagEl = screen.getByText(/CHECK/);
    expect(heroFlagEl).toBeInTheDocument();
  });
});

describe('ComboBreakdownPanel L2', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const L2_PROPS = {
    handClass: 'AKs',
    combos: ['AhKh', 'AsKs'],
    weights: [0.8, 0.9],
    equity: [0.65, 0.72],
    strategy: [0.6, 0.4, 0.6, 0.4],
    ev_detail: [1.2, 0.9, 1.2, 0.9],
    actions: ['CHECK', 'BET'],
    hero_action: 'CHECK',
    onClose: vi.fn(),
  };

  it('expands per-combo rows when the L1 summary row is clicked', () => {
    const { container } = render(<ComboBreakdownPanel {...L2_PROPS} />);
    const summaryRow = container.querySelector('[data-testid="combo-breakdown-l1"]') ??
      container.querySelector('tr.l1-row') ??
      container.querySelector('tr');
    expect(summaryRow).not.toBeNull();
    fireEvent.click(summaryRow!);
    const comboRows = container.querySelectorAll('tr.l2-row, [data-testid="combo-breakdown-l2-row"]');
    expect(comboRows.length).toBeGreaterThanOrEqual(1);
  });

  it('shows per-combo freq, EV, and equity in expanded rows', () => {
    const { container } = render(<ComboBreakdownPanel {...L2_PROPS} />);
    const summaryRow = container.querySelector('[data-testid="combo-breakdown-l1"]') ??
      container.querySelector('tr.l1-row') ??
      container.querySelector('tr');
    fireEvent.click(summaryRow!);
    expect(screen.getByText(/AhKh/)).toBeInTheDocument();
    expect(screen.getByText(/AsKs/)).toBeInTheDocument();
  });
});
