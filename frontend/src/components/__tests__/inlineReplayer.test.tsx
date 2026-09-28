import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import InlineReplayer from '../InlineReplayer';
import type { ReplayStep } from '../InlineReplayer';

const STEP_A: ReplayStep = {
  step_idx: 0,
  street: 'preflop',
  board: [],
  pot: 1.5,
  big_blind: 1,
  actor: 'Hero',
  action_taken: 'Hero: raises to 3',
  seat_map: {
    '1': { name: 'Hero', stack: 97 },
    '3': { name: 'Villain', stack: 99 },
  },
};

const STEP_B: ReplayStep = {
  step_idx: 1,
  street: 'flop',
  board: ['Ah', 'Kd', '2c'],
  pot: 6,
  big_blind: 1,
  actor: 'Villain',
  action_taken: 'Villain: checks',
  seat_map: {
    '1': { name: 'Hero', stack: 97 },
    '3': { name: 'Villain', stack: 96 },
  },
};

describe('InlineReplayer.onStepChange', () => {
  it('fires onStepChange with step 0 on mount', () => {
    const onStepChange = vi.fn();
    render(
      <InlineReplayer steps={[STEP_A, STEP_B]} onStepChange={onStepChange} />,
    );
    expect(onStepChange).toHaveBeenCalledWith(STEP_A);
  });

  it('fires onStepChange when stepping forward', () => {
    const onStepChange = vi.fn();
    const { container } = render(
      <InlineReplayer steps={[STEP_A, STEP_B]} onStepChange={onStepChange} />,
    );
    const nextBtn = container.querySelector('button:last-of-type');
    expect(nextBtn).not.toBeNull();
    fireEvent.click(nextBtn!);
    expect(onStepChange).toHaveBeenLastCalledWith(STEP_B);
  });

  it('does not throw when onStepChange is not provided (optional prop)', () => {
    expect(() =>
      render(<InlineReplayer steps={[STEP_A, STEP_B]} />),
    ).not.toThrow();
  });
});

// SIM god-view steps carry no hero_seat — only `actor`. hideVillainCards must
// key off the heroPosition label, else it fails open and leaks villain cards.
const SIM_STEP: ReplayStep = {
  step_idx: 0,
  street: 'turn',
  board: ['Ah', 'Kd', '2c', '7s'],
  pot: 6,
  big_blind: 1,
  actor: 'BTN',
  action_taken: 'BTN: bets 4',
  seat_map: {
    '1': { name: 'BTN', stack: 97 },
    '3': { name: 'SB', stack: 96 },
  },
  seat_cards: { BTN: ['Qh', 'Qs'], SB: ['9d', '9c'] },
};

describe('InlineReplayer.hideVillainCards (SIM path, no hero_seat)', () => {
  it('hides villain cards via heroPosition and shows the hero its own', () => {
    const { getAllByRole, queryByRole } = render(
      <InlineReplayer steps={[SIM_STEP]} hideVillainCards heroPosition="BTN" />,
    );
    // Hero (BTN) cards face up; villain (SB) cards face-down, never leaked.
    expect(queryByRole('img', { name: 'Qh' })).not.toBeNull();
    expect(queryByRole('img', { name: '9d' })).toBeNull();
    expect(getAllByRole('img', { name: 'back' }).length).toBeGreaterThan(0);
  });

  it('suppresses the actor pill when action_taken is stripped (gap step)', () => {
    const { container } = render(
      <InlineReplayer
        steps={[{ ...SIM_STEP, action_taken: undefined }]}
        hideVillainCards
        heroPosition="BTN"
      />,
    );
    expect(container.querySelector('.felt-pill-banner')).toBeNull();
  });
});
