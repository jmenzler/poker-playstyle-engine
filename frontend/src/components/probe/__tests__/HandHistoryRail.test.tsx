import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import HandHistoryRail from '../HandHistoryRail';
import type { EncodeForm } from '@/lib/probeEncoder';

function form(): EncodeForm {
  return {
    heroPos: 'BTN',
    preflop: [{ pos: 'BTN', verb: 'open', size: 2.5 }],
    flop: { board: ['Ah', 'Kd', '7c'], actions: [] },
    turn: { board: [null], actions: [] },
    river: { board: [null], actions: [] },
    heroHole: ['Qs', 'Qh'],
  } as EncodeForm;
}

const reached = { preflop: true, flop: true, turn: false, river: false };

// Regression: Chip received its street id via the React-reserved `key` prop, which
// React strips — so onFocus fired with undefined and set activeStep out of range,
// crashing StepEditorPanel on form[undefined].board. The id now rides `streetKey`.
describe('HandHistoryRail chip focus', () => {
  it('emits the real street name (never undefined) on chip click', () => {
    const onFocus = vi.fn();
    render(
      <HandHistoryRail form={form()} focusedStreet="preflop" onFocus={onFocus} reachedMap={reached} onEditBoard={vi.fn()} />,
    );
    const chips = screen.getAllByRole('button').filter((b) => b.className.includes('hh-rail-chip'));
    expect(chips.length).toBeGreaterThanOrEqual(4);
    chips.forEach((c) => fireEvent.click(c));
    expect(onFocus).toHaveBeenCalled();
    for (const call of onFocus.mock.calls) {
      expect(call[0]).toBeDefined();
      expect(['preflop', 'flop', 'turn', 'river']).toContain(call[0]);
    }
  });
});
