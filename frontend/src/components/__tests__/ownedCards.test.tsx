import { describe, it, expect, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import InlineReplayer from '../InlineReplayer';
import CardPicker from '../CardPicker';

describe('owned card rendering', () => {
  it('renders ranks and suits without requesting card graphics', () => {
    const { container } = render(
      <InlineReplayer steps={[{ step_idx: 0, board: ['Ah', 'Kd', 'Qs', 'Jc'] }]} />,
    );
    for (const [card, glyph] of [['Ah', '♥'], ['Kd', '♦'], ['Qs', '♠'], ['Jc', '♣']]) {
      const face = screen.getByRole('img', { name: card });
      expect(face).toHaveTextContent(card[0]);
      expect(face).toHaveTextContent(glyph);
      expect(face).not.toHaveAttribute('src');
    }
    expect(container.querySelector('img')).toBeNull();
  });

  it('renders unknown and invalid cards face down without exposing their input', () => {
    const { container } = render(
      <InlineReplayer steps={[{
        step_idx: 0,
        board: ['../private.svg'],
        seat_map: { '1': { name: 'Player', stack: 100 } },
      }]} />,
    );
    expect(screen.getAllByRole('img', { name: 'back' })).toHaveLength(3);
    expect(container.innerHTML).not.toContain('../private.svg');
    expect(container.querySelector('img')).toBeNull();
  });

  it('keeps picker selection and duplicate disabling with text cards', () => {
    const onChange = vi.fn();
    const { container } = render(
      <CardPicker target="hero" slots={2} selected={[]} usedElsewhere={['Kd']} onChange={onChange} />,
    );
    const ace = screen.getByRole('img', { name: 'Ah' });
    expect(ace).toHaveTextContent('♥');
    fireEvent.click(screen.getByRole('button', { name: 'Ah' }));
    expect(onChange).toHaveBeenCalledWith(['Ah']);
    expect(screen.getByRole('button', { name: /Kd/ })).toBeDisabled();
    expect(container.querySelector('img')).toBeNull();
  });

  it('keeps a text dealer marker on the felt', () => {
    const { container } = render(
      <InlineReplayer steps={[{
        step_idx: 0,
        button_seat: 1,
        seat_map: { '1': { name: 'Player', stack: 100 } },
      }]} />,
    );
    expect(container.querySelector('.felt-dealer')).toHaveTextContent('D');
    expect(container.querySelector('.felt-dealer img')).toBeNull();
  });
});
