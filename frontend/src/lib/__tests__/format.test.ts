import { describe, it, expect } from 'vitest';
import { truncateClusterKey, nAnnotation, formatCI } from '../format';

describe('truncateClusterKey', () => {
  it('returns key unchanged when short', () => {
    expect(truncateClusterKey('abc')).toBe('abc');
  });

  it('returns key unchanged when exactly maxLen', () => {
    const key = 'a'.repeat(60);
    expect(truncateClusterKey(key, 60)).toBe(key);
  });

  it('truncates mid-string with ellipsis when too long', () => {
    const long =
      'street_class=flop|pot_type=srp|hero_pos_rel=ip|texture=dry_rainbow|action_history=ccb50c';
    const out = truncateClusterKey(long, 30);
    expect(out.length).toBeLessThanOrEqual(31); // 30 + ellipsis char
    expect(out).toContain('…');
    // Both ends preserved
    expect(out.startsWith(long.slice(0, 14))).toBe(true);
    expect(out.endsWith(long.slice(-14))).toBe(true);
  });
});

describe('nAnnotation', () => {
  it('omits shrunk label when false', () => {
    expect(nAnnotation(200, false)).toBe('(n=200)');
  });

  it('includes shrunk label when true', () => {
    expect(nAnnotation(8, true)).toBe('(n=8, shrunk)');
  });
});

describe('formatCI', () => {
  it('formats mean + CI bounds with 3 decimals', () => {
    expect(formatCI(0.42, 0.18, 0.71)).toBe('0.420 [0.180–0.710]');
  });

  it('rounds values to 3 decimal places', () => {
    expect(formatCI(0.1234, 0.0987, 0.4567)).toBe('0.123 [0.099–0.457]');
  });
});
