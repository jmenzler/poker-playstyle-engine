/** placeholder source — real per-hand data deferred to backlog 999.8 */

export type FillModeProvider = (handClass: string) => number;

const RANKS_HI = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];

function comboCount(label: string): number {
  if (label.length === 2) return 6;
  if (label.endsWith('s')) return 4;
  return 12;
}

const HAND_STRENGTH: Record<string, number> = (() => {
  const order: string[] = [];
  for (const r of RANKS_HI) order.push(r + r);
  ['AK','AQ','AJ','AT','KQ','KJ','KT','QJ','QT','JT','T9'].forEach((h) => order.push(h + 's'));
  ['A9','A8','A7','A6','A5','A4','A3','A2'].forEach((h) => order.push(h + 's'));
  ['98','87','76','65','54','K9','Q9','J9','T8','97','86','75','64','53'].forEach((h) => order.push(h + 's'));
  ['AK','AQ','AJ','AT','KQ','KJ','KT','QJ','QT','JT'].forEach((h) => order.push(h + 'o'));
  ['A9','A8','A7','K9','Q9','J9','T9','98'].forEach((h) => order.push(h + 'o'));
  const tier: Record<string, number> = {};
  order.forEach((h, i) => { tier[h] = i; });
  for (let i = 0; i < 13; i++) {
    for (let j = 0; j < 13; j++) {
      const r1 = RANKS_HI[i], r2 = RANKS_HI[j];
      let label: string;
      if (i === j) label = r1 + r2;
      else if (j > i) label = r1 + r2 + 's';
      else label = r2 + r1 + 'o';
      if (tier[label] === undefined) tier[label] = 160 + (label.charCodeAt(0) % 8);
    }
  }
  return tier;
})();

/** placeholder source — real per-hand data deferred to backlog 999.8 */
export function placeholderEquity(handClass: string): number {
  const s = HAND_STRENGTH[handClass] ?? 169;
  const eq = 88 - s * 0.34;
  return Math.max(28, Math.min(86, +eq.toFixed(1)));
}

/** placeholder source — real per-hand data deferred to backlog 999.8 */
export function placeholderEvLoss(handClass: string): number {
  const s = HAND_STRENGTH[handClass] ?? 169;
  const seed = (handClass.charCodeAt(0) * 31 + (handClass[1] || '').charCodeAt(0) * 7 + ((handClass[2] || '').charCodeAt(0) || 0)) % 100;
  const t = s / 169;
  const bell = 4 * t * (1 - t);
  return +(bell * 0.18 + (seed / 100) * 0.03).toFixed(3);
}

/** placeholder source — real per-hand data deferred to backlog 999.8 */
export function placeholderArrival(handClass: string): number {
  return +(comboCount(handClass) / 1326 * 100).toFixed(2);
}

export const PLACEHOLDER_FILL_MODES = new Set(['arrival']);
