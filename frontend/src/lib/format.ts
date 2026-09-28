// Display formatters for the terminal aesthetic (D-NEW-21 #1 + #2).
//
// truncateClusterKey  → mid-string ellipsis for long cluster_keys
// nAnnotation         → "(n=N)" or "(n=N, shrunk)" tail after every CI
// formatCI            → "0.420 [0.180–0.710]" with 3-decimal precision

export function truncateClusterKey(ck: string, maxLen = 60): string {
  if (ck.length <= maxLen) return ck;
  const half = Math.floor((maxLen - 1) / 2);
  return `${ck.slice(0, half)}…${ck.slice(-half)}`;
}

// Parse a `k=v|k=v|...` cluster_key into its fields. Lossy keys carry no felt,
// but they DO encode street/pos/pot_type/n_players — enough for a useful spot
// line when felt_snapshot is NULL.
export function parseClusterKey(ck: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const part of (ck ?? '').split('|')) {
    const eq = part.indexOf('=');
    if (eq > 0) out[part.slice(0, eq)] = part.slice(eq + 1);
  }
  return out;
}

// Human spot line derived from a cluster_key alone (felt-less fallback).
// e.g. "preflop · IP · limp · 3-way" — omits fields the key doesn't carry.
export function clusterKeyDesc(ck: string): string {
  const f = parseClusterKey(ck);
  const parts: string[] = [];
  if (f.street_class) parts.push(f.street_class);
  if (f.hero_pos_rel) parts.push(f.hero_pos_rel);
  if (f.pot_type) parts.push(f.pot_type);
  if (f.n_players_active) parts.push(`${f.n_players_active}-way`);
  return parts.join(' · ');
}

export function nAnnotation(n: number, shrunk: boolean): string {
  return `(n=${n}${shrunk ? ', shrunk' : ''})`;
}

export function formatCI(mean: number, ciLow: number, ciHigh: number): string {
  return `${mean.toFixed(3)} [${ciLow.toFixed(3)}–${ciHigh.toFixed(3)}]`;
}
