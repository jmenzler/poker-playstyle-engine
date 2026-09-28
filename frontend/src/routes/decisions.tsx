import { Fragment, useEffect, useState } from 'react';
import { api } from '@/lib/api';
import HandReplay from '@/components/HandReplay';

// Decisions browser — surfaces the Milvus preflop_decisions / postflop_decisions
// collections. Read-only filter+list panel with click-to-expand detail.
//
// Data source: GET /api/decisions/sample (see src/api/decisions.py).

type CollectionName = 'preflop_decisions' | 'postflop_decisions';

interface DecisionRow {
  decision_id: string;
  street_class?: string;
  street?: string;
  pot_type?: string;
  hero_pos_rel?: string;
  n_players_active?: number;
  spr_x100?: number;
  hero_action_type?: string;
  active?: boolean;
  confidence?: number;
  gto_score?: number;
  feature_spec_version?: number;
}

interface SampleResponse {
  collection: string;
  filter: string;
  count: number;
  offset?: number;
  limit?: number;
  total?: number;
  rows: DecisionRow[];
}

interface Filters {
  street: string;
  pot_type: string;
  hero_pos_rel: string;
  hero_action_type: string;
  n_players_active: string;
}

const EMPTY_FILTERS: Filters = {
  street: '',
  pot_type: '',
  hero_pos_rel: '',
  hero_action_type: '',
  n_players_active: '',
};

const PAGE_SIZES = [50, 100, 250, 500, 1000] as const;
const FULL_SEARCH_PAGE = 1000;
const FULL_SEARCH_CAP = 5000;

// Distinct values from Milvus (sampled). Milvus VARCHAR == is exact-match, so
// dropdowns prevent case/spelling mismatches.
const STREET_VALUES = ['preflop', 'flop', 'turn', 'river'];
const POT_TYPE_VALUES = ['srp', '3bet', 'limp'];
const POS_VALUES = ['IP', 'OOP'];
const ACTION_VALUES = ['bet', 'call', 'check', 'fold', 'raise'];
const N_PLAYERS_VALUES = ['2', '3', '4', '5', '6'];

function buildQuery(
  collection: CollectionName,
  f: Filters,
  limit: number,
  offset: number,
): string {
  const parts: string[] = [`collection=${collection}`, `limit=${limit}`, `offset=${offset}`];
  if (f.street) parts.push(`street=${encodeURIComponent(f.street)}`);
  if (f.pot_type) parts.push(`pot_type=${encodeURIComponent(f.pot_type)}`);
  if (f.hero_pos_rel) parts.push(`hero_pos_rel=${encodeURIComponent(f.hero_pos_rel)}`);
  if (f.hero_action_type)
    parts.push(`hero_action_type=${encodeURIComponent(f.hero_action_type)}`);
  if (f.n_players_active)
    parts.push(`n_players_active=${encodeURIComponent(f.n_players_active)}`);
  return parts.join('&');
}

function sprDisplay(v: number | undefined): string {
  if (v === undefined || v === null) return '—';
  if (v < 0) return '—'; // preflop sentinel
  return (v / 100).toFixed(2);
}

export default function Decisions() {
  const [collection, setCollection] = useState<CollectionName>('postflop_decisions');
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [rows, setRows] = useState<DecisionRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<DecisionRow | null>(null);
  const [limit, setLimit] = useState<number>(50);
  const [offset, setOffset] = useState<number>(0);
  const [total, setTotal] = useState<number>(0);
  const [capped, setCapped] = useState<boolean>(false);

  // Initial fetch on mount with no filters.
  useEffect(() => {
    runFetch(collection, EMPTY_FILTERS, limit, 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runFetch(c: CollectionName, f: Filters, l: number, o: number) {
    setLoading(true);
    setErr(null);
    setCapped(false);
    try {
      const r = await api.get<SampleResponse>(
        `/api/decisions/sample?${buildQuery(c, f, l, o)}`,
      );
      setRows(r.rows);
      setOffset(o);
      setTotal(r.total ?? 0);
      setSelected(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(msg);
      setRows([]);
    } finally {
      setLoading(false);
    }
  }

  async function onApply() {
    // Full filtered search: page through all matching rows up to FULL_SEARCH_CAP.
    setLoading(true);
    setErr(null);
    setSelected(null);
    const accum: DecisionRow[] = [];
    let pageOffset = 0;
    let totalFromServer = 0;
    let truncated = false;
    try {
      while (accum.length < FULL_SEARCH_CAP) {
        const remaining = FULL_SEARCH_CAP - accum.length;
        const pageLimit = Math.min(FULL_SEARCH_PAGE, remaining);
        const r = await api.get<SampleResponse>(
          `/api/decisions/sample?${buildQuery(collection, filters, pageLimit, pageOffset)}`,
        );
        accum.push(...r.rows);
        totalFromServer = r.total ?? totalFromServer;
        if (r.rows.length < pageLimit) break;
        pageOffset += pageLimit;
        if (accum.length >= FULL_SEARCH_CAP) {
          truncated = true;
          break;
        }
      }
      setRows(accum);
      setTotal(totalFromServer);
      setOffset(0);
      setCapped(truncated);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(msg);
      setRows([]);
    } finally {
      setLoading(false);
    }
  }

  function onReset() {
    setFilters(EMPTY_FILTERS);
    runFetch(collection, EMPTY_FILTERS, limit, 0);
  }

  function onChangeCollection(next: CollectionName) {
    setCollection(next);
    runFetch(next, filters, limit, 0);
  }

  function onChangeLimit(next: number) {
    setLimit(next);
    runFetch(collection, filters, next, 0);
  }

  function onPage(direction: -1 | 1) {
    const nextOffset = Math.max(0, offset + direction * limit);
    runFetch(collection, filters, limit, nextOffset);
  }

  function setF<K extends keyof Filters>(key: K, value: string) {
    setFilters((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <div className="space-y-4">
      <h1 className="text-[14px] font-bold">
        :: DECISIONS :: browse milvus ::{' '}
        <span className="text-fg-muted">
          {rows.length} shown
          {total > 0 && ` · ${total.toLocaleString()} total in ${collection}`}
        </span>
        {capped && (
          <span className="text-warning ml-2">
            (capped at {FULL_SEARCH_CAP.toLocaleString()} — narrow filter for more)
          </span>
        )}
        {loading && <span className="text-warning ml-2">(loading…)</span>}
      </h1>

      {/* FILTER BAR */}
      <section className="border border-border p-3">
        <div className="grid grid-cols-6 gap-2 text-[12px]">
          <label className="flex flex-col">
            <span className="text-fg-caption text-[10px]">collection</span>
            <select
              value={collection}
              onChange={(e) => onChangeCollection(e.target.value as CollectionName)}
              className="bg-page border border-border px-2 py-1"
            >
              <option value="postflop_decisions">postflop_decisions</option>
              <option value="preflop_decisions">preflop_decisions</option>
            </select>
          </label>
          <label className="flex flex-col">
            <span className="text-fg-caption text-[10px]">street</span>
            <select
              value={filters.street}
              onChange={(e) => setF('street', e.target.value)}
              className="bg-page border border-border px-2 py-1"
            >
              <option value="">any</option>
              {STREET_VALUES.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col">
            <span className="text-fg-caption text-[10px]">pot_type</span>
            <select
              value={filters.pot_type}
              onChange={(e) => setF('pot_type', e.target.value)}
              className="bg-page border border-border px-2 py-1"
            >
              <option value="">any</option>
              {POT_TYPE_VALUES.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col">
            <span className="text-fg-caption text-[10px]">hero_pos_rel</span>
            <select
              value={filters.hero_pos_rel}
              onChange={(e) => setF('hero_pos_rel', e.target.value)}
              className="bg-page border border-border px-2 py-1"
            >
              <option value="">any</option>
              {POS_VALUES.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col">
            <span className="text-fg-caption text-[10px]">hero_action</span>
            <select
              value={filters.hero_action_type}
              onChange={(e) => setF('hero_action_type', e.target.value)}
              className="bg-page border border-border px-2 py-1"
            >
              <option value="">any</option>
              {ACTION_VALUES.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col">
            <span className="text-fg-caption text-[10px]">n_players</span>
            <select
              value={filters.n_players_active}
              onChange={(e) => setF('n_players_active', e.target.value)}
              className="bg-page border border-border px-2 py-1"
            >
              <option value="">any</option>
              {N_PLAYERS_VALUES.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="flex gap-2 mt-3 text-[12px] items-center">
          <button
            onClick={onApply}
            disabled={loading}
            className="border border-success text-success px-2 py-1"
            title="Search the full collection for the filter — paginates server-side up to the cap."
          >
            [ apply (search all) ]
          </button>
          <button
            onClick={onReset}
            disabled={loading}
            className="border border-border px-2 py-1"
          >
            [ reset ]
          </button>
          <span className="text-fg-caption text-[10px] ml-4">page size</span>
          <select
            value={limit}
            onChange={(e) => onChangeLimit(parseInt(e.target.value, 10))}
            className="bg-page border border-border px-2 py-1"
            disabled={loading || capped}
            title={capped ? 'paging disabled while showing full-search results' : 'rows per page in browse mode'}
          >
            {PAGE_SIZES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <button
            onClick={() => onPage(-1)}
            disabled={loading || offset === 0 || capped}
            className="border border-border px-2 py-1 disabled:opacity-40"
          >
            [ ◀ prev ]
          </button>
          <button
            onClick={() => onPage(1)}
            disabled={loading || capped || rows.length < limit}
            className="border border-border px-2 py-1 disabled:opacity-40"
          >
            [ next ▶ ]
          </button>
          {!capped && total > 0 && (
            <span className="text-fg-muted text-[10px] ml-2">
              {offset + 1}–{offset + rows.length} of {total.toLocaleString()}
            </span>
          )}
        </div>
      </section>

      {err && <div className="text-destructive text-[12px]">ERROR: {err}</div>}

      {/* TABLE */}
      <section className="border border-border p-3">
        <table className="w-full text-[12px]">
          <thead className="text-fg-muted">
            <tr>
              <th className="text-left p-1">decision_id</th>
              <th className="text-left p-1">street</th>
              <th className="text-left p-1">pot_type</th>
              <th className="text-left p-1">pos</th>
              <th className="text-right p-1">n_p</th>
              <th className="text-right p-1">spr</th>
              <th className="text-left p-1">action</th>
              <th className="text-right p-1">gto</th>
              <th className="text-right p-1">conf</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const isOpen = selected?.decision_id === r.decision_id;
              return (
                <Fragment key={r.decision_id}>
                  <tr
                    onClick={() =>
                      setSelected((cur) =>
                        cur?.decision_id === r.decision_id ? null : r,
                      )
                    }
                    className={`border-b border-border-faint hover:bg-surface-2 cursor-pointer ${
                      isOpen ? 'bg-surface-2' : ''
                    }`}
                  >
                    <td className="p-1 text-fg-cluster">{r.decision_id?.slice(0, 16)}</td>
                    <td className="p-1">{r.street ?? '—'}</td>
                    <td className="p-1">{r.pot_type ?? '—'}</td>
                    <td className="p-1">{r.hero_pos_rel ?? '—'}</td>
                    <td className="p-1 text-right">{r.n_players_active ?? '—'}</td>
                    <td className="p-1 text-right">{sprDisplay(r.spr_x100)}</td>
                    <td className="p-1">{r.hero_action_type ?? '—'}</td>
                    <td className="p-1 text-right">{r.gto_score?.toFixed(2) ?? '—'}</td>
                    <td className="p-1 text-right">{r.confidence?.toFixed(2) ?? '—'}</td>
                  </tr>
                  {isOpen && (
                    <tr className="border-b border-success">
                      <td colSpan={9} className="bg-surface-2 p-4">
                        <DetailPanel
                          row={r}
                          onClose={() => setSelected(null)}
                        />
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
            {rows.length === 0 && !loading && (
              <tr>
                <td colSpan={9} className="p-4 text-fg-muted text-center">
                  no decisions match the filter
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function dpIndexOf(decisionId: string | undefined): number | null {
  const m = decisionId?.match(/_dp(\d+)$/);
  return m ? Number(m[1]) : null;
}

function DetailPanel({ row, onClose }: { row: DecisionRow; onClose: () => void }) {
  const handId = row.decision_id?.split('_dp')[0] ?? '';
  const [showReplay, setShowReplay] = useState(false);
  return (
    <div className="border border-success p-3">
      <div className="flex items-center justify-between mb-3">
        <div className="text-[12px] font-bold">
          :: DECISION DETAIL :: {row.decision_id}
        </div>
        <div className="flex gap-2">
          {handId && (
            <button
              onClick={() => setShowReplay((v) => !v)}
              className="text-[12px] border border-accent text-accent px-2 py-[2px]"
              title="Replay the full hand inline"
            >
              {showReplay ? '[ hide replay ▲ ]' : '[ visualize spot ▶ ]'}
            </button>
          )}
          <button
            onClick={onClose}
            className="text-[12px] border border-border px-2"
            aria-label="close detail panel"
          >
            [ × ]
          </button>
        </div>
      </div>
      {showReplay && handId && (
        <div className="mb-4 border-b border-border-faint pb-4">
          <HandReplay handId={handId} anchorStepIdx={dpIndexOf(row.decision_id)} />
        </div>
      )}
      <div className="grid grid-cols-2 gap-6 text-[12px]">
        <div>
          <div className="text-fg-caption text-[10px] mb-1">SPOT</div>
          <Kv k="street_class" v={row.street_class} />
          <Kv k="street" v={row.street} />
          <Kv k="pot_type" v={row.pot_type} />
          <Kv k="hero_pos_rel" v={row.hero_pos_rel} />
          <Kv k="n_players_active" v={row.n_players_active} />
          <Kv k="spr" v={sprDisplay(row.spr_x100)} />
        </div>
        <div>
          <div className="text-fg-caption text-[10px] mb-1">ACTION + META</div>
          <Kv k="hero_action_type" v={row.hero_action_type} />
          <Kv k="gto_score" v={row.gto_score?.toFixed(4)} />
          <Kv k="confidence" v={row.confidence?.toFixed(4)} />
          <Kv k="active" v={row.active === undefined ? undefined : String(row.active)} />
          <Kv k="feature_spec_version" v={row.feature_spec_version} />
          <Kv k="spr_x100 (raw)" v={row.spr_x100} />
        </div>
      </div>
    </div>
  );
}

function Kv({ k, v }: { k: string; v: string | number | undefined }) {
  return (
    <div className="flex justify-between">
      <span className="text-fg-muted">{k}</span>
      <span>{v ?? '—'}</span>
    </div>
  );
}
