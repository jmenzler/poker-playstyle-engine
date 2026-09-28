import { Fragment, useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { truncateClusterKey, clusterKeyDesc } from '@/lib/format';
import FeltSnapshot from '@/components/FeltSnapshot';

interface CoverageRow {
  obs_id: string;
  cluster_key: string;
  ts: string;
  session_id: string;
  max_neighbor_distance: number;
  street: string;
  action_taken?: string;
  felt_snapshot?: {
    street?: string;
    hero_position?: string;
    hero_hole_cards?: string[];
    board_cards?: string[];
    pot_size_bb?: number;
    effective_stack_bb?: number;
    action_sequence?: string[];
  } | null;
}

function TextFallback({ r }: { r: CoverageRow }) {
  const fs = r.felt_snapshot;
  const spot = fs
    ? `${fs.street ?? '?'} · ${fs.hero_position ?? '?'} · pot ${fs.pot_size_bb != null ? `${fs.pot_size_bb}bb` : '—'} · eff ${fs.effective_stack_bb != null ? `${fs.effective_stack_bb}bb` : '—'}`
    : clusterKeyDesc(r.cluster_key);
  const seq = fs?.action_sequence;
  return (
    <div>
      <div className="text-[10px] text-fg-caption mb-2">
        :: SPOT DESCRIPTION :: {r.obs_id?.slice(0, 12)}
      </div>
      <div className="text-[12px] text-fg-muted">
        {spot || '—'}
        {seq && seq.length > 0
          ? <><br />{seq.join(' → ')}{r.action_taken ? ` → ${r.action_taken}` : ''}</>
          : r.action_taken && <><br />hero: {r.action_taken}</>}
      </div>
    </div>
  );
}

export default function Coverage() {
  const [rows, setRows] = useState<CoverageRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [expandedObs, setExpandedObs] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<{ coverage: CoverageRow[] }>('/api/leaks?type=coverage&limit=50')
      .then((d) => setRows(d.coverage ?? []))
      .catch((e) => setErr(String(e)));
  }, []);

  function toggleExpand(obsId: string) {
    setExpandedObs((prev) => (prev === obsId ? null : obsId));
  }

  return (
    <div className="space-y-4">
      <h1 className="text-[14px] font-bold">
        :: COVERAGE GAPS :: flagged_sparse=TRUE :: {rows.length} clusters
      </h1>
      {err && <div className="text-destructive text-[12px]">ERROR: {err}</div>}
      <table className="w-full text-[12px]">
        <thead className="text-fg-muted border-b border-border">
          <tr>
            <th className="text-left p-2">obs_id</th>
            <th className="text-left p-2">cluster_key</th>
            <th className="text-left p-2">street</th>
            <th className="text-right p-2">max_dist</th>
            <th className="text-left p-2">session</th>
            <th className="text-left p-2">ts</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <Fragment key={r.obs_id}>
              <tr
                className={`border-b border-border-faint hover:bg-surface-2 cursor-pointer${expandedObs === r.obs_id ? ' border-l-2 border-accent' : ''}`}
                onClick={() => toggleExpand(r.obs_id)}
                tabIndex={0}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && toggleExpand(r.obs_id)}
              >
                <td className="p-2">{r.obs_id?.slice(0, 8)}</td>
                <td className="p-2 text-fg-cluster">
                  {truncateClusterKey(r.cluster_key)}
                </td>
                <td className="p-2">{r.street}</td>
                <td className="p-2 text-right text-warning">
                  {r.max_neighbor_distance?.toFixed(3)}
                </td>
                <td className="p-2">{r.session_id?.slice(0, 8)}</td>
                <td className="p-2 text-fg-muted">{r.ts?.slice(0, 19)}</td>
              </tr>
              {expandedObs === r.obs_id && (
                <tr key={r.obs_id + '-expanded'}>
                  <td colSpan={6} className="bg-surface p-4 border-b border-border">
                    <div className="text-[10px] text-fg-caption mb-2 flex items-center gap-2">
                      :: FELT SNAPSHOT :: {r.obs_id?.slice(0, 12)}
                      <button
                        className="ml-4 border border-border px-2 text-[10px]"
                        aria-label="collapse spot detail"
                        onClick={(e) => { e.stopPropagation(); setExpandedObs(null); }}
                      >
                        [ ▲ collapse ]
                      </button>
                    </div>
                    {r.felt_snapshot
                      ? <FeltSnapshot
                          board={r.felt_snapshot.board_cards}
                          heroHole={r.felt_snapshot.hero_hole_cards}
                          pot={r.felt_snapshot.pot_size_bb}
                          effectiveStack={r.felt_snapshot.effective_stack_bb}
                          street={r.felt_snapshot.street}
                          heroPos={r.felt_snapshot.hero_position}
                          actionSequence={r.felt_snapshot.action_sequence}
                          actionTaken={r.action_taken}
                        />
                      : <TextFallback r={r} />}
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
          {rows.length === 0 && !err && (
            <tr>
              <td colSpan={6} className="p-4 text-fg-muted text-center">
                no coverage gaps detected.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
