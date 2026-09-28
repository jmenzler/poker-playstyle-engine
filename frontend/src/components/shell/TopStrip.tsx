import { useEffect, useState } from 'react';
import { api } from '@/lib/api';

// 24px top status strip per UI-SPEC v2.2 §Top status strip.
// Renders title + 3 subsystem markers (DB / Milvus / solver) + session context
// (right-aligned). Health values come from /api/dashboard's health object.

interface HealthSnapshot {
  db?: string;
  milvus?: string;
  solver?: string;
  [k: string]: string | undefined;
}

interface DashboardLite {
  health?: HealthSnapshot;
  session?: { session_id?: string } | null;
  kb_growth?: { total_nodes?: number } | null;
  recent_activity?: { event_type?: string; summary?: string }[] | null;
}

function dotColor(state: string | undefined): string {
  if (state === 'healthy') return 'text-success';
  if (state === 'down' || state === 'unhealthy') return 'text-destructive';
  if (state === 'reconnecting' || state === 'running') return 'text-warning';
  // 'unavailable' (solver binary absent) + 'unknown' fall through to neutral grey
  return 'text-fg-muted';
}

export default function TopStrip() {
  const [health, setHealth] = useState<HealthSnapshot>({});
  const [session, setSession] = useState<string>('—');
  const [kbNodes, setKbNodes] = useState<string>('—');
  const [lastPatch, setLastPatch] = useState<string>('—');

  useEffect(() => {
    let alive = true;
    api
      .get<DashboardLite>('/api/dashboard')
      .then((d) => {
        if (!alive) return;
        setHealth(d.health ?? {});
        if (d.session?.session_id) setSession(d.session.session_id);
        if (typeof d.kb_growth?.total_nodes === 'number')
          setKbNodes(String(d.kb_growth.total_nodes));
        const patch = d.recent_activity?.find((a) =>
          a.event_type?.includes('patch'),
        );
        if (patch?.summary) setLastPatch(patch.summary);
      })
      .catch(() => {
        // offline / backend unreachable — leave defaults
      });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <header
      className="bg-surface flex items-center px-4 text-[11px] border-b border-border"
      style={{ height: 24 }}
    >
      <span className="text-[12px] font-bold text-fg mr-6">poker-engine :: study</span>
      <span className="mr-4">
        <span className={dotColor(health.db)}>■</span> DB
      </span>
      <span className="mr-4">
        <span className={dotColor(health.milvus)}>■</span> Milvus
      </span>
      <span className="mr-4">
        <span className={dotColor(health.solver)}>■</span> solver
      </span>
      <span className="ml-auto text-fg-muted">
        session: {session} · last patch: {lastPatch} · KB nodes: {kbNodes}
      </span>
    </header>
  );
}
