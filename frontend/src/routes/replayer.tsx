import { useEffect, useState } from 'react';
import { useParams } from 'react-router';
import { api } from '@/lib/api';
import InlineReplayer, { ReplayStep } from '@/components/InlineReplayer';
import './replayer-felt.css';

// Standalone route: fetches the unified by-hand replay payload (same endpoint as
// the inline /hands expansion) then delegates rendering to <InlineReplayer>, so
// both surfaces render identically. Route path, param, and deep-link contract are
// unchanged per D-13; raw HM3 history is the only fallback when there are no steps.

interface Hm3Hand {
  hand_id: string;
  gamenumber: string;
  handtimestamp: string | null;
  handhistory: string;
}

export default function Replayer() {
  const { hand_id } = useParams<{ hand_id: string }>();
  const [steps, setSteps] = useState<ReplayStep[]>([]);
  const [hm3, setHm3] = useState<Hm3Hand | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!hand_id) return;
    setErr(null);
    setSteps([]);
    setHm3(null);
    // Unified read path (D-06/D-07): the standalone route consumes the SAME
    // by-hand endpoint as the inline /hands expansion, so both render identically.
    // The endpoint normalizes SIM (stored felt + derived seats) and HM (on-demand
    // reconstruction) into one ReplayStep[] shape — no client-side fallback chain.
    api
      .get<ReplayStep[]>(`/api/hands/by-hand/${encodeURIComponent(hand_id)}/replay`)
      .then((parsed) => {
        if (parsed.length > 0) {
          setSteps(parsed);
          return;
        }
        // No replay steps for this hand_id; fall back to the raw HM3 history text
        // so deep-links never render blank.
        return api
          .get<Hm3Hand>(`/api/hm3/hand?hand_id=${encodeURIComponent(hand_id)}`)
          .then(setHm3)
          .catch((e) =>
            setErr(
              `no replay steps for hand_${hand_id}; HM3 lookup also failed: ${
                e instanceof Error ? e.message : String(e)
              }`,
            ),
          );
      })
      .catch((e) => setErr(String(e)));
  }, [hand_id]);

  if (err) return <div className="text-destructive p-6 text-[12px]">ERROR: {err}</div>;

  if (hm3) {
    return (
      <div className="min-h-screen bg-page text-fg font-mono p-6 space-y-4">
        <h1 className="text-[14px] font-bold">
          :: REPLAYER :: hand_{hand_id}{' '}
          <span className="text-fg-muted text-[12px]">
            ({hm3.gamenumber} · {hm3.handtimestamp ?? '—'})
          </span>
        </h1>
        <div className="text-[10px] text-fg-caption">
          observations table empty (Phase 2 writes Milvus only) — showing raw HM3 history
        </div>
        <pre className="border border-border bg-surface-2 p-4 text-[11px] whitespace-pre-wrap overflow-x-auto">
          {hm3.handhistory}
        </pre>
      </div>
    );
  }

  if (!steps.length) {
    return (
      <div className="text-fg-muted p-6 text-[12px]">loading hand {hand_id}…</div>
    );
  }

  return (
    <div className="min-h-screen bg-page text-fg font-mono p-6 space-y-4">
      <h1 className="text-[14px] font-bold">
        :: REPLAYER :: hand_{hand_id}{' '}
        <span className="text-fg-muted text-[12px]">({steps.length} steps)</span>
      </h1>
      <InlineReplayer steps={steps} />
    </div>
  );
}
