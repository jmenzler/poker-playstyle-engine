import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import InlineReplayer, { ReplayStep } from '@/components/InlineReplayer';

interface HandRow {
  hand_id?: string;
  obs_id: string;
}

export interface HandReplayProps {
  // Replay a specific hand. Mutually exclusive with clusterKey.
  handId?: string;
  // Replay a representative hand from a cluster_key (probe): the component first
  // resolves one observed hand in that cluster, then replays it.
  clusterKey?: string;
  // Position to anchor at the bottom of the felt (the clicked spot's hero).
  heroPosition?: string | null;
  // Anchor on the actor of a specific dp step (e.g. the clicked decision_id's
  // dp index) — used when the caller knows the dp but not the position label.
  anchorStepIdx?: number | null;
  rx?: number;
  ry?: number;
}

// Fetches the unified by-hand replay payload and renders the InlineReplayer.
// Shared by /hands, /decisions, /probe, and the standalone /replayer route so
// every clickable hand gets the same inline god-view replay.
export default function HandReplay({
  handId,
  clusterKey,
  heroPosition = null,
  anchorStepIdx = null,
  rx = 380,
  ry = 210,
}: HandReplayProps) {
  const [steps, setSteps] = useState<ReplayStep[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [resolvedHand, setResolvedHand] = useState<string | null>(handId ?? null);

  // Resolve a representative hand_id when given a cluster_key (probe path).
  useEffect(() => {
    if (handId) {
      setResolvedHand(handId);
      return;
    }
    if (!clusterKey) return;
    setResolvedHand(null);
    setErr(null);
    api
      .get<HandRow[]>(`/api/hands?cluster_key=${encodeURIComponent(clusterKey)}&limit=1`)
      .then((rows) => {
        const id = rows[0]?.hand_id ?? rows[0]?.obs_id;
        if (id) setResolvedHand(id);
        else setErr(`no observed hand for cluster ${clusterKey}`);
      })
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, [handId, clusterKey]);

  useEffect(() => {
    if (!resolvedHand) return;
    setSteps(null);
    setErr(null);
    api
      .get<ReplayStep[]>(`/api/hands/by-hand/${encodeURIComponent(resolvedHand)}/replay`)
      .then(setSteps)
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, [resolvedHand]);

  if (err) return <div className="text-destructive p-2 text-[12px]">ERROR: {err}</div>;
  if (!resolvedHand || steps === null) {
    return <div className="text-fg-muted p-2 text-[12px]">loading hand…</div>;
  }
  if (steps.length === 0) {
    return <div className="text-fg-muted p-2 text-[12px]">no replay steps for hand {resolvedHand}</div>;
  }
  return (
    <div className="space-y-1">
      {clusterKey && (
        <div className="text-[10px] text-fg-caption">
          representative hand for cluster · {resolvedHand}
        </div>
      )}
      <InlineReplayer
        steps={steps}
        rx={rx}
        ry={ry}
        heroPosition={
          heroPosition ??
          (anchorStepIdx != null ? (steps[anchorStepIdx]?.actor ?? null) : null)
        }
      />
    </div>
  );
}
