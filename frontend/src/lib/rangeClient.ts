// Typed client for the Phase-17 ranges contract: REST + dedicated SSE subscriber.
import { api, getApiBase, attachJobListeners, SSEHandlers } from '@/lib/api';

export interface RangeSeatPayload {
  combos: string[];
  weights: number[];
  equity: number[];
  strategy?: number[];
  ev_detail?: number[];
  actions?: string[];
  hero_action?: string | null;
}

export interface RangesContract {
  decision_id: string;
  street: string;
  hero_seat: number;
  multiway: boolean;
  narrowing: 'ok' | 'multiway_hu_unsupported' | 'nav_failed';
  oop: RangeSeatPayload | null;
  ip: RangeSeatPayload | null;
}

export async function getRanges(handId: string): Promise<RangesContract[]> {
  return api.get<RangesContract[]>(`/api/hands/by-hand/${encodeURIComponent(handId)}/ranges`);
}

export async function postSolveRanges(
  handId: string,
  force = false,
): Promise<{ job_id: string; status: string; hand_id: string }> {
  return api.post(`/api/hands/by-hand/${encodeURIComponent(handId)}/solve-ranges${force ? '?force=true' : ''}`);
}

export function subscribeRangeSolveJob(
  handId: string,
  jobId: string,
  handlers: SSEHandlers,
): () => void {
  const es = new EventSource(
    `${getApiBase()}/api/hands/by-hand/${encodeURIComponent(handId)}/sse/${jobId}`,
  );
  attachJobListeners(es, handlers);
  return () => es.close();
}
