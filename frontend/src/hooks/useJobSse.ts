// Generic job SSE subscription hook (Plan 06-09 Task 1).
//
// Wraps `subscribeJob` (lib/api.ts) with React lifecycle — handles unmount
// cleanup, jobId changes, and exposes status/progress/result/error state.
// Used by Strategy panel for Stage B verification and Eval panel for match runs.

import { useEffect, useState } from 'react';
import { subscribeJob } from '@/lib/api';

export type JobStatus = 'idle' | 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export interface JobProgress {
  hands_done?: number;
  hands_total?: number;
  stage?: string;
  [key: string]: unknown;
}

export interface JobState<TResult = unknown> {
  status: JobStatus;
  progress: JobProgress | null;
  result: TResult | null;
  error: string | null;
}

const INITIAL: JobState = { status: 'idle', progress: null, result: null, error: null };

/**
 * Subscribe to a Stage B / Eval job via SSE. Returns the current job state.
 *
 * @param jobId — null/undefined → no subscription (returns INITIAL state)
 * @param kind  — selects which SSE endpoint (/api/verify/sse vs /api/eval/sse)
 */
export function useJobSse<TResult = unknown>(
  jobId: string | null | undefined,
  kind: 'verify' | 'eval',
): JobState<TResult> {
  // Derive initial state from jobId so the "no subscription" case doesn't
  // require a synchronous setState inside useEffect (react-hooks rule).
  const [state, setState] = useState<JobState<TResult>>(
    jobId
      ? { status: 'running', progress: null, result: null, error: null }
      : (INITIAL as JobState<TResult>),
  );

  useEffect(() => {
    if (!jobId) {
      // jobId became null after a prior run — reset asynchronously via a
      // microtask so we're not synchronous-during-effect-body.
      const id = setTimeout(() => setState(INITIAL as JobState<TResult>), 0);
      return () => clearTimeout(id);
    }
    const ssePath = kind === 'verify' ? '/api/verify/sse' : '/api/eval/sse';
    // Reset state when jobId/kind transitions; we're subscribing to a new
    // external source, so a synchronous setState body call is the correct
    // sync point. The lint rule's cascading-render concern doesn't apply
    // here because the subsequent SSE callbacks all use functional setState.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setState({ status: 'running', progress: null, result: null, error: null });

    const close = subscribeJob(jobId, ssePath, {
      onProgress: (data) =>
        setState((s) => ({ ...s, status: 'running', progress: data as JobProgress })),
      onDone: (data) =>
        setState((s) => ({
          ...s,
          status: 'done',
          result: data as TResult,
          progress: null,
        })),
      onError: (err) =>
        setState((s) => ({ ...s, status: 'failed', error: String(err) })),
    });

    return close;
  }, [jobId, kind]);

  return state;
}
