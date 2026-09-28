// REST + SSE client for the FastAPI backend (Plan 06-07).
//
// VITE_API_BASE_URL overrides the loopback API used for local development.
//
// SSE subscriptions use EventSource with three event types per Plan 06-07
// (`progress`, `done`, `error`) — see /api/verify/sse and /api/eval/sse.

const API_BASE: string =
  (import.meta.env?.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:8765';

export function getApiBase(): string {
  return API_BASE;
}

async function _request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const r = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`API ${method} ${path} failed: ${r.status} ${text}`);
  }
  // 204 No Content
  if (r.status === 204) return undefined as unknown as T;
  return (await r.json()) as T;
}

export const api = {
  get: <T>(path: string) => _request<T>('GET', path),
  post: <T>(path: string, body?: unknown) => _request<T>('POST', path, body ?? {}),
  del: <T>(path: string) => _request<T>('DELETE', path),
};

export interface SSEHandlers {
  onProgress?: (data: unknown) => void;
  onDone?: (data: unknown) => void;
  onError?: (err: string) => void;
}

// Wire the standard progress/done/error listeners for a job EventSource.
//
// The named server `error` event carries data = JSON.stringify({error: str}) —
// parse it and surface the message. The SAME 'error' listener also fires for
// native transport errors (e.data undefined): only close+report when the
// connection is permanently CLOSED, else let EventSource auto-reconnect.
export function attachJobListeners(es: EventSource, handlers: SSEHandlers, onClosed?: () => void): void {
  es.addEventListener('progress', (e) => {
    try {
      handlers.onProgress?.(JSON.parse((e as MessageEvent).data));
    } catch (err) {
      handlers.onError?.(`progress parse failed: ${String(err)}`);
    }
  });
  es.addEventListener('done', (e) => {
    try {
      handlers.onDone?.(JSON.parse((e as MessageEvent).data));
    } catch (err) {
      handlers.onError?.(`done parse failed: ${String(err)}`);
    }
    es.close();
  });
  es.addEventListener('error', (e) => {
    const data = (e as MessageEvent).data;
    if (typeof data === 'string') {
      let msg = data;
      try {
        const parsed = JSON.parse(data);
        if (parsed && typeof parsed.error === 'string') msg = parsed.error;
      } catch {
        // Non-JSON payload — surface the raw string.
      }
      handlers.onError?.(msg);
      es.close();
      return;
    }
    // Native transport drop (no payload). The job's true terminal state arrives
    // via the server-sent `done`/`error` events, never the transport layer — so a
    // closed connection must NOT be reported as a job failure. Signal it so the
    // caller can reconnect (a long job's closing event must not be lost), but never
    // surface it as onError.
    if (es.readyState === EventSource.CLOSED) {
      es.close();
      onClosed?.();
    }
  });
}

// Returns an unsubscribe function. Closes the EventSource on done/error/cleanup.
const _MAX_SSE_RECONNECTS = 10;

export function subscribeJob(
  jobId: string,
  ssePath: '/api/verify/sse' | '/api/eval/sse',
  handlers: SSEHandlers,
): () => void {
  let terminated = false;
  let attempts = 0;
  let es: EventSource | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const open = () => {
    if (terminated) return;
    es = new EventSource(`${API_BASE}${ssePath}/${jobId}`);
    attachJobListeners(
      es,
      {
        onProgress: handlers.onProgress,
        onDone: (d) => {
          terminated = true;
          handlers.onDone?.(d);
        },
        onError: (msg) => {
          terminated = true;
          handlers.onError?.(msg);
        },
      },
      () => {
        // Transport closed with no terminal event — on a long job the done/error
        // may still be coming. Reconnect (capped) so the closing event isn't lost
        // and the row doesn't freeze at the last progress tick.
        if (terminated || attempts >= _MAX_SSE_RECONNECTS) return;
        attempts += 1;
        timer = setTimeout(open, 2000);
      },
    );
  };
  open();

  return () => {
    terminated = true;
    if (timer) clearTimeout(timer);
    es?.close();
  };
}
