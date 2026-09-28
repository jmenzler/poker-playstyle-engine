import { describe, it, expect, vi, beforeEach } from 'vitest';
import { api, getApiBase, subscribeJob } from '../api';

function mockEventSource() {
  const listeners: Record<string, ((e: Event) => void)[]> = {};
  const es = {
    readyState: 1,
    addEventListener: vi.fn((type: string, cb: (e: Event) => void) => {
      listeners[type] = [...(listeners[type] ?? []), cb];
    }),
    close: vi.fn(),
  };
  globalThis.EventSource = Object.assign(
    vi.fn(function () {
      return es;
    }),
    { CONNECTING: 0, OPEN: 1, CLOSED: 2 },
  ) as unknown as typeof EventSource;
  return { es, fire: (type: string, e: Partial<Event>) => listeners[type]?.forEach((cb) => cb(e as Event)) };
}

describe('api', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('exposes a non-empty API base URL', () => {
    expect(getApiBase()).toMatch(/^https?:\/\//);
  });

  it('GET resolves to parsed JSON on 2xx', async () => {
    globalThis.fetch = vi.fn(
      () =>
        Promise.resolve(
          new Response(JSON.stringify({ ok: true, n: 7 }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          }),
        ) as unknown as ReturnType<typeof fetch>,
    ) as unknown as typeof fetch;

    const out = await api.get<{ ok: boolean; n: number }>('/api/health');
    expect(out.ok).toBe(true);
    expect(out.n).toBe(7);
  });

  it('GET throws Error with status code on non-2xx', async () => {
    globalThis.fetch = vi.fn(() =>
      Promise.resolve(new Response('boom', { status: 500 })),
    ) as unknown as typeof fetch;

    await expect(api.get('/api/oops')).rejects.toThrow(/500/);
  });

  it('subscribeJob surfaces the parsed .error message from a named server error event', () => {
    const { es, fire } = mockEventSource();
    const onError = vi.fn();
    subscribeJob('JOB', '/api/verify/sse', { onError });

    fire('error', { data: JSON.stringify({ error: 'solver crashed: boom' }) } as unknown as Event);

    expect(onError).toHaveBeenCalledWith('solver crashed: boom');
    expect(es.close).toHaveBeenCalledTimes(1);
  });

  it('subscribeJob does NOT close on a transient transport error (lets EventSource reconnect)', () => {
    const { es, fire } = mockEventSource();
    const onError = vi.fn();
    subscribeJob('JOB', '/api/verify/sse', { onError });

    es.readyState = 0; // CONNECTING — browser is auto-reconnecting
    fire('error', {} as unknown as Event); // native transport error: no .data

    expect(es.close).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });

  it('subscribeJob closes + reports on a permanent transport error (readyState CLOSED)', () => {
    const { es, fire } = mockEventSource();
    const onError = vi.fn();
    subscribeJob('JOB', '/api/verify/sse', { onError });

    es.readyState = 2; // CLOSED — permanent failure
    fire('error', {} as unknown as Event);

    expect(es.close).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledTimes(1);
  });

  it('POST serializes body as JSON and sets Content-Type', async () => {
    const fetchSpy = vi.fn(() =>
      Promise.resolve(
        new Response(JSON.stringify({ accepted: true }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    ) as unknown as typeof fetch;
    globalThis.fetch = fetchSpy;

    const out = await api.post<{ accepted: boolean }>('/api/verify', { cluster_key: 'x' });
    expect(out.accepted).toBe(true);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const callArgs = (fetchSpy as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    const init = callArgs[1] as RequestInit;
    expect(init.method).toBe('POST');
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json');
    expect(init.body).toBe(JSON.stringify({ cluster_key: 'x' }));
  });
});
