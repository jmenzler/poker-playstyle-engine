import { describe, it, expect, vi, beforeEach } from 'vitest';
import { getRanges, subscribeRangeSolveJob } from '../rangeClient';

describe('getRanges', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('issues a GET whose URL ends with the ranges path for the given hand', async () => {
    const handId = '700000000001';
    globalThis.fetch = vi.fn(() =>
      Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    ) as unknown as typeof fetch;

    await getRanges(handId);

    const calledUrl = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(calledUrl).toMatch(/\/api\/hands\/by-hand\/700000000001\/ranges$/);
  });

  it('resolves to the parsed JSON array returned by the server', async () => {
    globalThis.fetch = vi.fn(() =>
      Promise.resolve(
        new Response(
          JSON.stringify([
            {
              decision_id: 'd1',
              street: 'flop',
              hero_seat: 0,
              multiway: false,
              narrowing: 'ok',
              oop: null,
              ip: null,
            },
          ]),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    ) as unknown as typeof fetch;

    const result = await getRanges('123');
    expect(result).toHaveLength(1);
    expect(result[0].decision_id).toBe('d1');
  });
});

describe('subscribeRangeSolveJob', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('constructs an EventSource whose URL ends with the sse path for the given hand and job', () => {
    const listeners: Record<string, ((e: MessageEvent) => void)[]> = {};
    const mockEs = {
      addEventListener: vi.fn((type: string, cb: (e: MessageEvent) => void) => {
        listeners[type] = [...(listeners[type] ?? []), cb];
      }),
      close: vi.fn(),
    };
    // Arrow functions cannot be used with `new`; use a regular function so the
    // constructor pattern works.
    globalThis.EventSource = vi.fn(function () { return mockEs; }) as unknown as typeof EventSource;

    subscribeRangeSolveJob('HID', 'JOB', {});

    const constructedUrl = (globalThis.EventSource as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(constructedUrl).toMatch(/\/api\/hands\/by-hand\/HID\/sse\/JOB$/);
  });

  it('fires onDone and closes the EventSource when the "done" event is received', () => {
    const listeners: Record<string, ((e: MessageEvent) => void)[]> = {};
    const mockEs = {
      addEventListener: vi.fn((type: string, cb: (e: MessageEvent) => void) => {
        listeners[type] = [...(listeners[type] ?? []), cb];
      }),
      close: vi.fn(),
    };
    globalThis.EventSource = vi.fn(function () { return mockEs; }) as unknown as typeof EventSource;

    const onDone = vi.fn();
    subscribeRangeSolveJob('HID', 'JOB', { onDone });

    const donePayload = { job_id: 'JOB', status: 'done' };
    listeners['done'][0]({ data: JSON.stringify(donePayload) } as MessageEvent);

    expect(onDone).toHaveBeenCalledTimes(1);
    expect(onDone).toHaveBeenCalledWith(donePayload);
    expect(mockEs.close).toHaveBeenCalledTimes(1);
  });

  it('surfaces the parsed .error message from a named server error event', () => {
    const listeners: Record<string, ((e: MessageEvent) => void)[]> = {};
    const mockEs = {
      addEventListener: vi.fn((type: string, cb: (e: MessageEvent) => void) => {
        listeners[type] = [...(listeners[type] ?? []), cb];
      }),
      close: vi.fn(),
    };
    globalThis.EventSource = vi.fn(function () { return mockEs; }) as unknown as typeof EventSource;

    const onError = vi.fn();
    subscribeRangeSolveJob('HID', 'JOB', { onError });

    listeners['error'][0]({ data: JSON.stringify({ error: 'navigation failed' }) } as MessageEvent);

    expect(onError).toHaveBeenCalledWith('navigation failed');
    expect(mockEs.close).toHaveBeenCalledTimes(1);
  });

  it('returns an unsubscribe function that closes the EventSource', () => {
    const mockEs = {
      addEventListener: vi.fn(),
      close: vi.fn(),
    };
    globalThis.EventSource = vi.fn(function () { return mockEs; }) as unknown as typeof EventSource;

    const unsub = subscribeRangeSolveJob('HID', 'JOB', {});
    unsub();

    expect(mockEs.close).toHaveBeenCalledTimes(1);
  });
});
