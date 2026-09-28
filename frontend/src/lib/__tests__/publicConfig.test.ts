import { expect, it, vi } from 'vitest';

it('uses a loopback API URL when no override is supplied', async () => {
  vi.stubEnv('VITE_API_BASE_URL', undefined);
  vi.resetModules();
  try {
    const { getApiBase } = await import('../api');
    expect(getApiBase()).toBe('http://127.0.0.1:8765');
  } finally {
    vi.unstubAllEnvs();
    vi.resetModules();
  }
});
