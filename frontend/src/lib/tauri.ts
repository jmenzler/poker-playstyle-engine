// Tauri 2.x detached WebviewWindow helpers (D-NEW-19).
//
// Lazy-imports @tauri-apps/api so the same code runs in browser-only
// fallback (where Tauri is unavailable). If the import fails or WebviewWindow
// throws, falls back to `window.open` so the operator still gets a separate
// browser window.

export interface ReplayerWindowOptions {
  width?: number;
  height?: number;
}

export async function openReplayer(
  handId: string,
  opts: ReplayerWindowOptions = {},
): Promise<void> {
  const label = `replayer-${handId}`;
  const url = `/replayer/${handId}`;
  const title = `Replayer · hand_${handId}`;
  const width = opts.width ?? 1280;
  const height = opts.height ?? 800;

  try {
    const mod = await import('@tauri-apps/api/webviewWindow');
    const WebviewWindow = mod.WebviewWindow;
    const webview = new WebviewWindow(label, { url, title, width, height });
    webview.once('tauri://created', () => {
      // eslint-disable-next-line no-console
      console.log(`Replayer ${label} created`);
    });
    webview.once('tauri://error', (e: unknown) => {
      // eslint-disable-next-line no-console
      console.error(`Replayer ${label} spawn failed:`, e);
    });
  } catch (err) {
    // Not running in Tauri (e.g. dev fallback in a plain browser tab).
    // Open the replayer route in a new browser window so the operator still
    // gets a detached surface.
    // eslint-disable-next-line no-console
    console.warn('Tauri WebviewWindow unavailable — falling back to window.open', err);
    window.open(url, label, `width=${width},height=${height}`);
  }
}
