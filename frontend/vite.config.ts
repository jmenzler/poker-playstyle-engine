import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Tauri convention: dev server on 127.0.0.1:1420 (matches tauri.conf.json devUrl).
// envPrefix exposes both VITE_* and TAURI_* env vars to the React app.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 1420,
    strictPort: true,
    host: '127.0.0.1',
  },
  clearScreen: false,
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    target: ['es2021', 'chrome105', 'safari13'],
    sourcemap: !!process.env.TAURI_DEBUG,
  },
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
});
