import type { Config } from 'tailwindcss';

// UI-SPEC v2.2 terminal palette — pure-black + sharp corners + JetBrains Mono.
// Tailwind v4 is CSS-first; this config primarily extends spacing/font/color
// for the legacy class consumers. Tokens are also declared in src/styles/globals.css.
const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        page: '#000000',
        surface: '#0a0a0a',
        'surface-2': '#0d0d0d',
        border: '#262626',
        'border-faint': '#1a1a1a',
        fg: '#d4d4d4',
        'fg-muted': '#888888',
        'fg-caption': '#666666',
        'fg-cluster': '#9ca3af',
        accent: '#3b82f6',
        destructive: '#ef4444',
        success: '#10b981',
        warning: '#f59e0b',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      spacing: {
        sidebar: '240px',
        strip: '24px',
        drawer: '480px',
        row: '36px',
      },
      borderRadius: { DEFAULT: '0', none: '0', sm: '0', md: '0', lg: '0', full: '0' },
    },
  },
  plugins: [],
};
export default config;
