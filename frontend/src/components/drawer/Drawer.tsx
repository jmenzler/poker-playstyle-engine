import { useEffect } from 'react';
import type { ReactNode } from 'react';

// Generic right-edge drawer per UI-SPEC v2.2 §Drawer Patterns.
//
// - 480px fixed width, anchored to right edge (between top + bottom strips)
// - Esc-to-close, click-outside-to-close, [×]-to-close
// - 120ms slide-in animation (the only horizontal motion permitted)
// - Bg #0a0a0a (secondary surface), left edge 1px #262626 border
// - Header (36px, bottom border), scrollable body, optional 48px footer.

interface DrawerProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  footer?: ReactNode;
}

export default function Drawer({ open, onClose, title, children, footer }: DrawerProps) {
  // Esc-to-close
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <>
      {/* Click-outside overlay — UI-SPEC: drawer doesn't dim the panel, but
          a click on the un-covered area closes the drawer. */}
      <div
        className="fixed inset-0 z-40"
        onClick={onClose}
        aria-hidden="true"
        data-testid="drawer-overlay"
      />
      <aside
        role="dialog"
        aria-label={title}
        className="fixed right-0 bg-surface border-l border-border z-50 flex flex-col"
        style={{
          width: 480,
          top: 24, // below TopStrip
          bottom: 24, // above CmdlineStrip
          animation: 'drawerSlideIn 120ms ease-out',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header: 36px, 1px bottom border */}
        <header
          className="px-4 flex items-center justify-between border-b border-border"
          style={{ height: 36 }}
        >
          <h2 className="text-[14px] font-bold">{title}</h2>
          <button
            onClick={onClose}
            className="text-[12px] border border-border px-2"
            aria-label="close drawer"
          >
            [ × ]
          </button>
        </header>

        {/* Body: scrollable */}
        <div className="flex-1 overflow-y-auto p-4 text-[13px]">{children}</div>

        {/* Bottom action bar: 48px, 1px top border (optional) */}
        {footer && (
          <footer
            className="px-4 flex items-center justify-end border-t border-border"
            style={{ height: 48 }}
          >
            {footer}
          </footer>
        )}
      </aside>
      <style>{`
        @keyframes drawerSlideIn {
          from { transform: translateX(100%); }
          to   { transform: translateX(0); }
        }
      `}</style>
    </>
  );
}
