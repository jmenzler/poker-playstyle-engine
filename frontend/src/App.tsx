import { Outlet, useNavigate } from 'react-router';
import { useEffect } from 'react';
import Sidebar from './components/shell/Sidebar';
import TopStrip from './components/shell/TopStrip';
import CmdlineStrip from './components/shell/CmdlineStrip';
import { EvalRunsProvider } from './state/evalRuns';

// Shell layout: TopStrip (24px) over [Sidebar 240px | content] over CmdlineStrip (24px).
// Keyboard hotkeys 1..7 jump to the corresponding sidebar route (UI-SPEC v2.2).
export default function App() {
  const navigate = useNavigate();

  useEffect(() => {
    const routes = [
      '/dashboard',
      '/probe',
      '/strategy',
      '/coverage',
      '/hands',
      '/patches',
      '/eval',
    ];

    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName ?? '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || target?.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key >= '1' && e.key <= '7') {
        const idx = parseInt(e.key, 10) - 1;
        navigate(routes[idx]);
      }
      if (e.key === '9') navigate('/gap-resolver');
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [navigate]);

  return (
    <EvalRunsProvider>
      <div className="flex flex-col h-screen bg-page text-fg font-mono text-[13px]">
        <TopStrip />
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />
          <main className="flex-1 overflow-auto px-6 py-4">
            <Outlet />
          </main>
        </div>
        <CmdlineStrip />
      </div>
    </EvalRunsProvider>
  );
}
