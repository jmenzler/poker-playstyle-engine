import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider, createBrowserRouter } from 'react-router';
import App from './App';
import './styles/globals.css';

// All routes are loaded lazily. Routes shipped in Plan 06-08 Task 3 are real
// modules; Plan 06-09 routes (probe/strategy/coverage/eval) currently resolve
// to placeholder modules via the catch fallback.
function Placeholder({ name }: { name: string }) {
  return (
    <div className="p-6 text-fg-muted text-[12px]">
      :: {name.toUpperCase()} :: pending plan 09
    </div>
  );
}

const lazyOrPlaceholder = (loader: () => Promise<{ default: React.ComponentType }>, name: string) =>
  React.lazy(() =>
    loader().catch(() => ({ default: () => <Placeholder name={name} /> })),
  );

const Dashboard = lazyOrPlaceholder(() => import('./routes/dashboard'), 'dashboard');
const Patches = lazyOrPlaceholder(() => import('./routes/patches'), 'patches');
const Hands = lazyOrPlaceholder(() => import('./routes/hands'), 'hands');
const Replayer = lazyOrPlaceholder(() => import('./routes/replayer'), 'replayer');
// Plan 06-09 routes — shipped in this plan; lazy-loaded same as Plan 06-08 routes.
const Probe = lazyOrPlaceholder(() => import('./routes/probe'), 'probe');
const Strategy = lazyOrPlaceholder(() => import('./routes/strategy'), 'strategy');
const Coverage = lazyOrPlaceholder(() => import('./routes/coverage'), 'coverage');
const Eval = lazyOrPlaceholder(() => import('./routes/eval'), 'eval');
const Decisions = lazyOrPlaceholder(() => import('./routes/decisions'), 'decisions');
const GapResolver = lazyOrPlaceholder(() => import('./routes/gap-resolver'), 'gap-resolver');
const PatchDetail = lazyOrPlaceholder(() => import('./routes/patch-detail'), 'patch-detail');

const suspend = (node: React.ReactNode) => (
  <React.Suspense fallback={<Placeholder name="loading" />}>{node}</React.Suspense>
);

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: suspend(<Dashboard />) },
      { path: 'dashboard', element: suspend(<Dashboard />) },
      { path: 'probe', element: suspend(<Probe />) },
      { path: 'strategy', element: suspend(<Strategy />) },
      { path: 'coverage', element: suspend(<Coverage />) },
      { path: 'hands', element: suspend(<Hands />) },
      { path: 'patches', element: suspend(<Patches />) },
      { path: 'eval', element: suspend(<Eval />) },
      { path: 'decisions', element: suspend(<Decisions />) },
    ],
  },
  // D-NEW-19: replayer mounts in a detached WebviewWindow at this route.
  { path: '/replayer/:hand_id', element: suspend(<Replayer />) },
  { path: '/gap-resolver', element: suspend(<GapResolver />) },
  { path: '/patches/:patch_id', element: suspend(<PatchDetail />) },
]);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
