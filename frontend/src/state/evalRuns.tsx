import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';

// Durable in-flight eval-match state. Lives above the router so a running
// match survives tab switches (route unmount). Persisted to localStorage so it
// also survives a full reload while the backend job keeps running.

export interface EvalRun {
  job_id: string;
  opponent: string;
  hands: number;
  seed: number;
  started_at: string; // ISO; used as the reconciliation key against persisted rows
}

interface EvalRunsContext {
  runs: EvalRun[];
  addRun: (run: EvalRun) => void;
  removeRun: (jobId: string) => void;
}

const STORAGE_KEY = 'poker-engine.evalRuns';

const Ctx = createContext<EvalRunsContext | null>(null);

function load(): EvalRun[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as EvalRun[]) : [];
  } catch {
    return [];
  }
}

export function EvalRunsProvider({ children }: { children: React.ReactNode }) {
  const [runs, setRuns] = useState<EvalRun[]>(load);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(runs));
    } catch {
      // best-effort persistence; ignore quota/availability errors
    }
  }, [runs]);

  const addRun = useCallback((run: EvalRun) => {
    setRuns((prev) => [run, ...prev.filter((r) => r.job_id !== run.job_id)]);
  }, []);

  const removeRun = useCallback((jobId: string) => {
    setRuns((prev) => prev.filter((r) => r.job_id !== jobId));
  }, []);

  const value = useMemo(
    () => ({ runs, addRun, removeRun }),
    [runs, addRun, removeRun],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useEvalRuns(): EvalRunsContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useEvalRuns must be used within EvalRunsProvider');
  return ctx;
}
