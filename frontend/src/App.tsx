import { useEffect, useState } from 'react';
import { NavLink, Route, Routes } from 'react-router-dom';
import { api } from './api';
import { ScalesMark } from './components/Logo';
import Chat from './pages/Chat';
import Documents from './pages/Documents';
import History from './pages/History';
import How from './pages/How';
import Insights from './pages/Insights';
import Roadmap from './pages/Roadmap';
import Verification from './pages/Verification';

const NAV = [
  ['/', 'Research'],
  ['/documents', 'Case file / Authorities'],
  ['/insights', 'Sources considered'],
  ['/verification', 'Citation checks'],
  ['/roadmap', 'Indian law roadmap'],
  ['/how-it-works', 'How it works'],
  ['/history', 'History'],
] as const;

type Health = { llm_configured: boolean; nli_models: Record<string, string> } | 'down' | null;

function HealthPill({ health }: { health: Health }) {
  if (health === null) return null;
  if (health === 'down') {
    return <span className="rounded-full bg-red-50 px-3 py-1 text-xs text-red-800 ring-1 ring-inset ring-red-200">Backend unreachable (is it running on :8000?)</span>;
  }
  const nli = Object.values(health.nli_models);
  const loading = nli.some((s) => s === 'loading');
  return (
    <span className="flex items-center gap-2 text-xs text-indigo-100">
      <span className={`h-2 w-2 rounded-full ${health.llm_configured ? 'bg-emerald-400' : 'bg-amber-400'}`} />
      {health.llm_configured ? 'Research engine ready' : 'OPENAI_API_KEY not set'}
      {loading && <span className="text-gold-100">· citation-check model loading…</span>}
    </span>
  );
}

export default function App() {
  const [health, setHealth] = useState<Health>(null);
  useEffect(() => {
    const load = () => api.health().then(setHealth).catch(() => setHealth('down'));
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="min-h-screen bg-ivory">
      <div className="z-30 md:sticky md:top-0">
        <header className="bg-indigo-900 text-white">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-6 py-4">
            <div className="flex items-center gap-3">
              <ScalesMark className="h-9 w-9 text-gold-400" />
              <div>
                <div className="font-serif text-2xl font-semibold leading-none tracking-tight">NyayaCheck</div>
                <div className="mt-1 text-xs text-indigo-100">Verified legal research support (India)</div>
              </div>
            </div>
            <HealthPill health={health} />
          </div>
          <nav className="mx-auto flex max-w-6xl flex-wrap gap-1 px-6 pb-0 text-sm" aria-label="Main">
            {NAV.map(([to, label]) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `border-b-2 px-3 py-2 font-medium ${isActive ? 'border-gold-400 text-white' : 'border-transparent text-indigo-200 hover:text-white'}`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
        </header>
        <div role="note" className="border-b border-gold-400/50 bg-gold-50 px-6 py-1.5 text-center text-xs text-gold-700">
          <strong>Decision-support tool for legal professionals. Not legal advice.</strong> Verify every authority against the official source before relying on it.
        </div>
      </div>
      <main className="mx-auto max-w-6xl px-6 py-8">
        <Routes>
          <Route path="/" element={<Chat />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/insights" element={<Insights />} />
          <Route path="/insights/:id" element={<Insights />} />
          <Route path="/verification" element={<Verification />} />
          <Route path="/verification/:id" element={<Verification />} />
          <Route path="/roadmap" element={<Roadmap />} />
          <Route path="/how-it-works" element={<How />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </div>
  );
}
