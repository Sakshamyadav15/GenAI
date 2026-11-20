import { Link } from 'react-router-dom';
import { Card, PageHeader } from '../components/ui';

interface Step {
  title: string;
  body: string;
  tone: 'plain' | 'adaptive' | 'verify';
}

const STEPS: Step[] = [
  { title: '1. Your research question', body: 'You state a question about the authorities in the case file.', tone: 'plain' },
  {
    title: '2. Difficulty assessment',
    body: 'The system lists the separate points that must be found (sub-questions) and the kind of reasoning needed, and adds simple wording cues. These give a score from 0 to 1 and a label: simple, moderate or complex.',
    tone: 'adaptive',
  },
  {
    title: '3. How many passages to read',
    body: 'The score decides how many passages to retrieve, within a range for its label (by default 1–3 for simple, 4–7 for moderate, 8–12 for complex).',
    tone: 'adaptive',
  },
  {
    title: '4. Retrieve authorities',
    body: 'The question and each sub-question are searched separately in the case file, and the results are merged. If the match looks weak, the search is widened once.',
    tone: 'adaptive',
  },
  { title: '5. Draft an answer with citations', body: 'The model answers using only the numbered passages, and cites [n] after each statement that relies on one.', tone: 'plain' },
  { title: '6. Break out the propositions', body: 'The answer is split into single, self-contained propositions. Each keeps the citations of the sentence it came from.', tone: 'verify' },
  {
    title: '7. Check each proposition',
    body: 'A language-inference model checks whether the cited passage actually supports the proposition. Doubtful cases go to a second model that must quote the words it relies on. The result is: supported by authority, partly supported, or not supported.',
    tone: 'verify',
  },
  {
    title: '8. Refine or finalise',
    body: 'If propositions fail, more passages are retrieved and the answer is redrafted, up to a fixed number of retries. The best draft is returned; statements still not supported are removed and the caution is stated.',
    tone: 'verify',
  },
];

const TONES = {
  plain: 'border-slate-200 bg-white',
  adaptive: 'border-indigo-200 bg-indigo-50',
  verify: 'border-emerald-200 bg-emerald-50',
} as const;

export default function How() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="How the system works"
        subtitle="A conventional retrieval system always reads the same number of passages and trusts its own answer. This prototype changes both ends: it chooses how much to read from the question, and it checks every proposition against its cited passage. These checks are general-purpose; they are not yet specific to Indian law (see the roadmap)."
      />

      <div className="flex flex-wrap gap-4 text-xs text-slate-600">
        <span className="flex items-center gap-2"><i className="h-3 w-3 rounded border border-slate-300 bg-white" /> Standard retrieval step</span>
        <span className="flex items-center gap-2"><i className="h-3 w-3 rounded border border-indigo-300 bg-indigo-100" /> Adaptive reading depth</span>
        <span className="flex items-center gap-2"><i className="h-3 w-3 rounded border border-emerald-300 bg-emerald-100" /> Citation checking</span>
      </div>

      <ol className="grid gap-3 md:grid-cols-2">
        {STEPS.map((s, i) => (
          <li key={s.title} className={`relative rounded-xl border p-4 ${TONES[s.tone]}`}>
            <h3 className="text-base font-semibold text-slate-900">{s.title}</h3>
            <p className="mt-1 text-sm leading-relaxed text-slate-700">{s.body}</p>
            {i < STEPS.length - 1 && i % 2 === 0 && (
              <span aria-hidden className="absolute -right-3 top-1/2 hidden -translate-y-1/2 text-slate-300 md:block">→</span>
            )}
          </li>
        ))}
      </ol>

      <div className="grid gap-4 md:grid-cols-2">
        <Card title="The baseline it is compared with">
          <p className="text-sm leading-relaxed text-slate-700">
            Fixed top-k retrieval (default k=5) feeding the same prompt, model and documents, with no difficulty assessment and no citation
            checking. Everything else is identical, so any difference in the results comes from the two changes above.
          </p>
        </Card>
        <Card title="The variants compared in testing">
          <ul className="space-y-1 text-sm text-slate-700">
            <li><strong>B</strong> Baseline: fixed k, no citation checking</li>
            <li><strong>A</strong> Adaptive reading depth only</li>
            <li><strong>V</strong> Citation checking only (fixed k)</li>
            <li><strong>A+V</strong> Both: the checked answer</li>
          </ul>
          <p className="mt-2 text-xs text-slate-500">A− (adaptive depth without splitting the question into sub-questions) separates the effect of depth from decomposition.</p>
        </Card>
      </div>

      <Card title="Where to look next">
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">
          <li><Link className="text-indigo-700 hover:underline" to="/">Research</Link>: ask a question and watch each step stream in.</li>
          <li><Link className="text-indigo-700 hover:underline" to="/insights">Sources considered</Link>: the difficulty score, how many passages were read, and which.</li>
          <li><Link className="text-indigo-700 hover:underline" to="/verification">Citation checks</Link>: every proposition, its supporting passage and its result.</li>
          <li><Link className="text-indigo-700 hover:underline" to="/roadmap">Indian law roadmap</Link>: what is planned and not yet available.</li>
        </ul>
      </Card>
    </div>
  );
}
