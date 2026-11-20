import { Badge, Card, PageHeader } from '../components/ui';

interface Planned {
  title: string;
  what: string;
  why: string;
}

// Everything here is NOT available in this application yet. Do not describe any of it as working.
const PLANNED: Planned[] = [
  {
    title: 'Indian citation parsing',
    what: 'Recognise and normalise AIR, SCC, SCC OnLine, SCR, SCALE and INSC (neutral) citations, including parallel citations and pinpoint paragraph references.',
    why: 'To resolve each citation in a brief to the judgment it names before checking what it says.',
  },
  {
    title: 'Criminal-code transition check (IPC / CrPC / Evidence Act to BNS / BNSS / BSA)',
    what: 'Flag a section cited under the wrong code for the date of the offence. The new codes took effect on 1 July 2024; offences up to 30 June 2024 remain under the earlier substantive law.',
    why: 'Citing the wrong code for the date of offence is a concrete, checkable error.',
  },
  {
    title: 'Court-hierarchy (Article 141) binding check',
    what: 'Tell whether a cited decision is binding or only persuasive for the forum concerned: Supreme Court law binds all courts, a High Court binds courts within its territory.',
    why: 'A citation can be genuine yet not binding on the court it is placed before.',
  },
  {
    title: 'Pinpoint paragraph verification',
    what: 'Check that the cited paragraph, not merely the cited case, supports the proposition, and that any quotation appears in it.',
    why: 'Most citation errors lie in the paragraph, not in the case name.',
  },
  {
    title: 'Calibrated abstention',
    what: 'Report how often the system is right at each level of "I cannot verify this", so a reader can see when to trust it.',
    why: 'A tool used on legal questions must know when not to answer.',
  },
];

export default function Roadmap() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Indian law roadmap"
        subtitle="What is planned for Indian legal research, and what is not yet available. Nothing on this page works in the application today."
      />

      <div className="rounded-lg border border-gold-400/60 bg-gold-50 p-4 text-sm leading-relaxed text-gold-700">
        <strong>Honest status.</strong> The checks listed below are <strong>in development</strong>. Parts of some of them exist as
        early prototype code in the project, but they are not connected to this interface, and none has been validated by lawyers. Until
        they are, this application checks citations only in a general way (see below) and has no knowledge of Indian citation formats,
        the new criminal codes or the court hierarchy.
      </div>

      <Card title="What the application does today">
        <ul className="list-disc space-y-1 pl-5 text-sm leading-relaxed text-slate-700">
          <li>Answers questions from the documents in the case file, with numbered citations to the passages relied on.</li>
          <li>Checks each proposition in the answer against the passage it cites, and marks it supported, partly supported or not supported.</li>
          <li>Retries with extra retrieval when propositions are not supported, and removes what it still cannot support, saying so.</li>
          <li>Chooses how many passages to retrieve from the difficulty of the question.</li>
        </ul>
        <p className="mt-3 text-xs text-slate-500">
          These checks are general-purpose and automatic. They do not know Indian law, and the demonstration documents are synthetic.
        </p>
      </Card>

      <div className="space-y-3">
        <h2 className="font-serif text-xl font-semibold text-indigo-900">Planned Indian-law checks</h2>
        {PLANNED.map((p) => (
          <section key={p.title} className="rounded-xl border border-slate-200 bg-white p-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-base font-semibold text-slate-900">{p.title}</h3>
              <span className="flex gap-2">
                <Badge tone="gold">In development</Badge>
                <Badge>Not yet available</Badge>
              </span>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-slate-700">{p.what}</p>
            <p className="mt-1 text-xs text-slate-500">Why: {p.why}</p>
          </section>
        ))}
      </div>

      <Card title="Validation">
        <p className="text-sm leading-relaxed text-slate-700">
          <strong>Expert (lawyer) validation is pending.</strong> No result shown in this application has been reviewed by a person with
          legal training. Treat every output as a lead to be checked against the official source, not as authority.
        </p>
      </Card>
    </div>
  );
}
