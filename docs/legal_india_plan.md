# Plan: Verified Legal Research Support for Indian Law

Status: plan, written 2026-11-03. Nothing in this document is a result. All scores are the author's
judgment, not measurements.

## 1. Positioning

A **decision-support** tool for Indian lawyers, law clerks and judges. It is not legal advice and never
makes a decision. Two connected features:

1. **Citation Auditor (core).** Paste a brief, memo or draft judgment. The system extracts each cited
   proposition, finds the cited authority, and checks whether the authority supports the proposition,
   with pinpoint paragraph evidence. Unsupported, misattributed, and unverifiable citations are flagged.
2. **Verified research assistant (front end).** Ask a question; get an answer in which every statement
   carries a pinpoint citation that passes the same audit, or an explicit "not verified".

## 2. What is new relative to prior work

Generic claim-level verification for legal RAG already exists (see the literature check in the project
discussion; open each paper before citing). Our contribution must be India-specific and narrow:

| Idea | Why it matters in India |
|---|---|
| Indian citation parsing (AIR, SCC, SCR, INSC neutral citations, SCC OnLine, party-name short forms) | Existing tools target US Bluebook citations |
| Court hierarchy as a verification signal (Art. 141: Supreme Court law binds all courts; a High Court binds courts in its territory) | A citation can be real yet non-binding for the forum |
| Statute transition: IPC to BNS, CrPC to BNSS, Evidence Act to BSA (in force 1 July 2024; offences before that date stay under IPC, procedure moved to BNSS) | A frequent, checkable error: citing the wrong code for the date of offence |
| Calibrated abstention (risk-coverage curve) | A tool for judges must know when not to answer |
| Adversarial citation benchmark built from real Indian judgments (injected fake, misattributed and misquoted citations) | Cheap to build, objective to score |

## 3. Data (verify licences before use)

- **Supreme Court judgments:** AWS Open Data registry "Indian Supreme Court Judgments", sourced from eCourts,
  1950 onward, parquet metadata, CC-BY-4.0 per the registry page. The licence may cover the compiled
  dataset rather than the underlying texts; confirm on the registry and repository before publishing derived data.
- **High Court judgments:** a parallel open dataset covering 25 High Courts is reported; confirm location and licence.
- **Statutes:** official bare acts from India Code (indiacode.nic.in) and Gazette texts. Mapping tables from
  blogs and commercial guides disagree in places; **only official texts are authoritative.**
- **Existing benchmarks to compare against or reuse:** IL-PCR / IL-PCSR (precedent and statute retrieval),
  ILSI (statute identification), ILDC (judgment prediction; known preprocessing issues), InLegalBERT (Indian legal encoder).

## 4. System changes (isolated in `backend/app/legal_india/`)

1. Judgment ingestion with **paragraph-level chunks** and stable paragraph ids (pinpoint citations).
2. `citations`: parser and normaliser for Indian citation formats; resolver to a corpus case id.
3. `statutes`: section extractor and an IPC/CrPC/Evidence-Act to BNS/BNSS/BSA mapper built from official texts.
4. `hierarchy`: court and bench metadata, binding/persuasive rules.
5. Verifier: existing NLI + judge pipeline, extended with legal checks (authority exists, quote exists in
   the cited paragraph, pinpoint is correct, forum-binding status, statute-in-force on the offence date).
6. Calibration and abstention reporting.
7. Auditor UI: a document pane with each citation marked verified / unsupported / unverifiable.

## 5. Evaluation (what a reviewer will ask for)

1. **Adversarial citation test:** inject (a) fabricated cases, (b) real case with wrong proposition,
   (c) wrong paragraph, (d) wrong code (IPC vs BNS), (e) altered quotes. Report detection precision,
   recall and F1 per type, against a trivial baseline (existence check only) and an LLM-only checker.
2. **Expert labels:** two people with legal training label about 100 claims; report agreement (Cohen's kappa)
   and score the automatic judges against them. Without this the evaluation is automatic-only and weak.
3. **Calibration:** risk-coverage curve and selective accuracy.
4. **Ablations:** each legal check on and off.
5. **Cost and latency**, reported honestly (the NLI step was the bottleneck in the generic system).

## 6. Risks and guardrails

- Never presented as legal advice; mandatory human review; the UI states limits of verification.
- Confidentiality: documents of clients and courts must not leave the user's environment unless the user
  accepts it; offer an on-premises configuration with a local model.
- Verification of "good law" (not overruled) needs treatment data that is not openly available; start with
  what can be checked (existence, quote, paragraph, hierarchy, statute transition) and state that limit.
- Do not publish judgment text beyond what the licence allows.

## 7. Honest assessment so far

| Lens | Score (author's judgment) |
|---|---|
| Novelty vs prior work | 4/10 generic, about 6/10 with the India-specific checks |
| Evidence and rigour | 3/10 now (earlier generic benchmark was negative; no legal data, no expert labels) |
| Venue fit | 5/10 (workshop or JURIX/ICAIL tier realistic; main ACL/EMNLP unlikely at present) |
| Viability and risk | 6/10 |

## 8. Milestones

1. **M1 (started):** Indian citation parser and statute mapper with tests; small corpus pulled from the open dataset.
2. **M2:** paragraph-level ingestion; adversarial benchmark generator; baselines.
3. **M3:** legal verifier and calibration; first measured results.
4. **M4:** expert labelling; ablations; Auditor UI.
5. **M5:** write-up (negative or mixed results reported as they are).
