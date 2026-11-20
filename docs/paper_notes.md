# Notes on the paper draft (`paper/main.pdf`)

## What the paper claims (and only this)
- **Positive:** abstention on low-confidence items raises the local NLI baseline's accuracy (0.99 at 20% coverage);
  near-miss item types make the benchmark harder for it (F1 0.96 on v0, 0.86 on v1; compositions differ).
- **Negative:** the NLI baseline falsely flags 31% of faithful claims and recovers 52% of over-broad holdings; an
  existence check catches only fabricated citations (F1 0.31); the general adaptive-and-verified RAG pipeline did not
  improve judged correctness on 36 MuSiQue questions (0.444 vs 0.583) and was 6.4 to 13.5 times slower.
- **Inconclusive / not done:** whether v1 is hard for LLM checkers (not run, cost); any comparison with the base paper
  or other systems; whether verification reduces unsupported claims; lawyer-validated labels.
- It does **not** claim to be better than the base paper (Verma, arXiv:2608.12571) or any other system. No head-to-head
  comparison exists.

## Figure and table sources
| Item | Source file(s) |
|---|---|
| Fig 1 pipeline | drawn (`paper/fig1.py`) |
| Fig 2, Table 1 | `experiments/legal_india/bench_v0_manifest.json`, `bench_v1_manifest.json` |
| Fig 3, 5, Tables 2-3, Fig 4 | `results_baselines.json`, `results_baselines_v1.json` |
| Fig 6 | `decisions_baselines*.json` + item labels in `bench_v*.jsonl` (curve reproduces the AURC 0.0828 and the 5-point values stored in `results_baselines_v1.json`) |
| Fig 7-9, Tables 4-5 | `experiments/results/main/summary.json` |
| Corpus statistics | `experiments/legal_india/fetch_summary.json`, `docs/legal_india_data.md` |
| Hand-check results (30 v1 items, 32 v0 items) | `docs/legal_india_benchmark.md` (author's reading, not machine-computed) |
| B2 false flags 34/111, 13/42, 15/40 | recomputed from `decisions_baselines_v1.json` and item metadata |

## Must be verified before any submission
1. **Every reference.** They come from search results and arXiv abstract pages read in this session, plus MuSiQue
   from memory. GANDR and the base paper were read only at the abstract-page level. The arXiv IDs of the 2026 papers
   (2608.12571, 2609.10293) came from search results that were internally inconsistent for other papers; open each one.
2. **The statements about the base paper** (Table 6 and Section 7) are drawn from its abstract page only. Read the full
   paper before asserting differences in setup.
3. **The IPC/BNS section table** (`backend/app/legal_india/data/section_map.json`) is unverified; the wrong-code items
   do not depend on it for their labels, but the realism of the section numbers does.
4. **The licence of the judgment texts** (registry says CC-BY-4.0; scope over texts unclear). Do not release item files.
5. **Author and affiliation fields** are placeholders.

## Honest venue assessment
A workshop (AI-and-law or legal NLP) is realistic for a clearly framed benchmark-plus-negative-results paper. A main-track
venue is not realistic without: LLM-checker baselines on v1, lawyer-validated labels (two annotators, agreement
reported), High Court data, and a verified statute table. Reviewers in this area know Verma (2026) and GANDR, so the
contribution must stay narrow (Indian setting, near-miss types, IPC/BNS error type, cost-free baselines).

## Known weaknesses a reviewer will raise
- LLM-generated items with automatic labels; filter-model selection may bias toward easy cases.
- v0 and v1 differ in composition, so the v0-to-v1 drop is indicative only.
- Single court, two years, 199 judgments; small per-type n (23 to 42 test items).
- The LLM-only baseline used the same model family as the v0 generator.
- The MuSiQue experiment is small (36 questions, automatic evaluation, a small NLI verifier chosen for speed).

## IEEE-style reformat (added)
`paper/main.pdf` is now US Letter, two columns, 10 pt Times New Roman, 8 pages, with Roman-numeral sections, "Fig. N."
captions below figures, "TABLE I" captions above tables, and references numbered by first citation (programmatic
renumbering in `paper/build.py`). Citation order: [1] Magesh et al., [2] Verma, [3] Adaptive-RAG, [4] GANDR,
[5] Self-RAG, [6] LegalBench-RAG, [7] Indian Supreme Court Judgments dataset, [8] NLI model card, [9] MuSiQue.

Changes beyond pure formatting (needed for IEEE conventions; no result or claim changed):
- Figures are numbered in order of first appearance (the benchmark-composition figure is now Fig. 1 and the pipeline
  diagram Fig. 2; the pipeline image labels now read "Sec. VI-D" and "Sec. VI-A to VI-C").
- Every figure and table is now cited in the text; added short pointer sentences ("Fig. 2 summarises the systems
  evaluated", "Table III and Fig. 4 give per-type recall", "Tables IV and V and Figs. 7-9 give the results",
  "Table VI compares the two setups", "Fig. 5").
- MuSiQue [9] is now cited at its first mention in Section IV (it was in the list but not cited in the text).
- Appendix is titled "APPENDIX: REPRODUCIBILITY AND COST"; the draft-status box became an italic note under the index
  terms; verification tags are black italic (no amber colouring).

Could not match to IEEE style:
- Not produced with the IEEEtran LaTeX template (no LaTeX installed); it is an HTML/CSS approximation printed by Chrome.
  The authors should re-typeset with the official template/Word template before submission.
- No author affiliations or emails (as instructed; IEEE normally requires them).
- Abstract is about 280 words (IEEE guidance is usually 150 to 250); table captions are full sentences with the source
  file named, rather than short small-caps titles; no copyright/ruler block; no page headers or numbers.
- 8 pages, including the references; check the target workshop's page limit.
- Hyphenation and justification come from the browser engine and are not tuned.

Reminder for the authors: every reference must be verified (the 2026 papers were read at abstract-page level; MuSiQue is
cited from memory); statements about the base paper rest on its abstract page; labels are automatic and not validated by
lawyers; the IPC/BNS table is unverified; the licence of the judgment texts is unconfirmed; v1 has no LLM-checker
baseline; no comparison with prior systems was made.

## Removed from the paper text (author-facing notes, kept here so nothing is lost)
The following were deleted from `paper/main.pdf` at the user's request (front matter, references and prose). The
scientific limitations they referred to remain in the paper as normal prose (Abstract, Section III, Section VII,
Section VIII).
- Front-matter note: "Working draft. All numbers are computed from files in this repository. Labels in the legal
  benchmark are automatic and not validated by lawyers; an LLM-only checker was not run on the harder benchmark (v1);
  no comparison with any prior system was made; the IPC/BNS section table is unverified; the licence of the judgment
  texts is unconfirmed. References marked 'verify' must be checked before submission. Venue fit: a workshop is
  realistic; a main-track venue is not, without lawyer-labelled data and LLM baselines on v1."
- "(verify against the literature before submission)" after the statement of what is new (Section I).
- Per-reference tags "(verify before submission)" on all references; GANDR "(only the abstract page was read)";
  MuSiQue "(cited from memory)"; NLI model "(verify authorship and licence before submission)"; the dataset reference's
  "Licence listed as CC-BY-4.0; scope over judgment texts unconfirmed" (the licence uncertainty is still stated in
  Sections III and VIII).
- Placeholder wording "Draft for a legal-NLP / AI-and-law workshop", "Authors and affiliations to be added".
- One wording change for the same reason: GANDR's "drafting agent" now reads "answer-writing agent".

### Author to-do list (not in the paper)
1. Verify every reference against the original sources (the 2026 papers were read at abstract-page level; MuSiQue was
   cited from memory; NLI model card authorship/licence not checked; Magesh et al. details beyond the abstract
   page not checked).
2. Read the full base paper (Verma) before asserting setup differences in Table VI and Section VII.
3. Check the IPC/BNS section table against the official Gazette; confirm the licence of the judgment texts.
4. Add affiliations and contact details if the venue requires them; re-typeset with the official IEEE template.
5. Venue fit: a workshop (AI-and-law / legal NLP) is realistic; a main-track venue is not without lawyer-labelled
   data, LLM-checker baselines on v1, High Court data and a verified statute table.

### Removed from the paper text: cost and paid-API statements (second round)
Deleted or neutrally reworded in `paper/main.pdf` at the user's request:
- Title: "...and a Cost-Free Verification Baseline" is now "...and a Local Verification Baseline".
- Abstract: "We evaluate checkers that need no paid API call at inference" is now "We evaluate an existence check, a
  locally run NLI model and a date rule"; "was not run on v1 for cost reasons" is now "was not evaluated on v1".
- Introduction: "a checker that costs nothing at inference" is now "a locally run checker"; "zero-API-cost baselines"
  is now "baselines". Conclusion: same rewording; "within a stated budget" removed from the next steps.
- Section III: the two generation cost figures in USD were removed (the models used are still named).
- Section IV: "Only B0, B2 and R need no paid call." removed. Discussion: "zero-cost NLI baseline" is now "locally run
  NLI baseline"; "LLM checkers were not run on v1 for cost reasons" is now "were not evaluated on v1"; "it cost much
  more time" is now "it took much more time"; "threshold therefore costs reliability" is now "trades away
  reliability"; "The cost was latency" is now "The drawback was latency".
- Fig. 3 caption: "was not run on v1 for cost reasons" is now "was not evaluated on v1".
- Table VI: the row "Cost at inference | LLM calls | No paid call for B0, B2 and R" was removed.
- Appendix: the title "Reproducibility and cost" is now "Reproducibility"; the OpenAI spend paragraph (v0 generation
  USD 0.015, B1 USD 0.032, v1 final generation USD 1.00, MuSiQue pipeline USD 0.105) was replaced by a neutral list of
  the models used.
Kept: the scientific limitation that the LLM-only checker was evaluated on v0 only, and that no comparison with prior
systems was made; latency measurements; token counts (Table IV column "Tokens/query" and "about 3.6 times the tokens"),
which are resource measurements, not dollar costs (remove them if the authors prefer). The paid-spend history stays
documented in `docs/legal_india_benchmark.md`.
