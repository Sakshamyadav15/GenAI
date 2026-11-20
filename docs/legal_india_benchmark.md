# Adversarial citation benchmark for Indian law: v0 and v1 (Supreme Court judgments)

**Read this first: what v1 did and did not run.** v1 (near-miss corruptions, below) was generated and
scored only partly, because the OpenAI budget was withdrawn mid-run (see "v1: what ran" and "OpenAI spend").
The strongest baseline, an LLM-only checker, was **not run on v1**, so v1 has not yet been shown to be
harder than v0 for that checker. Only the free baselines (existence, local NLI, wrong-code rule) were run.

Status: milestone M2, first measured baselines. Everything below is measured on the files in
`experiments/legal_india/` (`bench_v0.jsonl`, `results_baselines.json`); nothing is hand-entered.

**Headline: this benchmark is too easy to discriminate between systems.** A plain LLM-only checker
reaches F1 0.99 on the test split. It is a sanity check for the generator and the metrics, not yet a
benchmark that can show an advantage of a better auditor. See "What to fix before relying on it".

## v1: near-miss benchmark (added 2026-11-04)

Motivation: on v0 an LLM-only checker scored F1 0.99, so v0 cannot separate a better auditor from a plain
prompt. v1 keeps the matched design (one base proposition from one paragraph of one judgment gives several
items) and adds corruptions that look right on the surface. Everything below is measured on
`experiments/legal_india/bench_v1.jsonl`, `bench_v1_manifest.json` and `results_baselines_v1.json`.
The item file and the decisions contain judgment quotes and are gitignored (redistribution licence unconfirmed).

### Types

| Type | Test n | Construction (labels are automatic by construction) |
|---|---|---|
| none, verbatim / paraphrase / conditional | 111 (case 105, statute 6) | faithful claim cited to its true paragraph; paraphrase controls reword the claim (numbers must survive) |
| fabricated_case | 42 | citation that does not resolve (as in v0) |
| wrong_paragraph | 42 | right case, unrelated paragraph (as in v0) |
| near_paragraph | 38 | right case, the most similar OTHER paragraph (text-embedding-3-small), similarity capped at 0.75, and a filter model (gpt-4.1-mini) judged that it does not state any of the claim |
| overbroad_holding | 23 | claim states the rule without a condition the quote states ("unless", "only if", "subject to" ...). Source paragraphs were chosen because they contain a condition word; the dropped words must occur in the paragraph and not in the claim; a filter model must not judge the broadened claim fully supported |
| swapped_party_or_outcome | 33 | one party role or outcome swapped; a bare negation is rejected |
| altered_quote | 42 | negation or a changed number only |
| wrong_code | 12 | statute claim cites a BNS section for a pre-1 July 2024 judgment (carries the IPC/BNS surface cue, as in v0) |

Dev split: 89 items; test split: 343 items; split by source judgment, seed 13, so no judgment is in both.
Generator model gpt-4.1 (a different model from the gpt-4o-mini checker used in v0). Counts per type and the
rejection counts are in `bench_v1_manifest.json`.

**Not built: wrong_court_binding.** Every judgment in the corpus is a Supreme Court judgment, which binds all
courts under Art. 141, so no High Court or bench-strength mismatch can be derived from the corpus metadata.
Building it needs High Court data (see `docs/legal_india_data.md`).

### Label quality (hand check)

I read 30 stratified items (7 near_paragraph, 6 overbroad_holding, 6 swapped, 3 altered, 8 controls).
Clearly mislabelled: 0. Questionable: 2 of 30 (7%): one near_paragraph item whose cited paragraph partly
covers the claim (a related writ petition description), and one whose cited paragraph echoes a detail of
the claim (the weapon). One swap ("can be entertained" to "cannot be entertained") slipped past the
negation guard, so it overlaps with altered_quote; I counted 1 such item among the 38 swaps. These are
author judgments on a small sample; no lawyer validated any label. During pilots, three label-quality
problems were found and fixed before the final generation: a near_paragraph candidate that restated the
claim (fixed by the similarity cap and a three-way filter), overbroad items that dropped a stylistic phrase
(fixed by requiring a condition word and a filter check), and swaps that were bare negations.

### v1: what ran

| Item | Status |
|---|---|
| Generation (432 items) | done |
| B0 existence only | done (no cost) |
| B2 NLI only (cross-encoder/nli-roberta-base, local) | done (no cost); threshold tuned on dev only (0.7) |
| R wrong-code rule (M1) | done (no cost) |
| B1 LLM only (gpt-4o-mini) on v1 | **not run**: the user withdrew the OpenAI budget and the project then hit its spend limit (HTTP 429) |
| B1b stronger LLM checker (gpt-4.1) | **skipped for cost**. A first run completed it before the budget was withdrawn, but its decisions were only held in memory and were lost when the job was stopped; they were never saved or reported |
| Risk-coverage calibration | done for B2 only (B1 not run) |

### v1 results (test split, n = 343, 95% bootstrap CI, 2000 resamples)

Overall (flag = positive class):

| System | Precision | Recall | F1 | False-flag rate on controls |
|---|---|---|---|---|
| B0 existence only | 1.00 | 0.18 [0.13, 0.23] | 0.31 [0.24, 0.38] | 0.00 |
| B2 NLI only | 0.86 | 0.87 [0.82, 0.91] | 0.86 [0.83, 0.89] | 0.31 [0.22, 0.39] |
| R wrong-code rule | 1.00 | 0.05 [0.03, 0.08] | 0.10 [0.05, 0.15] | 0.00 |

Per-type recall (n per type in the Types table):

| Type | B0 | B2 | R |
|---|---|---|---|
| fabricated_case | 1.00 | 1.00 | 0.00 |
| wrong_paragraph | 0.00 | 0.98 | 0.00 |
| near_paragraph | 0.00 | 0.92 [0.83, 1.00] | 0.00 |
| swapped_party_or_outcome | 0.00 | 0.76 [0.60, 0.91] | 0.00 |
| altered_quote | 0.00 | 0.81 [0.69, 0.92] | 0.00 |
| overbroad_holding | 0.00 | 0.52 [0.30, 0.73] | 0.00 |
| wrong_code | 0.00 | 1.00 | 1.00 |

Per-type precision and F1 (against controls of the same kind) are in `results_baselines_v1.json`.

B2's false flags: 34 of 111 controls overall; 13 of 42 verbatim case controls and 15 of 40 paraphrase controls.
So the NLI model's false alarms are not mainly a paraphrase effect: it flags about a third of faithful claims
even when the claim is a verbatim sentence from the paragraph (PDF extraction damages the text).

Risk-coverage for B2 (answer only the most confident share of items): accuracy 0.99 at 20% coverage, 0.93 at
40%, 0.88 at 60%, 0.87 at 80%, 0.81 at 100%; area under the risk-coverage curve 0.083 (lower is better). The
confidence is distance of the NLI score from the dev-tuned threshold, so abstention helps here.

### v0 versus v1 (overall F1 on the test split)

| System | v0 (n = 204) | v1 (n = 343) |
|---|---|---|
| B0 existence only | 0.36 [0.27, 0.45] | 0.31 [0.24, 0.38] |
| B1 LLM only (gpt-4o-mini) | 0.99 [0.98, 1.00] | not run |
| B2 NLI only | 0.96 [0.94, 0.98] | 0.86 [0.83, 0.89] |
| R wrong-code rule | 0.22 [0.14, 0.30] | 0.10 [0.05, 0.15] |

The two sets have different compositions, so the comparison is indicative, not controlled.

### Does v1 leave room for improvement?

- For the **NLI baseline: yes.** Its F1 fell from 0.96 to 0.86, with recall 0.52 on overbroad claims and
  0.76 on swaps, and 31% false flags on correct claims.
- For the **LLM-only baseline: unknown.** It was the baseline at ceiling on v0 (0.99) and it was not run on
  v1. Whether v1 is also hard for an LLM checker, which decides whether a legal verifier can show a gain,
  has not been tested. The two harder families (overbroad, swapped) also have the smallest test samples
  (23 and 33), so their intervals are wide.
- No claim is made that any verifier built on this benchmark improves on a baseline. No such verifier has been run.

### OpenAI spend on the v1 work

Measured from the usage logs: about USD 1.78. Pilots 0.55 (four pilot runs), final generation 1.00, and a
baseline run that was stopped after finishing B1 (0.03) and B1b (0.20) in memory. Not measured: a first
full generation that crashed on a network error before saving (its log was lost; by token volume it was
roughly the size of the successful run, so about 0.9 is a rough estimate) and a short debug script (small,
about 0.03). Estimated total therefore about USD 2.7, of which USD 1.78 is measured. The v0 work spent a
further USD 0.05 (generation 0.015, B1 0.032). After the budget was withdrawn, one B1 attempt was rejected
by the API with a 429 error and cost nothing; no further OpenAI calls were made.

### Limitations specific to v1

- Items are LLM-generated and labels are automatic by construction; no lawyer validated them. The
  near_paragraph and overbroad items rely on a smaller filter model (gpt-4.1-mini); items it judged
  "supported" were removed, which can bias what remains toward cases an LLM finds easy.
- Only 23 test overbroad items; overbroad source paragraphs were selected because they contain a condition,
  so they are not a random sample of holdings.
- The wrong_code items still carry the IPC/BNS surface cue; the IPC-to-BNS table is unverified.
- One corpus, one court, two years; 199 judgments.
- The hand check is 30 items by one person.

### Reproduce v1

```
cd backend
python -m scripts.legal_india_bench pilot-v1                 # small, paid (OpenAI)
python -m scripts.legal_india_bench generate-v1              # paid (OpenAI): about USD 1
python -m scripts.legal_india_bench run-v1                   # free: B0, B2 (local NLI), rule; skips B1
python -m scripts.legal_india_bench run-v1 --allow-paid      # also runs B1 (gpt-4o-mini), test split only
python -m scripts.legal_india_bench spot --v1 -n 30
```

## v0: what it is

286 items built from 199 Supreme Court judgments (2023 and 2024; the corpus sample described in
`docs/legal_india_data.md`). Case items use a matched design: one proposition drawn from one paragraph
yields a control plus four corruptions.

| Type | Count (dev / test) | Should be flagged | Construction |
|---|---|---|---|
| none (control) | 18 / 44 | no | faithful proposition, correct case and paragraph (case controls and statute controls both included; test statute controls: 9) |
| fabricated_case | 15 / 35 | yes | citation `YYYY INSC nnnn` with a serial far above the real range, verified not to resolve |
| wrong_case | 15 / 35 | yes | real citation of another corpus case, paragraph with low word overlap with the claim |
| wrong_paragraph | 15 / 35 | yes | right case, a paragraph with low word overlap with the claim |
| altered_quote | 15 / 35 | yes | LLM rewrites the proposition to change its meaning (negation, number, outcome, actor) |
| wrong_code | 4 / 20 | yes | statute claim cites a BNS section; the judgment predates 1 July 2024, so the offence did too and the IPC governs |

(Counts match `bench_v0_manifest.json`, which is authoritative. Dev total 82, test total 204.)

Dev/test split is 30/70 by **source judgment**, so no judgment appears in both. Seed 13, temperature 0,
generation model gpt-4o-mini. Generation cost: about USD 0.015 (all attempts).

For wrong_code, the label does not depend on the correctness of the IPC-to-BNS table: any BNS citation
about a pre-July-2024 offence is wrong. The table (unverified, from `section_map.json`) only chooses a
realistic section number.

## Baselines

- **B0 existence only:** flags a case citation that does not resolve to a corpus case. Nothing else.
- **B1 LLM only:** gpt-4o-mini sees the claim and the cited paragraph and answers supported or
  unsupported. No NLI, no rules. An unresolvable citation (no text) is flagged without a call.
- **B2 NLI only:** `cross-encoder/nli-roberta-base`; best entailment of the claim by the cited paragraph
  (3-sentence windows and the whole paragraph); flagged below a threshold tuned on **dev only**
  (chosen: 0.9, the largest grid value below the maximum of 0.95, which suggests the model is weakly
  calibrated for this text).
- **R (M1 rule):** the transition rule from `app.legal_india.transition`, wrong_code only. Shown for
  reference, not a general baseline.

## Results (test split, n = 204; 95% bootstrap CI, 2000 resamples)

Overall detection (flag = positive class):

| System | Precision | Recall | F1 | False-flag rate on controls |
|---|---|---|---|---|
| B0 existence only | 1.00 [1.00, 1.00] | 0.22 [0.15, 0.29] | 0.36 [0.27, 0.45] | 0.00 |
| B1 LLM only | 1.00 [1.00, 1.00] | 0.98 [0.96, 1.00] | 0.99 [0.98, 1.00] | 0.00 |
| B2 NLI only | 0.93 [0.90, 0.97] | 0.99 [0.97, 1.00] | 0.96 [0.94, 0.98] | 0.25 [0.13, 0.37] |
| R wrong-code rule | 1.00 [1.00, 1.00] | 0.13 [0.08, 0.18] | 0.22 [0.14, 0.30] | 0.00 |

Per-type recall (share of corrupted items flagged):

| Type (n) | B0 | B1 | B2 | R |
|---|---|---|---|---|
| fabricated_case (35) | 1.00 | 1.00 | 1.00 | 0.00 |
| wrong_case (35) | 0.00 | 1.00 | 0.97 | 0.00 |
| wrong_paragraph (35) | 0.00 | 0.91 | 1.00 | 0.00 |
| altered_quote (35) | 0.00 | 1.00 | 0.97 | 0.00 |
| wrong_code (20) | 0.00 | 1.00 | 1.00 | 1.00 |

Per-type precision and F1 (against controls of the same kind) and the CIs are in `results_baselines.json`.
B1 LLM calls cost USD 0.032 in total. Decisions per item are in `decisions_baselines.json`.

## Reading the results honestly

- B1 and B2 reach near-ceiling recall on every type, so the benchmark does not separate a strong
  auditor from a simple LLM prompt. A system built on top cannot show an advantage here.
- Why it is easy: the corruptions are blunt. A wrong_case or wrong_paragraph item cites text that shares
  almost no vocabulary with the claim (by construction), and a fabricated citation has no text at all.
- wrong_code is flagged by B1 and B2 without any legal knowledge: the claim says BNS while the cited
  judgment paragraph says IPC, a surface mismatch. The rule check R also finds all of them, but only them.
- B2 has a 25% false-flag rate: the NLI model rarely gives high entailment to long, OCR-damaged legal
  paragraphs, so a threshold that catches corruptions also flags faithful claims.
- Sample sizes are small (20 to 35 per type, 9 statute controls). CIs for per-type values are wide, and
  rates of exactly 0 or 1 have CIs that understate uncertainty under the bootstrap.

## Limitations and sources of bias

- Items are **LLM-generated** from judgments and **labels are automatic by construction**. No lawyer has
  validated them.
- A paragraph can contain several propositions; a proposition may be supported by other paragraphs too,
  so "wrong paragraph" is only approximately wrong.
- Spot checks by the author (32 items in two samples of 20 and 12, on slightly different generations of the
  set): no label was clearly wrong, but about 3 distinct items had weak construction: a base proposition
  that goes beyond its quote (the "dismissal" item, seen twice), an unsupported "only", and a mild
  alteration ("transfer to" became "withdraw from"). Roughly one item in ten may have a weak base or
  alteration, which makes the benchmark slightly noisy rather than invalid.
- PDF extraction splits words, so NLI and LLM judges see damaged text; this affects B2 most.
- The corpus is 199 judgments of one court and two years. Only 36 statute items could be built because
  few paragraphs mention an IPC section the table covers.
- The IPC-to-BNS table is unverified. The decision rule for wrong_code (offence date before 1 July 2024)
  is simple and ignores offences spanning the cut-off.
- Generator and baseline B1 use the same model family (gpt-4o-mini), which can inflate B1.
- Judgment text is quoted inside the items. The licence for redistribution of judgment texts is unconfirmed
  (see `docs/legal_india_data.md`), so `bench_v0.jsonl` should not be published until it is.

## What to fix before relying on it

1. Harder, near-miss corruptions: adjacent paragraphs of the same case, other cases on the same statute or
   topic, and subtle alterations (scope words such as "only", dates, party swaps).
2. Real citation errors from the wild (misquoted holdings in briefs) with lawyer labels.
3. Remove the BNS-versus-IPC surface cue (also include correct BNS citations for post-cut-off offences).
4. A different model family for the generator and for the baselines.
5. At least 100 claims labelled by two people with legal training, with agreement reported.

## Reproduce

```
cd backend
python -m scripts.legal_india_bench pilot      # cost estimate
python -m scripts.legal_india_bench generate --wrong-code 24 --statute-none 12
python -m scripts.legal_india_bench run
python -m scripts.legal_india_bench spot -n 20
```
