"""Builds paper/main.html and paper/main.pdf. Every number is computed from the repository's result files.

    python figures.py   # figures
    python build.py     # HTML + PDF (needs Chrome or Edge for the PDF step)
"""

from __future__ import annotations

import html
import subprocess
import sys
from pathlib import Path

import data
from data import ci, mci

HERE = Path(__file__).resolve().parent
V0, V1, MQ = data.v0, data.v1, data.musique
S0, S1 = V0["systems"], V1["systems"]
MV = MQ["variants"]
PAIRED = MQ["paired"]

# ------------------------------------------------------------------------------------------ numbers
m0, m1 = data.v0_manifest, data.v1_manifest
c0, c1 = m0["counts_by_split_and_type"], m1["counts_by_split_and_type"]
dev0 = sum(v for k, v in c0.items() if k.startswith("dev/"))
test0 = sum(v for k, v in c0.items() if k.startswith("test/"))
dev1 = sum(v for k, v in c1.items() if k.startswith("dev/"))
test1 = sum(v for k, v in c1.items() if k.startswith("test/"))
assert (dev0, test0, dev1, test1) == (82, 204, 89, 343), (dev0, test0, dev1, test1)

b0_v1 = S1["B0_existence_only"]["test"]["overall"]
b2_v1 = S1["B2_nli_only"]["test"]["overall"]
b2_v0 = S0["B2_nli_only"]["test"]["overall"]
b1_v0 = S0["B1_llm_only"]["test"]["overall"]
b0_v0 = S0["B0_existence_only"]["test"]["overall"]
r_v1 = S1["R_wrong_code_rule_m1"]["test"]["overall"]
r_v0 = S0["R_wrong_code_rule_m1"]["test"]["overall"]
pt1 = S1["B2_nli_only"]["test"]["per_type"]
rc = V1["calibration_risk_coverage_test"]["B2_nli_only"]
cov_acc = dict(zip(rc["coverage"], rc["accuracy"]))
lat = {k: MV[k]["overall"]["total_latency_ms"]["mean"] for k in ["B", "A", "A-", "V", "A+V"]}
ratio_v, ratio_av = lat["V"] / lat["B"], lat["A+V"] / lat["B"]
corr = {k: MV[k]["overall"]["correct"] for k in ["B", "A", "A-", "V", "A+V"]}
GRP = MQ["classifier"]["by_group"]
chunks_g = {k: {g: MV[k]["by_group"][g]["num_chunks"]["mean"] for g in ["simple", "moderate", "complex"]}
            for k in ["B", "A", "A+V"]}
P_av = PAIRED["A+V"]


def fx(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


# ------------------------------------------------------------------------------------------ html helpers
def esc(s: str) -> str:
    return html.escape(s, quote=False)


def figure(name: str, caption: str, wide: bool = False, width: str | None = None) -> str:
    cls = "fig wide" if wide else "fig"
    style = f' style="width:{width}"' if width else ""
    num = FIG_NUM[name]
    return (f'<figure class="{cls}"><img src="figures/{name}.png"{style} alt="Figure {num}">'
            f'<figcaption>Fig. {num}. {caption}</figcaption></figure>')


def table(num: int, caption: str, header: list[str], rows: list[list[str]], wide: bool = False, cls: str = "") -> str:
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    w = "tbl wide" if wide else "tbl"
    return (f'<div class="{w}"><div class="tcap"><div class="tn">TABLE {ROMAN[num]}</div><div class="tc">{caption}</div></div>'
            f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX"}
LETTER = "ABCDEFGH"
FIG_NUM = {"fig1_pipeline": 1, "fig2_taxonomy": 2, "fig3_f1_v0_v1": 3, "fig4_recall_by_type": 4,
           "fig5_false_flags": 5, "fig6_risk_coverage": 6, "fig7_musique_main": 7, "fig8_musique_sweeps": 8,
           "fig9_classifier": 9}

# ------------------------------------------------------------------------------------------ tables
TYPE_ROWS = [("none", "Faithful control"), ("fabricated_case", "Fabricated case"), ("wrong_case", "Wrong case"),
             ("wrong_paragraph", "Wrong paragraph"), ("near_paragraph", "Near paragraph"),
             ("overbroad_holding", "Over-broad holding"), ("swapped_party_or_outcome", "Swapped party or outcome"),
             ("altered_quote", "Altered quote"), ("wrong_code", "Wrong code (IPC/BNS)")]


def cnt(c, split, t):
    v = c.get(f"{split}/{t}", 0)
    return str(v) if v else "&ndash;"


t1 = table(1, "Benchmark composition (items per split; dev / test). Counts are read from bench_v0_manifest.json and "
              "bench_v1_manifest.json. Splits are by source judgment (no judgment appears in both).",
           ["Item type", "v0 dev", "v0 test", "v1 dev", "v1 test"],
           [[lbl, cnt(c0, "dev", t), cnt(c0, "test", t), cnt(c1, "dev", t), cnt(c1, "test", t)] for t, lbl in TYPE_ROWS]
           + [["<b>Total</b>", f"<b>{dev0}</b>", f"<b>{test0}</b>", f"<b>{dev1}</b>", f"<b>{test1}</b>"]])


def overall_row(label, m, ff=True):
    return [label, ci(m["precision"]), ci(m["recall"]), ci(m["f1"]), ci(m["false_flag_rate"])]


t2 = table(2, "Overall detection on the test split (flag = positive class); 95% bootstrap CIs (2,000 resamples). "
              "B1 (LLM-only) was run on v0 only. Source: results_baselines.json, results_baselines_v1.json.",
           ["System", "Precision", "Recall", "F1", "False-flag rate"],
           [["<i>v0 (n = %d)</i>" % b1_v0["n"], "", "", "", ""],
            overall_row("B0 existence only", b0_v0), overall_row("B1 LLM only (gpt-4o-mini)", b1_v0),
            overall_row("B2 local NLI", b2_v0), overall_row("R IPC/BNS date rule", r_v0),
            ["<i>v1 (n = %d)</i>" % b2_v1["n"], "", "", "", ""],
            overall_row("B0 existence only", b0_v1), overall_row("B1 LLM only", {k: {"value": float("nan"), "ci95": [float("nan")] * 2} for k in ("precision", "recall", "f1", "false_flag_rate")}),
            overall_row("B2 local NLI", b2_v1), overall_row("R IPC/BNS date rule", r_v1)],
           wide=True, cls="t2")
t2 = t2.replace("nan [nan, nan]", "not run")

PT_ORDER = [("fabricated_case", "Fabricated case"), ("wrong_paragraph", "Wrong paragraph"),
            ("near_paragraph", "Near paragraph"), ("altered_quote", "Altered quote"),
            ("swapped_party_or_outcome", "Swapped party or outcome"), ("overbroad_holding", "Over-broad holding"),
            ("wrong_code", "Wrong code")]
t3 = table(3, "Per-type recall on v1 (test split) with 95% bootstrap CIs; n is the number of corrupted items of that "
              "type. Rows in the same order as Figure 4. Source: results_baselines_v1.json.",
           ["Type", "n", "B0 existence", "B2 local NLI", "R date rule"],
           [[lbl, str(pt1[t]["n_type"]), ci(S1["B0_existence_only"]["test"]["per_type"][t]["recall"]),
             ci(pt1[t]["recall"]), ci(S1["R_wrong_code_rule_m1"]["test"]["per_type"][t]["recall"])]
            for t, lbl in PT_ORDER], wide=True)

names = [("B", "B Fixed top-5 (baseline)"), ("A", "A Adaptive retrieval"), ("A-", "A&minus; Adaptive, no decomposition"),
         ("V", "V Verification only"), ("A+V", "A+V Adaptive + verification")]
t4 = table(4, "MuSiQue ablation (36 test questions: 12 simple, 12 moderate, 12 complex). Mean [95% bootstrap CI]. "
              "Correctness is judged by gpt-4.1 against the gold answer; recall is measured against human-annotated "
              "MuSiQue supporting paragraphs. Latency is relative only (several queries ran concurrently on one CPU). "
              "Source: experiments/results/main/summary.json.",
           ["Variant", "Correct", "Recall@k", "Chunks", "Latency (s)", "Tokens/query"],
           [[lbl, mci(MV[k]["overall"]["correct"]), mci(MV[k]["overall"]["r_at_k"]),
             fx(MV[k]["overall"]["num_chunks"]["mean"]),
             fx(MV[k]["overall"]["total_latency_ms"]["mean"] / 1000, 1),
             f"{MV[k]['overall']['total_tokens']['mean']:.0f}"] for k, lbl in names], wide=True, cls="t4")


def dif(p, m, scale=1.0, d=3, unit=""):
    x = p[m]["diff"]
    return f"{x['mean'] * scale:+.{d}f} [{x['lo'] * scale:+.{d}f}, {x['hi'] * scale:+.{d}f}]{unit}"


t5 = table(5, "Paired differences against the fixed top-5 baseline (same 36 questions; paired bootstrap, 10,000 "
              "resamples). McNemar is the exact test on judged correctness (only the baseline correct / only the "
              "variant correct). Source: summary.json, paired.",
           ["Variant vs. B", "Correct", "McNemar p", "Unsupported rate", "Chunks", "Latency (s)"],
           [[lbl.split(" ", 1)[0], dif(PAIRED[k], "correct"), f"{PAIRED[k]['correct']['mcnemar_p']:.4f} "
             f"({PAIRED[k]['correct']['only_b']} vs {PAIRED[k]['correct']['only_a']})",
             dif(PAIRED[k], "unsupported_rate"), dif(PAIRED[k], "num_chunks", d=2),
             dif(PAIRED[k], "total_latency_ms", 1 / 1000, 1)] for k, lbl in names[1:]], wide=True, cls="t5")

t6 = table(6, "How this study differs from the base paper (Verma, 2026), as far as can be read from its arXiv abstract "
              "page. We did not compare against that system and make no claim of superiority.",
           ["Aspect", "Base paper (Verma, 2026)", "This study"],
           [["Corpora", "Two legal corpora of court opinions and briefs (not named on the abstract page)",
             "199 Supreme Court of India judgments (2023 to 2024), one court"],
            ["Error types", "Wrong cited case; wrong pinpoint page within the same case",
             "Fabricated case, wrong paragraph, near paragraph, over-broad holding, swapped party/outcome, altered "
             "quote, wrong criminal code for the offence date (IPC vs. BNS)"],
            ["Verifiers", "Fourteen LLM configurations, including reasoning models",
             "Existence check, local NLI model, rule check; an LLM-only checker (gpt-4o-mini) on v0 only"],
            ["Items", "Perturbations of real citations", "LLM-generated propositions from real paragraphs, then "
             "perturbed; labels automatic by construction"],
            ["Jurisdiction-specific checks", "Not applicable (as far as we can tell)",
             "IPC/BNS date rule with an unverified section table"],
            ["Did NOT do", "&ndash;", "No LLM checker on v1; no head-to-head with the base paper; no lawyer-"
             "labelled items; no High Court data; no good-law (overruled) check; no evaluation of a combined legal "
             "verifier"]], wide=True, cls="t6")

# ------------------------------------------------------------------------------------------ body
ABSTRACT = (
    f"Citations that point to a real authority which does not support the proposition they are cited for are a known "
    f"failure mode of legal AI tools, and an existence lookup cannot catch them. We study this failure in an Indian "
    f"setting. From {data.fetch['written']} Supreme Court of India judgments (2023 to 2024) we build a "
    f"citation-support benchmark in two versions (v0: {m0['n_items']} items; v1: {m1['n_items']} items). Each item "
    f"pairs a faithful proposition with controlled corruptions: fabricated citations, wrong paragraphs, altered quotes, "
    f"the wrong criminal code for the offence date, and in v1 three near-miss errors (a similar paragraph of the same "
    f"judgment, an over-broad holding, swapped parties or outcome). Items are LLM-generated and labelled automatically "
    f"by construction; no lawyer has validated them. We evaluate an existence check, a locally run NLI model and a date rule. The "
    f"existence check catches only fabricated citations (F1 {fx(b0_v1['f1']['value'])} on v1). The local NLI baseline "
    f"reaches F1 {ci(b2_v1['f1'])} on v1, down from {fx(b2_v0['f1']['value'])} on v0, but falsely flags "
    f"{fx(b2_v1['false_flag_rate']['value'] * 100, 0)}% of faithful claims and recovers only "
    f"{fx(pt1['overbroad_holding']['recall']['value'] * 100, 0)}% of over-broad holdings; answering only its most "
    f"confident 20% of items raises its accuracy to {fx(cov_acc[0.2])}. An LLM-only checker scored F1 "
    f"{fx(b1_v0['f1']['value'])} on v0 but was <i>not evaluated</i> on v1, so we cannot say how hard v1 is "
    f"for LLM checkers, and we claim no advantage over any prior system. Separately, we report a negative result for a "
    f"general adaptive-retrieval and claim-verification pipeline: on 36 MuSiQue questions it chose retrieval depth as "
    f"designed (about {fx(chunks_g['A']['simple'], 0)}, {fx(chunks_g['A']['moderate'], 0)} and "
    f"{fx(chunks_g['A']['complex'], 0)} chunks for simple, moderate and complex questions) but did not improve judged "
    f"answer correctness ({fx(corr['A+V']['mean'], 3)} against {fx(corr['B']['mean'], 3)} for fixed top-5) and was "
    f"{fx(ratio_v, 1)} to {fx(ratio_av, 1)} times slower with verification."
)

BODY = f"""
<h2>1&nbsp;&nbsp;Introduction</h2>
<p>Retrieval-augmented legal assistants cite their sources, and users reasonably read a citation as evidence. The
evidence is weaker than it looks. Magesh et al. [3] found that commercial legal research tools marketed as
mitigating hallucination still produce incorrect or unsupported statements, and Verma [1] isolates a specific
mechanism: a citation can point to a real case, even the right case, while the cited text does not support the
proposition. Models in that study caught wrong-case substitutions almost always (93&ndash;100%) but missed many
wrong-pinpoint errors (37&ndash;61% caught on court opinions), apparently accepting citations on topical overlap
rather than on support. A database lookup cannot detect this.</p>
<p>We ask a narrower, practical question for Indian law: <i>how much of this problem can a locally run
checker catch, and how does the answer change when the errors are near misses?</i> We build a benchmark
from Supreme Court of India judgments, measure three baselines, and report what they do and do not
catch. We also report, as a negative result, how a more elaborate general-purpose pipeline (complexity-adaptive
retrieval plus claim verification) behaved on a standard multi-hop benchmark.</p>
<p><b>Contributions, conservatively stated.</b> (i) A citation-support benchmark over Indian Supreme Court judgments
with seven corruption types in v1 (eight across v0 and v1), including three near-miss types and a statute-code transition error specific to the
1 July 2024 replacement of the IPC by the BNS (labels automatic, not lawyer-validated). (ii) Measurements of an
existence check, a local NLI model and a rule check on it, showing that the near-miss types are harder than the blunt
ones, that the NLI baseline pays for its recall with many false flags, and that abstention on low-confidence items
helps. (iii) A negative result for a general adaptive-and-verified RAG pipeline on 36 MuSiQue questions.</p>
<p><b>What is new and what is not.</b> Claim-level verification, NLI-based support checking, complexity-adaptive
retrieval [4] and citation-perturbation evaluation [1] are not new. What we add is the Indian setting, the near-miss
types and the IPC/BNS error type, as far as we know. We
did not run any LLM checker on v1 and did not compare against any prior system.</p>

<h2>2&nbsp;&nbsp;Related Work</h2>
<p><b>Citation support in legal AI.</b> Verma [1] (arXiv:2608.12571; accepted to the 1st Workshop on AI for Law at
ICML 2026 according to its arXiv page) perturbs real citations from court opinions and briefs, replacing the cited
case or changing only the pinpoint page, and evaluates fourteen model configurations; page-level verification
prompts raise recall but also false positives, and the author concludes that recognising the right legal topic and
verifying support are distinct capabilities. This is our base paper. Magesh et al. [3] evaluate commercial legal
research tools on a preregistered query set and find vendors' hallucination-free claims overstated.
<b>Claim auditing.</b> GANDR [2] pairs an answer-writing agent with a critic that audits each claim against its cited
source, reporting 70.8% strict accuracy on a 185-item legal benchmark and an audit F1 of 0.84 as a detector of
under-supported claims (figures as stated on the arXiv abstract page; we have not read the full paper).
<b>Retrieval and adaptivity.</b> Adaptive-RAG [4] routes questions by a trained complexity classifier; Self-RAG [5]
trains a model to decide when to retrieve and to critique its output. LegalBench-RAG [6] benchmarks the retrieval step
over legal documents. Our general pipeline differs from [4] by using an untrained hybrid complexity score and by
adding claim verification with bounded retries, but we do not evaluate it against [4] or [5].</p>

<h2>3&nbsp;&nbsp;Benchmark</h2>
<p><b>Corpus.</b> We sampled {data.fetch['sampled']} English judgments at random (seed 13; 100 from 2023 and 100
from 2024) from the Indian Supreme Court Judgments open dataset [8] and extracted text for
{data.fetch['written']} (one download failed with HTTP 404; no PDF was scanned or empty). Paragraphs were split on the
court's own numbering for {data.fetch['numbered']} judgments ({fx(data.fetch['numbered'] / data.fetch['written'] * 100, 0)}%)
and on blank lines for the other {data.fetch['blank_line']}; the mean is {data.fetch['paragraphs_mean']} paragraphs
per judgment (median {data.fetch['paragraphs_median']}). The registry lists the licence as CC-BY-4.0, but it is not
clear whether that covers the judgment texts or only the compilation, so we do not redistribute any judgment text.</p>
<p><b>Items and taxonomy.</b> An item is a claim, a cited authority (a case citation with a paragraph, or a statute
section) and a label saying whether a careful checker should flag it. Case items follow a matched design: one
proposition drawn from one paragraph yields a faithful control and several corruptions, so every corruption has a
control from the same source (Table 1; Figure 2). <i>Surface-level</i> types: a fabricated citation that does not
resolve; a real paragraph of another case or an unrelated paragraph of the right case; a claim whose meaning is changed
by a negation or number edit; and a statute claim citing a BNS section for a judgment that predates 1 July 2024.
<i>Near-miss</i> types (v1 only): the most similar <i>other</i> paragraph of the same judgment (similarity capped, and
a filter model judged that it does not state the claim); an over-broad holding that drops a condition such as
"unless" or "only if"; and one party role or outcome swapped.</p>
<p><b>Generation and filters.</b> v0 propositions and alterations were generated with gpt-4o-mini;
v1 used gpt-4.1 with gpt-4.1-mini as a filter and text-embedding-3-small for similarity. Items are rejected if the supporting quote is not
found in the paragraph, if a perturbation does not differ from its original, or, for near paragraphs and over-broad
holdings, if the filter model judges the corrupted claim still supported ({m1['generation_stats']['rejected']['near_candidate_supports_claim']}
near-paragraph candidates and {m1['generation_stats']['rejected']['overbroad_still_supported']} over-broad candidates were
rejected for that reason). This filtering may bias what remains toward cases an LLM finds easy.</p>
<p><b>Label quality.</b> Labels are automatic by construction and no lawyer has validated them. The author read 30
stratified v1 items: none was clearly mislabelled and 2 (7%) were questionable; one swap ("can" to "cannot") slipped
past the negation guard. A similar read of 32 v0 items found no clearly wrong label and about 3 weakly constructed
items. <i>Not built:</i> a wrong-court-binding type, because every judgment is from the Supreme Court, which binds all
courts under Article 141.</p>

{figure("fig2_taxonomy", f"Benchmark composition by item type and split (darker = test, lighter = dev). v0 has {m0['n_items']} items and v1 {m1['n_items']}. Near-miss types (orange) exist only in v1. Data: bench_v0_manifest.json, bench_v1_manifest.json.", wide=True)}
{t1}

<h2>4&nbsp;&nbsp;Methods</h2>
<p><b>Checkers.</b> <i>B0 (existence only)</i> flags a case citation that does not resolve to a corpus case.
<i>B1 (LLM only)</i> shows gpt-4o-mini the claim and the cited paragraph and asks for supported or unsupported.
<i>B2 (local NLI)</i> scores entailment of the claim by the cited paragraph with <tt>cross-encoder/nli-roberta-base</tt>
[9], using 3-sentence windows and the whole paragraph, and flags the claim when the best entailment is below a
threshold tuned on the dev split only. <i>R (date rule)</i> flags a statute citation whose code (IPC or BNS) is wrong
for the offence date, using the date of the judgment as a proxy; its IPC/BNS section table is partial and
<b>unverified</b>.</p>
<p><b>General pipeline (evaluated on MuSiQue [7] only).</b> A hybrid complexity score
S = (0.35&middot;needs + 0.25&middot;reasoning + 0.15&middot;multi-source + 0.15&middot;lexical + 0.10&middot;LLM-label),
normalised by the weight sum, sets retrieval depth k by label band (simple 1&ndash;3, moderate 4&ndash;7, complex
8&ndash;12; interpolated within the band). Multi-part questions are searched per sub-question and fused by Reciprocal
Rank Fusion; low retrieval confidence triggers a bounded expansion. The generated answer is split into claims, each
checked against its cited passage by an NLI model (support if entailment &ge; 0.75, contradiction if &ge; 0.75) with a
quote-validated LLM judge for ambiguous cases; unsupported claims trigger targeted retrieval and regeneration (at most
two retries), and unsupported sentences are removed with an uncertainty notice. The pipeline and the legal checks are
separate: we did <b>not</b> evaluate the pipeline on the legal benchmark, and no combined legal verifier was built.</p>
{figure("fig1_pipeline", "Systems in this paper. Top: the general adaptive-retrieval and verification pipeline, evaluated on MuSiQue (Section 6.4). Bottom: the legal citation checks evaluated on the Indian benchmark; B1 was run on v0 only.", wide=True)}

<h2>5&nbsp;&nbsp;Experiments</h2>
<p><b>Legal benchmark.</b> Items are split 30/70 into dev and test by source judgment (seed 13) so that no judgment
appears in both. B2's threshold is tuned on dev only and metrics are reported on test only
({test0} items for v0, {test1} for v1). The flag is the positive class; we report precision, recall, F1 and the
false-flag rate on faithful controls with 95% bootstrap intervals (2,000 resamples). Per-type precision and F1 use
the controls of the same kind. For abstention we rank test items by the distance of the NLI score from the threshold
and report accuracy on the most confident share (a risk-coverage curve). B2's tuned threshold is
{V0['b2_threshold_tuned_on_dev']} on v0 and {V1['b2_threshold_tuned_on_dev']} on v1.</p>
<p><b>MuSiQue.</b> 36 test questions (12 each from single-hop decompositions, 2-hop and 3&ndash;4-hop questions) over a
954-paragraph corpus built from their gold and distractor paragraphs. The pipeline uses gpt-4o-mini and
text-embedding-3-small with <tt>nli-roberta-base</tt> as verifier; evaluation uses gpt-4.1 as judge and a different NLI
model (<tt>nli-MiniLM2-L6-H768</tt>) for evidence metrics. Variants: fixed top-5 (B), adaptive depth with
sub-question retrieval (A), adaptive without decomposition (A&minus;), verification only (V) and both (A+V), plus a
fixed-k sweep and a retry sweep. Paired differences use a paired bootstrap (10,000 resamples) and an exact McNemar
test. <i>No answers or claims were labelled by humans</i>; only the retrieval gold paragraphs are human-annotated.</p>

<h2>6&nbsp;&nbsp;Results</h2>
<h3>6.1&nbsp;&nbsp;Overall detection</h3>
<p>Table 2 and Figure 3 give the overall picture. An existence check finds fabricated citations and nothing else
(F1 {fx(b0_v0['f1']['value'])} on v0, {fx(b0_v1['f1']['value'])} on v1). The LLM-only checker reached F1
{ci(b1_v0['f1'])} on v0, which shows that v0 cannot separate a strong auditor from a plain prompt. The local NLI
baseline fell from {ci(b2_v0['f1'])} on v0 to {ci(b2_v1['f1'])} on v1. The two sets differ in composition, so this is
indicative and not a controlled comparison. The date rule alone has precision 1.00 but recall
{fx(r_v1['recall']['value'])} on v1, because it can only address one error type.</p>
{t2}
{figure("fig3_f1_v0_v1", "Overall F1 on the test split (95% CIs). The LLM-only checker (B1) was not evaluated on v1, so v1 has not been shown to be harder than v0 for it. Data: results_baselines.json, results_baselines_v1.json.")}

<h3>6.2&nbsp;&nbsp;Where the NLI baseline fails</h3>
<p>The NLI baseline's recall on v1 is high for blunt errors (fabricated case, wrong paragraph, wrong code) and for
near paragraphs ({ci(pt1['near_paragraph']['recall'])}), but lower for altered quotes
({ci(pt1['altered_quote']['recall'])}), swapped parties or outcomes ({ci(pt1['swapped_party_or_outcome']['recall'])})
and especially over-broad holdings ({ci(pt1['overbroad_holding']['recall'])}; n = {pt1['overbroad_holding']['n_type']}).
An over-broad claim is largely entailed by the paragraph that states the stricter rule, which is what an entailment
model is built to accept. The intervals for swapped parties or outcomes and over-broad holdings are wide because those
types have only 33 and 23 test items.</p>
{t3}
{figure("fig4_recall_by_type", "Recall by item type for each checker on v0 (left) and v1 (right), with 95% CIs. Orange tick labels mark the near-miss types. Missing bars mean the checker was not run or the type does not exist in that version. Data: results_baselines.json, results_baselines_v1.json.", wide=True)}

<h3>6.3&nbsp;&nbsp;False flags and abstention</h3>
<p>The NLI baseline flags {fx(b2_v1['false_flag_rate']['value'] * 100, 0)}% of faithful claims on v1 (34 of 111 controls),
including 13 of 42 verbatim case controls and 15 of 40 paraphrase controls, so its false alarms are not mainly a
paraphrase effect; PDF extraction damages the text it reads (we did not isolate that cause). A recall-oriented
threshold therefore trades away reliability, a trade-off of the kind Verma [1] reports for page-level verification prompts
(we make no quantitative comparison; the verifiers differ). Abstaining on the least confident items helps:
accuracy is {fx(cov_acc[0.2], 2)} at 20% coverage, {fx(cov_acc[0.4], 2)} at 40%, {fx(cov_acc[0.6], 2)} at 60%,
{fx(cov_acc[0.8], 2)} at 80% and {fx(cov_acc[1.0], 2)} at full coverage (area under the risk-coverage curve
{fx(rc['aurc'], 3)}; Figure 6).</p>
<div class="pair">
{figure("fig5_false_flags", "False-flag rate on faithful controls (95% CIs). B1 was not run on v1. Data: results_baselines*.json.", width="100%")}
{figure("fig6_risk_coverage", "Risk-coverage curves of the NLI baseline on the test split; confidence is the distance of the NLI score from the dev-tuned threshold. Dotted lines: accuracy at full coverage. Data: decisions_baselines*.json with the item labels.", width="100%")}
</div>

<h3>6.4&nbsp;&nbsp;General pipeline on MuSiQue: a negative result</h3>
<p>Adaptive depth behaved as designed: mean chunks given to the generator were
{fx(chunks_g['A']['simple'], 1)}, {fx(chunks_g['A']['moderate'], 1)} and {fx(chunks_g['A']['complex'], 1)} for simple,
moderate and complex questions, against 5.0 for the fixed baseline (Figure 7b). The complexity label agreed with the
hop-count group for {fx(MQ['classifier']['accuracy'], 3)} of questions (Figure 9; hop count is a proxy, not a human
complexity label). It did not improve answers. Judged correctness was {mci(corr['B'])} for the baseline and
{mci(corr['A+V'])} for A+V. The paired difference for A+V is {dif(P_av, 'correct')}; the interval excludes zero but the
exact McNemar test gives p = {P_av['correct']['mcnemar_p']:.4f} (baseline correct and variant wrong on
{P_av['correct']['only_b']} questions, the reverse on {P_av['correct']['only_a']}), so the evidence is suggestive and
not conclusive at n = 36. Verification alone (V) was not clearly different ({dif(PAIRED['V'], 'correct')}).
The unsupported-claim rate did not differ reliably (A+V: {dif(P_av, 'unsupported_rate')}), citation metrics were
similarly unresolved, and retries (0, 1, 2) showed no resolved effect (Figure 8c). The drawback was latency:
{fx(ratio_v, 1)} times the baseline for V and {fx(ratio_av, 1)} times for A+V (relative only), and about
{MV['A+V']['overall']['total_tokens']['mean'] / MV['B']['overall']['total_tokens']['mean']:.1f} times the tokens
for A+V. A fixed depth of about five was best overall in the sweep (Figure 8a); correctness plateaued by k = 5 for simple
and moderate questions and peaked at k = 5 for complex ones (Figure 8b), so no group gained from deeper retrieval.</p>
{t4}
{t5}
{figure("fig7_musique_main", "MuSiQue ablation. (a) judged answer correctness with 95% CIs (B fixed top-5; A adaptive; A&minus; adaptive without decomposition; V verification only; A+V both). (b) mean chunks given to the generator by question group. (c) total latency on a log scale, relative only. Data: experiments/results/main/summary.json.", wide=True)}
{figure("fig8_musique_sweeps", "(a) fixed-k sweep, overall; (b) fixed-k sweep by question group (12 questions each); (c) retry sweep for A+V on an 18-question subset (95% CIs; mean attempts shown). Data: summary.json.", wide=True)}
{figure("fig9_classifier", f"Predicted complexity label against hop-count group on the {MQ['classifier']['n']} test questions. Data: summary.json, classifier.", width="72%")}

<h2>7&nbsp;&nbsp;Discussion</h2>
<p><b>What our results do and do not say about the base paper.</b> Verma [1] reports that LLM verifiers catch
wrong-case substitutions almost always but miss many wrong-pinpoint errors. In our setting, the nearest analogues of
a wrong case and a wrong pinpoint are our wrong-case (v0) and near-paragraph (v1) items. Our local NLI baseline caught
{ci(S0['B2_nli_only']['test']['per_type']['wrong_case']['recall'])} of wrong-case items and
{ci(pt1['near_paragraph']['recall'])} of near-paragraph items, and our LLM-only checker caught all wrong-case items on
v0. We do <b>not</b> claim to replicate or contradict [1]: the verifiers (a small NLI model against LLMs), the
corruptions (our near paragraphs were filtered so that they do not state the claim; changing only a page may leave the
supporting text largely intact), the corpora and the jurisdiction all differ. What our data supports is narrower: on
this benchmark a locally run NLI baseline is weak exactly where an entailment model is structurally weak (over-broad
claims, swaps), and it trades recall for false flags. LLM checkers were not evaluated on v1, so the central
question of whether near-miss errors are also hard for LLM verifiers in this Indian setting remains open.</p>
<p><b>Implications.</b> (i) Existence checks are not enough: they catch only fabricated citations. (ii) Abstention is
useful: the NLI baseline is far more accurate on its confident items. (iii) The general adaptive-and-verified pipeline
should not be assumed to help; on 36 questions it took much more time and gave no measured accuracy gain. A larger
evaluation with human labels would be needed before claiming either benefit or harm.</p>
{t6}

<h2>8&nbsp;&nbsp;Limitations and Ethics</h2>
<p><b>Labels and items.</b> Items are LLM-generated and labelled by construction; no lawyer validated any label, and
a paragraph can contain several propositions, so "wrong paragraph" is only approximately wrong. Source paragraphs for
over-broad holdings were chosen because they contain a condition word. The filtering described in Section 3 may bias
the benchmark toward cases an LLM finds easy. <b>Scope.</b> One court, two years, 199 judgments, English only; no High
Court data and no good-law (overruled or distinguished) information. The statute subset is small. <b>Statutes.</b> The
IPC/BNS section table is partial and unverified (it was assembled without access to the official Gazette) and must be
checked against official texts before any reliance; the date rule uses the judgment date as a proxy for the offence
date and ignores transitional provisions. <b>Baselines.</b> The LLM-only checker was run on v0 only, and it used the same
model family as the v0 generator, which may inflate it. The NLI baseline reads PDF-extracted text, which is damaged in
places. <b>MuSiQue.</b> 36 questions, automatic evaluation by an LLM judge and a strict NLI evaluator whose absolute
claim-support values are probably underestimates and were not validated by humans; latencies were measured with
concurrent queries on a CPU machine; the verifier used was a small NLI model for speed, not the application's default.
<b>Licence and ethics.</b> The licence covering the judgment texts is unconfirmed, so item files containing judgment
quotes are not released. This work is a research prototype and decision support only: it is not legal advice, it was not
evaluated by legal professionals, and any tool built on it should require human verification of every authority.</p>

<h2>9&nbsp;&nbsp;Conclusion</h2>
<p>We built an Indian Supreme Court citation-support benchmark with near-miss error types and measured three
baselines on it. <b>Positive:</b> abstention on low-confidence items markedly raises the NLI baseline's
accuracy, and the near-miss types make the benchmark harder for it. <b>Negative:</b> the NLI baseline falsely flags
about a third of faithful claims and misses about half of over-broad holdings; an existence check catches only
fabricated citations; and a general adaptive-and-verified RAG pipeline did not improve judged correctness on 36 MuSiQue
questions while being much slower. <b>Inconclusive:</b> whether v1 is hard for LLM checkers, whether verification
reduces unsupported claims, and any comparison with prior systems. Next steps are LLM-checker baselines on v1,
lawyer-validated labels, High Court data, and verification of the statute table against official texts.</p>

<h2>Appendix A&nbsp;&nbsp;Reproducibility</h2>
<p>All figures and tables are generated by <tt>paper/figures.py</tt> and <tt>paper/build.py</tt> from the result files
named in the captions. The legal benchmark scripts are <tt>backend/scripts/legal_india_bench.py</tt> (v0 and v1) and the
corpus script is <tt>backend/scripts/legal_india_fetch.py</tt>. Seeds are 13 throughout. Models used: {m0['model']} (v0 item generation and the B1 checker), {m1['generator_model']}
(v1 item generation) with {m1['near_paragraph_filter_model']} as the v1 filter and {m1['embedding_model']} for
similarity; <tt>{V0['nli']}</tt> (B2); and, for the MuSiQue experiment, {MQ['models']['chat']} (pipeline),
<tt>{MQ['models']['verifier_nli']}</tt> (verifier), {MQ['models']['judge']} (judge) and
<tt>{MQ['models']['evaluator_nli']}</tt> (evidence evaluator).</p>

@@REFS@@
"""

import re

CSS = """
@page { size: Letter; margin: 0.75in 0.625in 1in 0.625in; }
* { box-sizing: border-box; }
body { font-family: "Times New Roman", "Liberation Serif", Times, serif; font-size: 10pt; line-height: 1.2;
       color: #000; margin: 0; background: #fff; }
.paper { column-count: 2; column-gap: 0.25in; text-align: justify; hyphens: auto; }
.head { column-span: all; text-align: center; margin: 0 0 0.16in 0; }
.head h1 { font-size: 24pt; line-height: 1.12; margin: 0 0 0.12in 0; font-weight: bold; }
.head .authors { font-size: 11pt; }
p { margin: 0 0 0.06in 0; orphans: 2; widows: 2; }
p.abs { font-size: 9pt; font-weight: bold; font-style: italic; text-align: justify; margin-bottom: 0.07in; }
.note { font-size: 8pt; font-style: italic; text-align: left; border-top: 0.5pt solid #000;
        border-bottom: 0.5pt solid #000; padding: 0.03in 0; margin: 0.05in 0 0.1in 0; }
h2 { font-size: 10pt; font-weight: normal; text-transform: uppercase; text-align: center; margin: 0.12in 0 0.05in 0;
     break-after: avoid; letter-spacing: 0.02em; }
h3 { font-size: 10pt; font-weight: normal; font-style: italic; text-align: left; margin: 0.08in 0 0.03in 0;
     break-after: avoid; }
tt { font-family: "Consolas", "Courier New", monospace; font-size: 8.5pt; overflow-wrap: anywhere; }
figure { margin: 0.06in 0; break-inside: avoid; text-align: center; }
figure.wide { column-span: all; margin: 0.08in 0; }
figure.wide img { width: 93%; }
figure img { width: 100%; display: block; margin: 0 auto; }
figcaption { font-size: 8pt; line-height: 1.18; margin-top: 0.04in; text-align: justify; }
.pair { column-span: all; display: grid; grid-template-columns: 1fr 1fr; gap: 0.25in; }
.tbl { margin: 0.08in 0; break-inside: avoid; }
.tbl.wide { column-span: all; }
.tcap { font-size: 8pt; line-height: 1.18; text-align: center; margin-bottom: 0.04in; }
.tcap .tn { font-variant: small-caps; letter-spacing: 0.04em; }
.tcap .tc { text-align: center; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; line-height: 1.15; }
th { border-top: 0.6pt solid #000; border-bottom: 0.4pt solid #000; padding: 0.02in 0.04in; text-align: left;
     font-weight: bold; }
td { padding: 0.012in 0.04in; vertical-align: top; text-align: left; }
tbody tr:last-child td { border-bottom: 0.6pt solid #000; }
table.t2 td, table.t3 td, table.t5 td, table.t4 td { white-space: nowrap; }
table.t6 td:first-child { font-weight: bold; width: 18%; }
.refs { font-size: 8pt; line-height: 1.17; text-align: left; }
.ref { padding-left: 0.3in; text-indent: -0.3in; margin-bottom: 0.03in; }
.v { font-style: italic; }
"""

VERIFY = ""
REFS = {
    1: "A. Verma, &ldquo;Is this citation on point?,&rdquo; arXiv:2608.12571, 2026." + VERIFY,
    2: "C. Qian, Y. Wang, Y. Chen, L. Wu, and A. Stathopoulos, &ldquo;GANDR: Claim auditing for verifiable legal "
       "answer generation,&rdquo; arXiv:2609.10293, 2026.",
    3: "V. Magesh, F. Surani, M. Dahl, M. Suzgun, C. D. Manning, and D. E. Ho, &ldquo;Hallucination-free? Assessing the "
       "reliability of leading AI legal research tools,&rdquo; <i>J. Empirical Legal Studies</i>, 2025; "
       "arXiv:2405.20362." + VERIFY,
    4: "S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. Park, &ldquo;Adaptive-RAG: Learning to adapt "
       "retrieval-augmented large language models through question complexity,&rdquo; in <i>Proc. NAACL</i>, 2024; "
       "arXiv:2403.14403." + VERIFY,
    5: "A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, &ldquo;Self-RAG: Learning to retrieve, generate, and "
       "critique through self-reflection,&rdquo; in <i>Proc. ICLR</i>, 2024; arXiv:2310.11511." + VERIFY,
    6: "N. Pipitone and G. H. Alami, &ldquo;LegalBench-RAG: A benchmark for retrieval-augmented generation in the "
       "legal domain,&rdquo; arXiv:2408.10343, 2024." + VERIFY,
    7: "H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, &ldquo;MuSiQue: Multihop questions via "
       "single-hop question composition,&rdquo; <i>Trans. Assoc. Comput. Linguistics</i>, 2022.",
    8: "&ldquo;Indian Supreme Court Judgments,&rdquo; AWS Open Data registry (Dattam Labs). [Online]. Available: "
       "https://registry.opendata.aws/indian-supreme-court-judgments/ (accessed Nov. 3, 2026).",
    9: "<tt>cross-encoder/nli-roberta-base</tt>, Hugging Face model hub.",
}


def ieee(body: str) -> tuple[str, str]:
    """IEEE numbering: Roman section numbers, lettered subsections, 'Fig.'/'Table' references and citations
    renumbered in order of first appearance. Returns (body, reference_list_html)."""
    for old, new in [
        ("<b>Checkers.</b> <i>B0 (existence only)</i>", "<b>Checkers.</b> Figure 1 summarises the systems evaluated. <i>B0 (existence only)</i>"),
        ("(34 of 111 controls),", "(34 of 111 controls; Figure 5),"),
        ("<p>The NLI baseline's recall on v1 is high", "<p>Table 3 and Figure 4 give per-type recall. The NLI baseline's recall on v1 is high"),
        ("<p>Adaptive depth behaved as designed:", "<p>Tables IV and V and Figs. 7&ndash;9 give the results. Adaptive depth behaved as designed:"),
        ("<b>What our results do and do not say about the base paper.</b>", "<b>What our results do and do not say about the base paper.</b> Table 6 compares the two setups."),
        ("<p>All figures and tables are generated", '<p style="text-align:left">All figures and tables are generated'),
    ]:
        assert old in body, old
        body = body.replace(old, new, 1)
    body = re.sub(r"<h2>(\d)&nbsp;&nbsp;([^<]*)</h2>", lambda m: f"<h2>{ROMAN[int(m.group(1))]}. {m.group(2)}</h2>", body)
    body = re.sub(r"<h2>Appendix A&nbsp;&nbsp;([^<]*)</h2>", lambda m: f"<h2>Appendix: {m.group(1)}</h2>", body)
    body = re.sub(r"<h3>(\d)\.(\d)&nbsp;&nbsp;([^<]*)</h3>",
                  lambda m: f"<h3>{LETTER[int(m.group(2)) - 1]}. {m.group(3)}</h3>", body)
    body = body.replace("Section 6.4", "Section VI-D").replace("Section 3 ", "Section III ")
    body = re.sub(r"Table (\d)\b", lambda m: "Table " + ROMAN[int(m.group(1))], body)
    seen = re.findall(r'<img src="figures/(fig\d_[a-z0-9_]+)\.png"', body)
    fmap = {FIG_NUM[name]: i + 1 for i, name in enumerate(seen)}
    body = re.sub(r"<figcaption>Fig\. (\d)\.", lambda m: f"<figcaption>Fig. {fmap[int(m.group(1))]}.", body)
    body = re.sub(r"Figure (\d)", lambda m: "Fig. " + str(fmap[int(m.group(1))]), body)
    order: list[int] = []
    for m in re.finditer(r"\[(\d+)\]", body):
        n = int(m.group(1))
        if n not in order:
            order.append(n)
    assert set(order) == set(REFS), (order, sorted(REFS))
    new = {old: i + 1 for i, old in enumerate(order)}
    body = re.sub(r"\[(\d+)\]", lambda m: f"[{new[int(m.group(1))]}]", body)
    refs = "".join(f'<div class="ref">[{new[old]}] {REFS[old]}</div>' for old in order)
    return body, f'<h2>References</h2><div class="refs">{refs}</div>'


BODY_FINAL, REFS_HTML = ieee(BODY)
BODY_FINAL = BODY_FINAL.replace("@@REFS@@", REFS_HTML)

HEAD = f"""
<div class="head">
<h1>Beyond Existence Checks: Near-Miss Citation Errors in Indian Supreme Court Judgments and a Local Verification Baseline</h1>
<div class="authors">Nishchal K J, Riya Acharya, Saksham Yadav</div>
</div>
<p class="abs">Abstract&mdash;{ABSTRACT}</p>
<p class="abs">Index Terms&mdash;legal NLP, citation verification, retrieval-augmented generation, Indian law, benchmark,
negative results</p>

"""

DOC = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Beyond Existence Checks</title>
<style>{CSS}</style></head><body><div class="paper">{HEAD}{BODY_FINAL}</div></body></html>"""


def find_browser() -> str | None:
    for path in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                 r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if Path(path).exists():
            return path
    return None


def main() -> None:
    out_html = HERE / "main.html"
    out_html.write_text(DOC, encoding="utf-8")
    print("wrote", out_html)
    browser = find_browser()
    if not browser:
        print("No Chrome or Edge found; open main.html and print to PDF.")
        return
    pdf = HERE / "main.pdf"
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
           out_html.as_uri()]
    subprocess.run(cmd, check=True, timeout=180)
    print("wrote", pdf)


if __name__ == "__main__":
    sys.exit(main())
