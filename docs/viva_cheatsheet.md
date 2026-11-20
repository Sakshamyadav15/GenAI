# Viva cheat sheet (one page)

All numbers below were measured in this project (see `experiments/results/main/`). Evaluation is automatic
only: no human-labelled answers, 36 test questions. Say this first; it protects you.

## 30-second summary
We compare normal RAG (always fetch 5 chunks) with a proposed pipeline that (1) sets how many chunks to
fetch from the question's complexity and (2) checks every claim of the answer against its cited source,
retrying if something is unsupported. Base idea: Adaptive-RAG (Jeong et al., NAACL 2024, arXiv 2403.14403).
Our additions: no trained classifier, plus claim-level verification with retries and explicit uncertainty.

## What worked (measured)
- Adaptive depth behaves as designed: about 2 chunks for simple, 5 for moderate, 8 for complex questions.
- In the saved demo examples, simple questions used 1 chunk instead of 5 with the same answer.
- In two demo examples the same claim checker found unsupported claims in the baseline answer; the proposed
  system removed or hedged them.

## What did not work (say it before they ask)
- Answer correctness on 36 MuSiQue questions: baseline 0.583, proposed 0.444 (judge-scored, wide intervals,
  difference not clearly significant: McNemar p = 0.0625).
- Verification made answers much slower (about 6 to 14 times).
- Retries (0, 1, 2) showed no resolved effect.
- Unsupported-claim rate: 0.714 baseline vs 0.652 full system, interval includes zero.

## Likely questions
1. **Is your system better?** Not on accuracy in our small test. Adaptive retrieval and verification work as
   designed, and a larger evaluation with human labels would be needed to show an accuracy gain.
2. **What is new compared with Adaptive-RAG?** No training needed; a continuous complexity score; claim-level
   verification with a bounded retry loop and explicit "not verified" output.
3. **Why not trust cosine similarity?** Similarity only picks candidate evidence; an NLI model plus a
   quote-checked LLM judge decides support.
4. **How do you avoid grading yourself?** Evaluation uses different models from the system's verifier.
5. **Why were the demo questions chosen?** Hand-picked where recorded runs show the proposed system ahead;
   the page says so, and the full benchmark is mixed.
6. **Are saved demo answers real?** Yes: real recorded runs, tagged "Saved sample", with original latency.
7. **Limitations?** Small sample, automatic labels, CPU-bound NLI speed, LLM-based components can be
   inconsistent (the complexity classifier varied between runs on one question).
8. **Legal extension?** Only started: Indian citation parser, statute-transition checker (IPC to BNS; mapping
   table unverified), and a first benchmark. A plain LLM already solved the first, easy benchmark (F1 0.99),
   so a harder one is needed before claiming a legal verifier helps. Nothing legal is validated by lawyers.

## Do not say
- "It is more accurate." / "It beats the base paper." Neither is supported by our data.
- That the demo samples are a random or representative set.
