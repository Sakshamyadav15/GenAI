# Indian Acts research mode (fast, free prototype)

A second instance of the application for researching Indian statutes. Answers cite the Act and provision
("Constitution of India, Article 21"), each claim is checked against the cited text, and a question costs **zero
money** (local embeddings and local NLI, Gemini free tier for the one LLM call). Decision support only, **not legal
advice**: check every provision against the official text.

## What it is

| Part | How |
|---|---|
| Corpus | Acts split into **sections** (articles for the Constitution), one chunk per section, long sections split at clause boundaries; every chunk starts with "<Act>, Section N. <Title>" and carries Act, section, title and chapter as metadata |
| Embeddings | **Local** `BAAI/bge-small-en-v1.5` (no API, no cost), warmed up at start |
| LLM | **Gemini** (`gemini-2.5-flash` in the start script, thinking off) when `GEMINI_API_KEY` is set (`LLM_PROVIDER=auto`); OpenAI otherwise |
| Verification | One claim per sentence (no LLM call), checked against its cited provision by the **fast local NLI model** `cross-encoder/nli-roberta-base`; no LLM judge, no retries |
| Calls per question | **1 LLM call** (the answer) |
| Pipeline | `proposed` with `adaptive=false`: fixed top-4 retrieval, short answers (250 tokens) |
| Cost ceiling | `budget.max_cost_usd = 0.001` is enforced for priced models; Gemini free tier and local models count as USD 0 |
| Instance | Own data directory (`backend/data_legal`), own port 8001, own samples (`backend/seed/legal_demo_cache`) |

## Run it (Windows PowerShell)

Backend on port 8001:

```powershell
cd "D:\Adaptive rag - skill lab\backend"
.\scripts\start_legal.ps1
```

Frontend on port 5174 (second terminal):

```powershell
cd "D:\Adaptive rag - skill lab\frontend"
npm run dev:legal            # open http://localhost:5174
```

The script sets every instance-specific setting as a process environment variable (nothing is written to the shared
`.env`) and reads only the API keys from it. The general app keeps running on 8000 and 5173.

First-time setup (indexing needs no API calls):

```powershell
cd "D:\Adaptive rag - skill lab\backend"
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt -r requirements-legal.txt
python -m scripts.acts_fetch          # downloads the Acts that have a verified URL
python -m scripts.acts_index          # parses, embeds locally and indexes (add --reindex to rebuild)
```

## Acts (8-Act prototype scope)

The catalogue (`backend/seed/acts_catalog.json`) lists: Constitution of India, Bharatiya Nyaya Sanhita 2023,
Bharatiya Nagarik Suraksha Sanhita 2023, Bharatiya Sakshya Adhiniyam 2023, Indian Penal Code 1860, Code of Criminal
Procedure 1973, Indian Evidence Act 1872, Consumer Protection Act 2019, plus the RTI Act 2005 as a bonus.

| Act | Status |
|---|---|
| Constitution of India (as on 1 May 2024) | **Indexed**: 402 pages, 415 entries, 1,577 chunks (Legislative Department PDF) |
| Right to Information Act, 2005 (bonus) | **Indexed**: 22 pages, 31 of 31 sections, 126 chunks (Ministry of Heavy Industries copy, updated 2021) |
| BNS, BNSS, BSA, IPC, CrPC, Evidence Act, Consumer Protection Act | **Not obtained**: no official URL could be verified |

What was tried, politely and without bypassing anything: India Code (indiacode.nic.in) timed out on every request
(504 and read timeouts, retried once more later), file names I tried on legislative.gov.in and mha.gov.in returned
404, and one official site had an expired certificate. **To add the missing Acts**, download each official PDF
(India Code, legislative.gov.in or the ministry), save it as `backend/data_legal/raw/<slug>.pdf` (`python -m
scripts.acts_fetch` prints the file names), and run `python -m scripts.acts_index`. Then record their samples (see
below).

`acts_fetch.py` reads robots.txt, identifies itself with a User-Agent, waits 2 seconds between requests, makes one
attempt per file and uses no anti-bot tooling.

## Parse quality (measured)

| Act | Sections found | Missing from the sequence |
|---|---|---|
| Right to Information Act, 2005 | 31 of 31 | none |
| Constitution of India | 415 entries (Articles 1 to 395 incl. lettered) | 33 numbers (some genuinely repealed or omitted; others mis-parsed headings) |

The parser is heuristic; PDF extraction garbles some characters (for example "(1)" can come out as "(/)"), bilingual
PDFs lose their Hindi, and amendments or repeals are not tracked.

## Speed

Measured with a fake LLM and the real local parts (`python -m scripts.legal_latency`, 6 questions, no API call):

| Stage | Mean |
|---|---|
| Retrieval (query embedding + search) | 24 ms |
| Evidence embedding | 280 ms |
| NLI claim check | 129 ms |
| **Pipeline total** | **440 ms mean, 799 ms max** |

A real question adds the Gemini response time. Measured live on the legal instance: 3.3 s and 10.7 s for two
questions (the 10.7 s one included the one-time model load, which is now done at startup), with Gemini answering in
about 1 to 4 s. **Saved samples replay in 12 to 22 ms.**

## Cost

Per question: question embedding USD 0 (local), NLI USD 0 (local), one Gemini call USD 0 on the free tier. If the
instance is switched to OpenAI (`LLM_PROVIDER=openai`), `python -m scripts.acts_cost_estimate` gives the arithmetic:
generation 837 in x 0.15/1e6 + 250 out x 0.60/1e6 = USD 0.000276; with LLM claim extraction (not used in this mode)
USD 0.000269 more; worst case USD 0.000544, under the USD 0.001 ceiling. Prices can change; check current pricing.

## Gemini free-tier limits (the main practical constraint)

Gemini free tiers are limited **per model**: the error messages we hit were `limit: 20` requests per **day** on
`gemini-2.5-flash-lite` and `limit: 5` requests per **minute** on `gemini-2.5-flash`. Consequences:

- Because each live question uses one request, expect about 5 questions a minute and a few dozen a day per model.
- A 429 returns HTTP 429 with a clear message (not a crash); the client retries with backoff, and the sample recorder
  waits 50 s and resumes.
- If a model's daily quota is exhausted, set `GEMINI_MODEL` in `scripts/start_legal.ps1` to another Flash model
  (each has its own quota). Saved samples are recorded for one model name and replay only while that model is
  configured; change the model and re-record.
- Check current limits at https://ai.google.dev/gemini-api/docs/rate-limits.

## Budget preset

`PipelineConfig` accepts `{"preset": "budget"}` in a request's `overrides`; `LEGAL_BUDGET_MODE=true` makes it the
instance default. It sets `proposed.adaptive=false`, `baseline.top_k=4`, `generation.max_answer_tokens=250`,
`verification.use_llm_judge=false`, `verification.max_retries=0`, `verification.max_claims=6`,
`verification.nli_model=cross-encoder/nli-roberta-base`, `verification.llm_claims=false` and
`budget.max_cost_usd=0.001`. The hard ceiling skips optional steps (claim extraction, retries) if the estimated cost
would cross it and reports that in the response.

## Sample questions and saved samples

`backend/seed/legal_demo_questions.json` has 8 questions over the 8 Acts. **They were written before any run and
not filtered by the results.** Their answers are real recorded runs replayed through the same demo cache as the
general app (tagged "Saved sample"); any other question runs live.

| Question | Recorded? |
|---|---|
| Article 21 | yes (answered from the Constitution) |
| Article 19 freedoms | yes |
| Articles 32 and 226 (writ jurisdiction) | yes, but the recorded run **failed to find them**: fixed top-4 retrieval missed those articles and the verified answer says there is not enough verified evidence |
| Supreme Court's view on the right to privacy (case law) | yes (correctly not answerable from statute text) |
| BNS murder punishment, BNSS arrest without warrant, BSA electronic records, Consumer Protection remedies | **no**: those Acts are not indexed yet, so a recording would only show "not in the provided texts" |

Re-record after adding the Acts (it skips entries that already exist; an entry is replayed only while the indexed
documents, models and configuration are unchanged, so adding Acts invalidates the old ones):

```powershell
cd "D:\Adaptive rag - skill lab\backend"
.\.venv\Scripts\python.exe -m scripts.build_demo_cache --base http://127.0.0.1:8001 `
    --questions seed\legal_demo_questions.json --out seed\legal_demo_cache --corpus indian_acts --workers 1
```

Cost: 0 (Gemini free tier), about 3 requests per question, so 24 requests for the 8 questions, paced by the
recorder's 429 retry.

## Limitations

- Statute text only, no case law. Only 2 of the 8 target Acts (plus the bonus RTI Act) are indexed.
- Fixed top-4 retrieval with a small local model can miss the right provision (seen on the writ-jurisdiction and
  RTI time-limit questions); there is no query rewriting.
- One claim per sentence and a fast NLI model are less precise than LLM claim splitting with the larger NLI model.
- Amendments, repeals and in-force status are not tracked; parsing and PDF text are imperfect.
- Not validated by lawyers; not legal advice.
- Automated tests were **not** run for the Gemini, one-call-mode and start-script changes, by instruction.
