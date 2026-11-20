# Data card: Indian Supreme Court judgments sample

Written 2026-11-03 for milestone M1 of [legal_india_plan.md](legal_india_plan.md).

## Source
- **Dataset:** "Indian Supreme Court Judgments", AWS Open Data registry,
  <https://registry.opendata.aws/indian-supreme-court-judgments/>. Managed by Dattam Labs (contact@dattam.in).
- **Provenance:** judgments downloaded from the eCourts website (scr.sci.gov.in); code at
  <https://github.com/vanga/indian-supreme-court-judgments>.
- **Bucket:** `s3://indian-supreme-court-judgments`, region `ap-south-1`, anonymous read. Also reachable over
  HTTPS at `https://indian-supreme-court-judgments.s3.amazonaws.com/` (this is what our script uses; no AWS
  account or boto3 needed).
- **Coverage:** 1950 to 2025, about 35,000 English judgments (about 52 GB), regional-language versions for some.
  Updated roughly every two months.
- **Layout (confirmed by listing the bucket):**
  - `metadata/parquet/year=YYYY/metadata.parquet` (structured metadata)
  - `metadata/json/year=YYYY/`, `metadata/tar/year=YYYY/`
  - `data/pdf/year=YYYY/english/{path}_EN.pdf` (individual judgments; `{path}` is the `path` metadata column)
  - `data/tar/year=YYYY/{english,regional}/` (bulk archives)
- **Metadata fields:** title, petitioner, respondent, description, judge, author_judge, citation, case_id, cnr,
  decision_date, disposal_nature, court, available_languages, raw_html, path, nc_display, scraped_at, year.
  Example: `citation` = "[2024] 10 S.C.R. 108", `case_id` = "2024 INSC 735", `nc_display` = "2024INSC735".

## Licence and attribution
- **Registry states:** licence identifier **CC-BY-4.0**. No full licence text appears on the registry page.
- **Attribution requested by the registry:** "Indian Supreme Court Judgments was accessed on `DATE` from
  https://registry.opendata.aws/indian-supreme-court-judgments." This sample was accessed on **2026-11-03**.
- **Ambiguities (not resolved here):**
  1. The registry does not say whether CC-BY-4.0 covers the judgment texts or only the compiled dataset.
     The repository's `dataset.md` states no licence, no copyright statement and no terms (and is marked outdated).
  2. The texts originate from eCourts. Whether the underlying judgments carry any restriction is a legal question.
     My understanding, which should be checked by someone qualified, is that Indian copyright law permits
     reproduction of court judgments in general, but this is not verified.
  3. The README asks users to be considerate to the eCourts host and not scrape with high concurrency; the
     script uses 2 concurrent downloads and a pause between requests.
- **What is committed to git:** this card, the fetch script, and `experiments/legal_india/example_cases.jsonl`
  (3 cases, first 8 paragraphs each, with attribution). Raw PDFs, full extracted text and the sample corpus
  are gitignored. Before publishing any derived corpus, confirm the licence position above.

## Sample we built
- **Script:** `backend/scripts/legal_india_fetch.py` (`python -m scripts.legal_india_fetch --years 2023 2024 --per-year 100`;
  needs `pyarrow`, see `backend/requirements-legal.txt`). Random sample with seed 13 of judgments that have an English version.
- **Years:** 2023 (856 judgments in metadata) and 2024 (782), 100 sampled from each.
- **Result:** 200 sampled, **199 extracted**, 65.6 MB downloaded.
- **Output:** `experiments/legal_india/sample_corpus.jsonl` with fields case_id, title, court, date, citation,
  neutral_citation, judges, petitioner, respondent, disposal_nature, cnr, source_path, year, split_method and
  `paragraphs: [{para_no, text}]`. Summary in `experiments/legal_india/fetch_summary.json`.

## Extraction quality (measured)
- **Failures:** 1 of 200: `2023_3_30_compressed` returned HTTP 404 (the metadata `path` does not match a PDF key,
  probably a naming oddity for that file). No PDF failed to parse, and none was scanned or empty (0 below the 800-character threshold).
- **Paragraph splitting:** 167 cases (84%) were split on the court's own paragraph numbers ("1.", "2.", ...);
  32 (16%) fell back to blank-line blocks because fewer than 3 numbered paragraphs were found. Paragraph 0 holds the header text.
- **Per case:** mean 27.2 paragraphs, median 21. For the 167 numbered cases the median last paragraph number is 23 (range 3 to 220);
  11 numbered cases have 5 or fewer paragraphs, so short orders are included.
- **Paragraph length:** median about 760 characters, 95th percentile about 5,600 (a few very long paragraphs, usually where numbering was lost).
- **Metadata:** `citation` (S.C.R.) and neutral citation (INSC) present for all 199 cases.

## Known issues
- Numbering heuristic accepts a line only if it carries the next expected number; a skipped or mis-OCR'd
  number merges later paragraphs into one long block. It was checked on aggregate statistics only, not by hand on each case.
- Running headers and footers inside the PDFs (e.g. report series text) are only partly removed.
- Sub-paragraphs ("(a)", "(i)") are not split out; they stay inside their numbered paragraph.
- Only the English texts are used; regional versions are not handled.
- Only Supreme Court judgments, 2 years: this is a smoke-test sample, not a research corpus. No High Court data was fetched.
- Whether a case is later overruled or distinguished is not in the metadata.
