# Paper: Beyond Existence Checks (working draft, IEEE conference style)

Publication-style draft built entirely from result files in this repository. No number is typed by hand.

| File | What it is |
|---|---|
| `main.pdf` | The paper (US Letter, two columns, IEEE conference style, 8 pages) |
| `main.html` | Source of the PDF (print CSS) |
| `figures/` | 9 figures as 300 dpi PNG and vector PDF |
| `data.py` | Loads every number from `experiments/` |
| `figures.py`, `fig1.py` | Draws the figures from the data |
| `build.py` | Builds tables, text, IEEE numbering and the PDF |
| `_pages/` | Scratch renders used for visual checks (safe to delete) |

## IEEE-style formatting
US Letter; 10 pt Times New Roman; margins 0.75 in top, 1 in bottom, 0.625 in sides; 0.25 in column gap; centred bold
24 pt title; authors line; abstract (bold italic 9 pt, "Abstract—") and "Index Terms—"; centred all-caps Roman-numeral
section headings with italic lettered subsections; figure captions below as "Fig. N."; table captions above as "TABLE
I" (small caps) plus the caption; thin-rule 8 pt tables; references titled REFERENCES, numbered in order of first
citation (done programmatically in `build.py`, function `ieee()`); figures numbered in order of appearance.

## Rebuild

```powershell
cd paper
..\backend\.venv\Scripts\python.exe figures.py   # figures from the result files
..\backend\.venv\Scripts\python.exe build.py     # main.html and main.pdf (needs Chrome or Edge)
```

No LaTeX or pandoc is needed; the PDF is printed from `main.html` by headless Chrome (or Edge).
No OpenAI calls are made. The legal item files (`experiments/legal_india/bench_v*.jsonl`) are read only for labels
and ids and are gitignored (they contain judgment quotes; redistribution licence unconfirmed).

See `../docs/paper_notes.md` for what the paper claims, what it does not, and what must be verified before submission.
