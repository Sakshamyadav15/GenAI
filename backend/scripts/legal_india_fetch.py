"""Fetch a small sample of Indian Supreme Court judgments from the AWS Open Data bucket and prepare a
paragraph-level corpus.

    python -m scripts.legal_india_fetch --years 2023 2024 --per-year 100

Source: s3://indian-supreme-court-judgments (public, anonymous read, region ap-south-1), read over plain
HTTPS. Licence per the registry: CC-BY-4.0 (see docs/legal_india_data.md for the caveats).

Be considerate to the data host: a small sample, 2 concurrent downloads, a short pause between requests.
Raw downloads go to experiments/legal_india/raw/ (gitignored); the extracted corpus to
experiments/legal_india/sample_corpus.jsonl (also not committed).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from pypdf import PdfReader

BASE = "https://indian-supreme-court-judgments.s3.amazonaws.com"
ROOT = Path(__file__).resolve().parents[2] / "experiments" / "legal_india"
RAW = ROOT / "raw"
USER_AGENT = "adaptive-rag-research-sample/1.0 (academic; small sample)"
MAX_TOTAL_BYTES = 300 * 1024 * 1024
MIN_CHARS = 800  # below this a PDF is treated as scanned / empty

_NUMBERED = re.compile(r"^\s*(\d{1,3})\s*[.)]\s+(?=\S)")
_PAGE_NOISE = re.compile(r"^\s*(?:\d+\s*)?(?:Digital Supreme Court Reports|\[?\d{4}\]?\s*\d+\s*S\.C\.R\.\s*\d*)\s*$", re.I)


def fetch(url: str, dest: Path, retries: int = 3) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            dest.write_bytes(data)
            time.sleep(0.3)
            return len(data)
        except Exception:  # noqa: BLE001 - retry any transient network error
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return 0


def extract_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def split_paragraphs(text: str) -> tuple[list[dict], str]:
    """Split a judgment into numbered paragraphs. Returns (paragraphs, method).

    Indian judgments number their paragraphs "1.", "2.", ... A line starting with the *next expected*
    number begins a new paragraph; anything else is a continuation. Text before paragraph 1 becomes
    paragraph 0 (the header). Falls back to blank-line paragraphs when numbering is not found.
    """
    lines = [line for line in text.replace("\r", "").split("\n") if not _PAGE_NOISE.match(line)]
    paragraphs: list[dict] = []
    current: list[str] = []
    number = 0
    expected = 1
    for line in lines:
        match = _NUMBERED.match(line)
        if match and int(match.group(1)) == expected:
            if current:
                paragraphs.append({"para_no": number, "text": " ".join(current).strip()})
            number, expected = expected, expected + 1
            current = [_NUMBERED.sub("", line, count=1)]
        else:
            current.append(line.strip())
    if current:
        paragraphs.append({"para_no": number, "text": " ".join(current).strip()})
    paragraphs = [{"para_no": p["para_no"], "text": re.sub(r"\s+", " ", p["text"])} for p in paragraphs if p["text"].strip()]

    if sum(1 for p in paragraphs if p["para_no"] > 0) >= 3:
        return paragraphs, "numbered"

    blocks = [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n\s*\n", text) if len(b.strip()) > 40]
    return [{"para_no": i + 1, "text": b} for i, b in enumerate(blocks)], "blank_line"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--per-year", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    selected: list[dict] = []
    for year in args.years:
        meta_path = RAW / f"meta{year}.parquet"
        fetch(f"{BASE}/metadata/parquet/year={year}/metadata.parquet", meta_path)
        rows = pq.read_table(meta_path).drop(["raw_html"]).to_pylist()
        english = [r for r in rows if "ENG" in (r.get("available_languages") or "") and r.get("path")]
        rng.shuffle(english)
        selected.extend(english[: args.per_year])
        print(f"{year}: {len(rows)} judgments in metadata, {len(english)} with English text, sampling {min(args.per_year, len(english))}")

    total_bytes = 0
    failures: list[tuple[str, str]] = []

    def download(row: dict) -> tuple[dict, Path | None]:
        year = row["year"]
        pdf = RAW / "pdf" / f"year={year}" / f"{row['path']}_EN.pdf"
        try:
            return row, pdf if fetch(f"{BASE}/data/pdf/year={year}/english/{row['path']}_EN.pdf", pdf) else None
        except Exception as exc:  # noqa: BLE001
            failures.append((row["path"], f"download: {type(exc).__name__}: {exc}"))
            return row, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        downloaded = list(pool.map(download, selected))
    total_bytes = sum(p.stat().st_size for _, p in downloaded if p)
    print(f"downloaded {sum(1 for _, p in downloaded if p)} PDFs, {total_bytes / 1e6:.1f} MB")
    if total_bytes > MAX_TOTAL_BYTES:
        print("WARNING: above the 300 MB budget; reduce --per-year", file=sys.stderr)

    out = ROOT / "sample_corpus.jsonl"
    stats = {"written": 0, "no_text": 0, "extract_error": 0, "numbered": 0, "blank_line": 0}
    para_counts: list[int] = []
    with out.open("w", encoding="utf-8") as fh:
        for row, pdf in downloaded:
            if pdf is None:
                continue
            try:
                text = extract_text(pdf)
            except Exception as exc:  # noqa: BLE001 - corrupt or encrypted PDF
                stats["extract_error"] += 1
                failures.append((row["path"], f"extract: {type(exc).__name__}: {exc}"))
                continue
            if len(text.strip()) < MIN_CHARS:
                stats["no_text"] += 1
                failures.append((row["path"], f"no extractable text ({len(text.strip())} chars; likely scanned)"))
                continue
            paragraphs, method = split_paragraphs(text)
            stats[method] += 1
            stats["written"] += 1
            para_counts.append(len(paragraphs))
            fh.write(json.dumps({
                "case_id": row["case_id"], "title": row["title"], "court": row["court"],
                "date": row["decision_date"], "citation": row["citation"], "neutral_citation": row["nc_display"],
                "judges": row["judge"], "petitioner": row["petitioner"], "respondent": row["respondent"],
                "disposal_nature": row["disposal_nature"], "cnr": row["cnr"], "source_path": row["path"],
                "year": row["year"], "split_method": method, "paragraphs": paragraphs,
            }, ensure_ascii=False) + "\n")

    summary = {
        "sampled": len(selected), **stats, "failures": len(failures), "download_mb": round(total_bytes / 1e6, 1),
        "paragraphs_mean": round(statistics.mean(para_counts), 1) if para_counts else 0,
        "paragraphs_median": statistics.median(para_counts) if para_counts else 0,
    }
    (ROOT / "fetch_summary.json").write_text(json.dumps({**summary, "failure_list": failures[:50]}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
