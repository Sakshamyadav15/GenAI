"""Loads every number used in the paper from the repository's result files.

Nothing in the paper text or figures is typed in by hand: tables, figures and in-text numbers are all
computed from these files. Item files (bench_*.jsonl) are read only for labels and ids, never quoted.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGAL = ROOT / "experiments" / "legal_india"
MAIN = ROOT / "experiments" / "results" / "main"

FILES = {
    "v0_results": LEGAL / "results_baselines.json",
    "v1_results": LEGAL / "results_baselines_v1.json",
    "v0_manifest": LEGAL / "bench_v0_manifest.json",
    "v1_manifest": LEGAL / "bench_v1_manifest.json",
    "v0_decisions": LEGAL / "decisions_baselines.json",
    "v1_decisions": LEGAL / "decisions_baselines_v1.json",
    "v0_items": LEGAL / "bench_v0.jsonl",
    "v1_items": LEGAL / "bench_v1.jsonl",
    "fetch": LEGAL / "fetch_summary.json",
    "musique": MAIN / "summary.json",
    "musique_report": MAIN / "report.md",
}


def _json(name: str):
    return json.loads(FILES[name].read_text(encoding="utf-8"))


def _jsonl(name: str):
    return [json.loads(line) for line in FILES[name].read_text(encoding="utf-8").splitlines() if line.strip()]


v0 = _json("v0_results")
v1 = _json("v1_results")
v0_manifest = _json("v0_manifest")
v1_manifest = _json("v1_manifest")
fetch = _json("fetch")
musique = _json("musique")


def risk_coverage(version: str):
    """Accuracy of the NLI baseline on its most confident share of test items.

    Confidence is the distance of the NLI score from the dev-tuned threshold (the definition used in
    results_baselines_v1.json). Returns (coverage list, accuracy list, overall accuracy, AURC).
    """
    items = {r["id"]: r for r in _jsonl(f"{version}_items")}
    decisions = _json(f"{version}_decisions")["B2_nli_only"]
    res = v0 if version == "v0" else v1
    thr = res["b2_threshold_tuned_on_dev"]
    rows = []
    for r in decisions:
        item = items[r["id"]]
        if item["split"] != "test":
            continue
        truth = item["should_flag"] in (True, "True", 1)
        score = r["score"]
        conf = abs(score - thr) if score is not None else 1.0
        rows.append((conf, bool(r["flagged"]) == truth))
    rows.sort(key=lambda x: -x[0])
    n = len(rows)
    cov, acc, wrong = [], [], 0
    for k in range(1, n + 1):
        wrong += 0 if rows[k - 1][1] else 1
        cov.append(k / n)
        acc.append(1 - wrong / k)
    aurc = sum((1 - a) for a in acc) / n  # mean risk over coverage levels
    return cov, acc, acc[-1], aurc, thr


def f(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def ci(m: dict, d: int = 2) -> str:
    """'0.86 [0.83, 0.89]' from {value, ci95}."""
    lo, hi = m["ci95"]
    return f"{m['value']:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def mci(m: dict, d: int = 3) -> str:
    """'0.583 [0.417, 0.750]' from {mean, lo, hi}."""
    return f"{m['mean']:.{d}f} [{m['lo']:.{d}f}, {m['hi']:.{d}f}]"
