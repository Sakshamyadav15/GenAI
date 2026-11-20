"""Detection metrics with bootstrap confidence intervals. Evaluate on the TEST split only."""

from __future__ import annotations

import random
from collections.abc import Sequence

from app.legal_india.benchmark.baselines import Decision
from app.legal_india.benchmark.schema import FLAGGED_TYPES, BenchItem


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _counts(rows: Sequence[tuple[bool, bool]]) -> tuple[int, int, int, int]:
    """rows: (should_flag, flagged) -> tp, fp, fn, tn"""
    tp = sum(1 for s, f in rows if s and f)
    fp = sum(1 for s, f in rows if not s and f)
    fn = sum(1 for s, f in rows if s and not f)
    tn = sum(1 for s, f in rows if not s and not f)
    return tp, fp, fn, tn


def _stat(rows: Sequence[tuple[bool, bool]], name: str) -> float:
    tp, fp, fn, tn = _counts(rows)
    p, r, f1 = prf(tp, fp, fn)
    if name == "precision":
        return p
    if name == "recall":
        return r
    if name == "f1":
        return f1
    if name == "false_flag_rate":  # controls flagged / controls
        return fp / (fp + tn) if fp + tn else 0.0
    raise ValueError(name)


def bootstrap_ci(rows: Sequence[tuple[bool, bool]], name: str, n_boot: int = 2000, seed: int = 13) -> tuple[float, float]:
    if not rows:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(rows)
    values = sorted(_stat([rows[rng.randrange(n)] for _ in range(n)], name) for _ in range(n_boot))
    return values[int(0.025 * n_boot)], values[int(0.975 * n_boot) - 1]


def _summary(rows: Sequence[tuple[bool, bool]], names: Sequence[str], n_boot: int, seed: int) -> dict:
    out: dict = {"n": len(rows)}
    for name in names:
        lo, hi = bootstrap_ci(rows, name, n_boot, seed)
        out[name] = {"value": round(_stat(rows, name), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def evaluate(
    items: Sequence[BenchItem],
    decisions: Sequence[Decision],
    *,
    split: str = "test",
    n_boot: int = 2000,
    seed: int = 13,
) -> dict:
    """Overall and per-perturbation detection.

    Per type t: the evaluation set is the items of type t plus the controls (type "none") of the same
    citation kind (statute controls for wrong_code, case controls for the rest), so
    precision = flagged t-items / (flagged t-items + flagged controls); recall = share of t-items flagged.
    """
    by_id = {d.item_id: d for d in decisions}
    chosen = [i for i in items if i.split == split and i.id in by_id]
    rows = [(i.should_flag, by_id[i.id].flagged) for i in chosen]
    result: dict = {
        "split": split,
        "overall": _summary(rows, ("precision", "recall", "f1", "false_flag_rate"), n_boot, seed),
        "per_type": {},
    }
    def controls_of(kind: str) -> list[tuple[bool, bool]]:
        return [
            (i.should_flag, by_id[i.id].flagged)
            for i in chosen
            if i.perturbation_type == "none" and i.cited_authority.kind == kind
        ]

    for t in FLAGGED_TYPES:
        typed = [(i.should_flag, by_id[i.id].flagged) for i in chosen if i.perturbation_type == t]
        if typed:
            kind = "statute" if t == "wrong_code" else "case"
            controls = controls_of(kind)
            result["per_type"][t] = _summary([*typed, *controls], ("precision", "recall", "f1"), n_boot, seed)
            result["per_type"][t]["n_type"] = len(typed)
            result["per_type"][t]["n_controls"] = len(controls)
    return result


def risk_coverage(
    items: Sequence[BenchItem],
    decisions: Sequence[Decision],
    confidence: dict[str, float],
    *,
    split: str = "test",
    levels: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
) -> dict:
    """Selective prediction: answer only the most confident share of items and report accuracy on those.

    ``confidence[item_id]`` is the checker's confidence in its own decision. Accuracy counts a decision as
    right when flagged == should_flag. A well-calibrated checker is more accurate at low coverage.
    ``aurc`` is the area under the risk-coverage curve (mean error over all coverage prefixes; lower is better).
    """
    by_id = {d.item_id: d for d in decisions}
    rows = sorted(
        ((confidence[i.id], by_id[i.id].flagged == i.should_flag) for i in items if i.split == split and i.id in by_id and i.id in confidence),
        key=lambda r: -r[0],
    )
    n = len(rows)
    if not n:
        return {"n": 0, "coverage": [], "accuracy": [], "aurc": None}
    cumulative_error = 0
    errors_at = []
    for k, (_, correct) in enumerate(rows, start=1):
        cumulative_error += 0 if correct else 1
        errors_at.append(cumulative_error / k)
    accuracy = []
    for level in levels:
        k = max(1, round(level * n))
        accuracy.append(round(1 - errors_at[k - 1], 4))
    return {"n": n, "coverage": list(levels), "accuracy": accuracy, "aurc": round(sum(errors_at) / n, 4)}
