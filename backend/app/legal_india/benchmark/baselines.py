"""Baseline citation checkers (B0 existence, B1 LLM-only, B2 NLI-only) and the M1 rule check.

All checkers return one ``Decision`` per item; ``flagged=True`` means "this citation should be flagged".
None of them is the proposed auditor: they are the reference points it has to beat.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.structured import StructuredOutputError, structured_call
from app.core.usage import UsageTracker
from app.legal_india.benchmark.corpus import Corpus, parse_date
from app.legal_india.benchmark.schema import BenchItem
from app.legal_india.statutes import extract_statutes
from app.legal_india.transition import check_cited_section, load_section_map
from app.verification.evidence_matcher import sentence_windows
from app.verification.nli import EntailmentModel, NLIResult


@dataclass
class Decision:
    item_id: str
    flagged: bool
    score: float | None = None  # checker-specific support score, if it has one
    reason: str = ""


def evidence_text(item: BenchItem, corpus: Corpus) -> str | None:
    """The paragraph text the citation points to, or None if the citation cannot be resolved.

    Case citation: the cited paragraph of the cited case. Statute citation: the judgment paragraph in
    which the section is discussed (the statute text itself is not in the corpus).
    """
    if item.cited_authority.kind == "statute":
        case = corpus.cases.get(item.source_case_id)
    else:
        case = corpus.resolve(item.cited_authority.citation)
    if case is None or item.cited_para is None:
        return None
    para = case.paragraphs.get(item.cited_para)
    return para.text if para else None


def check_b0(items: list[BenchItem], corpus: Corpus) -> list[Decision]:
    """B0, existence only: flag a case citation that does not resolve to a corpus case. Nothing else is checked."""
    out = []
    for item in items:
        if item.cited_authority.kind == "case" and corpus.resolve(item.cited_authority.citation) is None:
            out.append(Decision(item.id, True, reason="citation does not resolve"))
        else:
            out.append(Decision(item.id, False, reason="exists or not a case citation"))
    return out


B1_PROMPT = """You check whether a legal claim is supported by a cited paragraph of an Indian court judgment.

Use only the paragraph. Answer "supported" only if the paragraph states or clearly implies the whole claim, including any section numbers, names, numbers and outcomes. Otherwise answer "unsupported".

Return JSON: {"verdict": "supported" | "unsupported", "reason": "<one short sentence>"}"""


class _Verdict(BaseModel):
    verdict: str
    reason: str = ""


async def check_b1(
    items: list[BenchItem],
    corpus: Corpus,
    model: BaseChatModel,
    model_name: str,
    usage: UsageTracker,
    concurrency: int = 8,
) -> list[Decision]:
    """B1, LLM only: the LLM sees the claim and the cited paragraph, nothing else (no NLI, no rules)."""
    sem = asyncio.Semaphore(concurrency)

    async def one(item: BenchItem) -> Decision:
        text = evidence_text(item, corpus)
        if text is None:
            return Decision(item.id, True, reason="no cited text to check against")
        async with sem:
            try:
                v = await structured_call(
                    model,
                    [
                        SystemMessage(content=B1_PROMPT),
                        HumanMessage(content=f"Paragraph:\n{text}\n\nClaim: {item.claim_text}"),
                    ],
                    _Verdict,
                    stage="bench_b1",
                    model_name=model_name,
                    usage=usage,
                    max_tokens=120,
                )
            except StructuredOutputError:
                return Decision(item.id, True, reason="unparseable verdict")
        supported = v.verdict.strip().lower() == "supported"
        return Decision(item.id, not supported, score=1.0 if supported else 0.0, reason=v.reason)

    return list(await asyncio.gather(*(one(i) for i in items)))


def score_b2(items: list[BenchItem], corpus: Corpus, nli: EntailmentModel) -> list[Decision]:
    """B2, NLI only: the best entailment of the claim by the cited paragraph (3-sentence windows and the whole paragraph).

    ``flagged`` is left False here; the runner applies a threshold tuned on the dev split.
    An unresolvable citation gets score 0.0 (nothing can entail the claim).
    """
    pairs: list[tuple[str, str]] = []
    owners: list[int] = []
    for idx, item in enumerate(items):
        text = evidence_text(item, corpus)
        if text is None:
            continue
        for premise in [*sentence_windows(text, 3), text]:
            pairs.append((premise, item.claim_text))
            owners.append(idx)
    results = nli.score(pairs) if pairs else []
    best = [0.0] * len(items)
    for idx, res in zip(owners, results, strict=True):
        best[idx] = max(best[idx], res.entailment)
    return [Decision(item.id, False, score=best[i]) for i, item in enumerate(items)]


def apply_threshold(decisions: list[Decision], threshold: float) -> list[Decision]:
    return [Decision(d.item_id, (d.score or 0.0) < threshold, d.score, d.reason) for d in decisions]


def tune_threshold(decisions: list[Decision], items: list[BenchItem], grid: list[float] | None = None) -> float:
    """Threshold maximising overall flag F1. Call with DEV items only."""
    grid = grid or [round(0.05 * k, 2) for k in range(1, 20)]
    labels = {i.id: i.should_flag for i in items}
    best_t, best_f1 = grid[0], -1.0
    for t in grid:
        tp = fp = fn = 0
        for d in decisions:
            flagged = (d.score or 0.0) < t
            if flagged and labels[d.item_id]:
                tp += 1
            elif flagged:
                fp += 1
            elif labels[d.item_id]:
                fn += 1
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def check_rule_wrong_code(items: list[BenchItem]) -> list[Decision]:
    """M1 transition rule: flag a statute citation whose code is wrong for the offence date.

    The offence date is bounded above by the judgment date; the judgment must predate 1 July 2024 for
    a BNS citation to be flagged, which is how the benchmark constructs wrong_code items. Case items
    are never flagged (this rule is only a wrong-code check, not a general auditor).
    """
    smap = load_section_map()
    out = []
    for item in items:
        if item.cited_authority.kind != "statute":
            out.append(Decision(item.id, False, reason="not a statute citation"))
            continue
        refs = [r for r in extract_statutes(item.claim_text) if r.act in {"IPC", "BNS"}]
        date_max: date = parse_date(item.judgment_date)
        flagged = any(check_cited_section(r, date_max, smap).status == "wrong_code" for r in refs)
        out.append(Decision(item.id, flagged, reason="wrong code for offence date" if flagged else "code ok"))
    return out


B1_CONF_PROMPT = """You check whether a legal claim is supported by a cited paragraph of an Indian court judgment.

Use only the paragraph. Answer "supported" only if the paragraph states or clearly implies the whole claim, including its conditions, section numbers, names, numbers and outcomes. If the claim is broader than the paragraph, changes who prevailed or an outcome, changes a number, or the paragraph only discusses a related point, answer "unsupported".

Return JSON: {"verdict": "supported" | "unsupported", "confidence": <number from 0 to 1: how sure you are of this verdict>, "reason": "<one short sentence>"}"""


class _VerdictConf(BaseModel):
    verdict: str
    confidence: float = 0.5
    reason: str = ""


async def check_b1_conf(
    items: list[BenchItem],
    corpus: Corpus,
    model: BaseChatModel,
    model_name: str,
    usage: UsageTracker,
    concurrency: int = 8,
    stage: str = "bench_b1v1",
) -> list[Decision]:
    """LLM-only checker that also states a confidence (v1). ``score`` is the verbalised probability that the claim is supported."""
    sem = asyncio.Semaphore(concurrency)

    async def one(item: BenchItem) -> Decision:
        text = evidence_text(item, corpus)
        if text is None:
            return Decision(item.id, True, score=0.0, reason="no cited text to check against")
        async with sem:
            try:
                v = await structured_call(
                    model,
                    [
                        SystemMessage(content=B1_CONF_PROMPT),
                        HumanMessage(content=f"Paragraph:\n{text}\n\nClaim: {item.claim_text}"),
                    ],
                    _VerdictConf,
                    stage=stage,
                    model_name=model_name,
                    usage=usage,
                    max_tokens=140,
                )
            except StructuredOutputError:
                return Decision(item.id, True, score=0.0, reason="unparseable verdict")
        supported = v.verdict.strip().lower() == "supported"
        conf = min(max(v.confidence, 0.0), 1.0)
        return Decision(item.id, not supported, score=conf if supported else 1.0 - conf, reason=v.reason)

    return list(await asyncio.gather(*(one(i) for i in items)))


class CachedNLI:
    """Wraps an entailment model with a persistent score cache, so re-runs do not repeat CPU inference."""

    def __init__(self, inner: EntailmentModel, path: Path) -> None:
        self._inner = inner
        self.name = inner.name
        self._path = path
        self._cache: dict[str, list[float]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(premise: str, hypothesis: str) -> str:
        return hashlib.sha1(f"{premise}\x00{hypothesis}".encode()).hexdigest()

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
        keys = [self._key(p, h) for p, h in pairs]
        missing = {k: pair for k, pair in zip(keys, pairs, strict=True) if k not in self._cache}
        self.hits += len(pairs) - len(missing)
        self.misses += len(missing)
        if missing:
            todo = list(missing.items())
            todo.sort(key=lambda kv: len(kv[1][0]))  # similar lengths together keeps padding small
            for start in range(0, len(todo), 64):  # persist progress so an interrupted run can resume
                chunk = todo[start : start + 64]
                for (k, _), r in zip(chunk, self._inner.score([pair for _, pair in chunk]), strict=True):
                    self._cache[k] = [r.entailment, r.neutral, r.contradiction]
                self._path.write_text(json.dumps(self._cache), encoding="utf-8")
        return [NLIResult(*self._cache[k]) for k in keys]
