"""Benchmark v1: near-miss corruptions, a different generator model, harder controls.

v0 was too easy (an LLM-only checker reached F1 0.99), because its corruptions shared almost no
vocabulary with the claim. v1 keeps the matched design (one base proposition from one paragraph of one
judgment -> several items) and adds corruptions that look right on the surface:

  near_paragraph            right case, the most similar OTHER paragraph of the same judgment that does
                            not support the claim (picked by embedding similarity, then checked)
  overbroad_holding         the claim states the holding more broadly than the paragraph (a condition or
                            qualifier that the paragraph states is dropped)
  swapped_party_or_outcome  one party role or outcome is swapped (allowed/dismissed, convicted/acquitted)
  altered_quote             negation or a changed number only (outcome/actor changes moved to the swap type)
  fabricated_case           unchanged from v0, kept for continuity
  wrong_paragraph           unchanged from v0 (unrelated paragraph), kept for continuity
  wrong_code                small statute subset (carries the IPC/BNS surface cue; used to run the M1 rule)

and two controls per base: the verbatim proposition and a faithful PARAPHRASE of it (tests whether a
checker falsely flags correct claims that are worded differently from the paragraph).

Not built: wrong_court_binding. Every judgment in the corpus is a Supreme Court judgment, which binds all
courts under Art. 141, so no High Court / bench-strength mismatch can be derived from the corpus metadata.

Labels are automatic by construction and have not been validated by lawyers. Each LLM-made perturbation
passes mechanical checks (the dropped qualifier really occurs in the paragraph and not in the new claim,
the swapped element really changed, numbers survive a paraphrase, ...). The near_paragraph candidate is
additionally checked by a second, smaller model (``filter_model``).
"""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass, field

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.structured import StructuredOutputError, structured_call
from app.core.usage import UsageTracker
from app.legal_india.benchmark.corpus import Case, Corpus, Paragraph, quote_in_text, squash, substantive_paragraphs
from app.legal_india.benchmark.generate import (
    _Base,
    _citation_label,
    _proposition,
    assign_splits,
    end_with_period,
    fabricated_citation,
    make_case_bases,
    make_statute_bases,
    overlap,
    _pick_unrelated_paragraph,
)
from app.legal_india.benchmark.schema import BenchItem, CitedAuthority
from app.legal_india.transition import SectionMap, load_section_map

OVERBROAD_PROMPT = """You create a deliberately WRONG version of a legal statement, to test a citation checker.

You are given a statement and the quote it rests on. First look in the QUOTE for a real legal condition or limitation: words such as "only", "unless", "except", "provided that", "subject to", "if", "where", "within", "without", "not exceeding", "no later than", "at least", "before", "after", "until", "in the absence of", "solely", "merely", "cannot", "must". It has to be a condition that changes WHEN, WHETHER or HOW MUCH the rule applies, not a stylistic phrase.

If you find one that the statement keeps, rewrite the statement so it claims a BROADER rule or wider scope by leaving that condition out (the result should be a statement the quote does NOT fully support). Keep it fluent, about the same length, on the same topic, and add no new facts.

"dropped" = the words of that condition, copied EXACTLY from the source quote (3 to 15 words, including the qualifier word). The rewritten statement must NOT contain those words.

If the quote contains no such condition, or the statement does not keep it, return empty strings.

Return JSON: {"overbroad": "...", "dropped": "..."}"""

SWAP_PROMPT = """You create a deliberately WRONG version of a legal statement, to test a citation checker.

Make exactly ONE swap in the statement: reverse the outcome (allowed/dismissed, upheld/set aside, convicted/acquitted, granted/refused) or swap the roles of two parties (who appealed, who prevailed, who was held liable). Keep everything else identical, fluent, and plausible.

Return JSON: {"swapped": "<the full new statement>", "original_element": "<the exact words in the ORIGINAL statement that you changed>", "new_element": "<the exact words that replace them in the new statement>"}"""

ALTER_PROMPT_V1 = """You create a deliberately WRONG version of a legal statement, to test a citation checker.

Change the meaning of the statement in ONE of these ways only: negate it, or change a number, date or amount to a different value. Keep it fluent, plausible and about the same length. Do not change any citation, section number or case name.

Return JSON: {"altered": "...", "change": "negation|number"}"""

PARAPHRASE_PROMPT = """Rewrite a legal statement in different words while keeping its meaning EXACTLY the same. Do not add, drop or generalise anything. Keep every name, number, date, section and amount unchanged. Change the sentence structure and vocabulary where you can.

Return JSON: {"paraphrase": "..."}"""

SUPPORT_FILTER_PROMPT = """You check how far a paragraph of an Indian court judgment states a given claim.

Answer:
- "full": the paragraph itself states or clearly implies the whole claim, including its conditions, names, numbers and outcome;
- "partial": the paragraph states part of the claim, or states it in a weaker, narrower or more qualified form;
- "no": the paragraph only discusses a related topic, or does not mention the claim.

Return JSON: {"supports": "full" | "partial" | "no"}"""


CONDITIONAL_PROPOSITION_PROMPT = """You extract one checkable statement from a paragraph of an Indian court judgment.

Write ONE self-contained sentence stating a legal rule, holding or direction that the paragraph states TOGETHER WITH a condition or limitation that applies to it (for example "only if", "unless", "except", "provided that", "subject to", "within N days", "not exceeding", "in the absence of", "cannot", "shall not"). The condition must be part of the sentence. Name the parties, statute or issue instead of saying "the appellant" or "the said Act" alone. Do not add anything the paragraph does not say. Also copy one supporting quote of 12 to 40 words EXACTLY as it appears in the paragraph (keep its spelling, even if words look broken by the text extraction); the quote must contain the condition.

If the paragraph has no rule with a condition, return an empty proposition.

Return JSON: {"proposition": "...", "supporting_quote": "..."}"""

_STRONG_CONDITION = re.compile(
    r"\b(only if|only when|only in|unless|except|provided that|subject to|within \d+|within a period|not exceeding|"
    r"in the absence of|shall not|cannot be|no later than|at least|not less than|not more than|save as|save in)\b",
    re.IGNORECASE,
)


class _Overbroad(BaseModel):
    overbroad: str = ""
    dropped: str = ""


class _Swap(BaseModel):
    swapped: str = ""
    original_element: str = ""
    new_element: str = ""


class _AlterV1(BaseModel):
    altered: str = ""
    change: str = ""


class _Paraphrase(BaseModel):
    paraphrase: str = ""


class _Support(BaseModel):
    supports: str = ""


@dataclass
class GenStatsV1:
    attempted_bases: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    errors: int = 0
    notes: list[str] = field(default_factory=list)
    base_stats: dict = field(default_factory=dict)

    def reject(self, kind: str) -> None:
        self.rejected[kind] = self.rejected.get(kind, 0) + 1


_NUMBER = re.compile(r"\d[\d,\.]*")


def numbers_in(text: str) -> set[str]:
    return {n.strip(",.") for n in _NUMBER.findall(text)}


def _in(needle: str, haystack: str) -> bool:
    return bool(needle.strip()) and needle.strip().lower() in haystack.lower()


_CONDITION_WORDS = re.compile(
    r"\b(only|unless|except|provided|subject to|if|where|whether|absence|absent|within|not exceeding|no later|"
    r"limited|at least|at most|minimum|maximum|before|after|until|prior to|save|other than|without|solely|"
    r"merely|exclusively|in case|when|on condition|must|shall not|cannot|not less|not more|upto|up to)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\bnot\b|n't|\bno\b", re.IGNORECASE)


def check_overbroad(res: _Overbroad | None, base: _Base) -> bool:
    """The dropped words are a real condition (contain a qualifier word), occur in the paragraph and not in the new
    claim, and the claim stays close to the original. The caller still asks a filter model whether the source
    paragraph fully supports the broadened claim."""
    if res is None or not res.overbroad.strip() or len(squash(res.dropped)) < 8:
        return False
    if not _CONDITION_WORDS.search(res.dropped):
        return False
    if squash(res.dropped) not in squash(base.para.text):
        return False
    if squash(res.dropped) in squash(res.overbroad):
        return False
    new = res.overbroad.strip()
    return new.lower() != base.proposition.lower() and overlap(new, base.proposition) >= 0.4


def check_swap(res: _Swap | None, base: _Base) -> bool:
    """One element of the original was replaced by a different one, and nothing else drifted far."""
    if res is None or not res.swapped.strip():
        return False
    orig, new, text = res.original_element, res.new_element, res.swapped
    if not (_in(orig, base.proposition) and _in(new, text)):
        return False
    if _in(orig, text) and text.lower().count(orig.strip().lower()) >= base.proposition.lower().count(orig.strip().lower()):
        return False  # the original element is still there, nothing was swapped
    if orig.strip().lower() == new.strip().lower():
        return False
    if _NEGATION.search(new) and not _NEGATION.search(orig):
        return False  # a bare negation belongs to altered_quote, not to a party/outcome swap
    return overlap(text, base.proposition) >= 0.45


def check_alter(res: _AlterV1 | None, base: _Base) -> bool:
    if res is None:
        return False
    altered = res.altered.strip()
    return bool(altered) and altered.lower() != base.proposition.lower() and overlap(altered, base.proposition) >= 0.35


def check_paraphrase(res: _Paraphrase | None, base: _Base) -> bool:
    """Different wording, same numbers (a paraphrase that loses or changes a number is not faithful)."""
    if res is None or not res.paraphrase.strip():
        return False
    text = res.paraphrase.strip()
    if text.lower() == base.proposition.lower():
        return False
    if not numbers_in(base.proposition) <= numbers_in(text):
        return False
    return 0.25 <= overlap(text, base.proposition) <= 0.9


async def _ask(
    model: BaseChatModel, name: str, system: str, user: str, schema: type[BaseModel], usage: UsageTracker,
    sem: asyncio.Semaphore, stage: str, max_tokens: int = 350,
):  # noqa: ANN202
    async with sem:
        for attempt in range(4):
            try:
                return await structured_call(
                    model, [SystemMessage(content=system), HumanMessage(content=user)], schema,
                    stage=stage, model_name=name, usage=usage, max_tokens=max_tokens,
                )
            except StructuredOutputError:
                return None
            except Exception:  # noqa: BLE001 - transient network/API errors: back off, then give up on this one item
                if attempt == 3:
                    return None
                await asyncio.sleep(2.0 * (attempt + 1))
        return None


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a @ b.T


async def make_conditional_bases(
    corpus: Corpus,
    model: BaseChatModel,
    model_name: str,
    rng: random.Random,
    n: int,
    usage: UsageTracker,
    sem: asyncio.Semaphore,
    stats: GenStatsV1,
) -> list[_Base]:
    """Base propositions that keep a real condition, drawn from paragraphs that contain one (source for overbroad_holding)."""
    picks: list[tuple[Case, Paragraph]] = []
    for case_id in sorted(corpus.cases):
        case = corpus.cases[case_id]
        paras = [p for p in substantive_paragraphs(case) if _STRONG_CONDITION.search(p.text)]
        rng.shuffle(paras)
        picks.extend((case, p) for p in paras[:2])  # at most two per judgment, to spread across cases
    rng.shuffle(picks)
    bases: list[_Base] = []
    cursor = 0
    while len(bases) < n and cursor < len(picks):
        batch = picks[cursor : cursor + int((n - len(bases)) * 1.5) + 2]
        cursor += len(batch)
        results = await asyncio.gather(
            *(_proposition(model, model_name, CONDITIONAL_PROPOSITION_PROMPT, p.text, "", usage, sem, "bench_conditional") for _, p in batch)
        )
        for (case, para), res in zip(batch, results, strict=True):
            if res is None or not res.proposition.strip():
                stats.reject("conditional_empty")
            elif not quote_in_text(res.supporting_quote, para.text):
                stats.reject("conditional_quote")
            elif len(bases) < n:
                bases.append(_Base(case, para, res.proposition.strip(), res.supporting_quote.strip()))
    return bases


async def near_paragraph(
    base: _Base,
    embeddings: Embeddings,
    emb_name: str,
    filter_model: BaseChatModel,
    filter_name: str,
    usage: UsageTracker,
    sem: asyncio.Semaphore,
    stats: GenStatsV1,
    top_n: int = 5,
    max_similarity: float = 0.75,
) -> tuple[Paragraph, float] | None:
    """The most similar other paragraph of the same judgment that does NOT support the claim."""
    cands = [
        p for p in base.case.paragraphs.values()
        if p.para_no > 0 and p.para_no != base.para.para_no and len(p.text) >= 200 and not quote_in_text(base.quote, p.text)
    ]
    if not cands:
        stats.reject("near_no_candidates")
        return None
    texts = [base.proposition] + [p.text[:2000] for p in cands]
    async with sem:
        vectors = np.asarray(await embeddings.aembed_documents(texts), dtype=np.float32)
    usage.record_embedding("bench_embedding", emb_name, texts)
    sims = _cos(vectors[:1], vectors[1:])[0]
    tried = 0
    for idx in np.argsort(-sims):
        if tried >= top_n:
            break
        if sims[int(idx)] > max_similarity:
            stats.reject("near_too_similar")  # almost a restatement of the claim: likely supports it
            continue
        tried += 1
        para = cands[int(idx)]
        verdict = await _ask(
            filter_model, filter_name, SUPPORT_FILTER_PROMPT,
            f"Paragraph:\n{para.text}\n\nClaim: {base.proposition}", _Support, usage, sem, "bench_filter", 60,
        )
        if verdict is None:
            stats.errors += 1
            continue
        if verdict.supports.strip().lower() == "no":
            return para, float(sims[int(idx)])
        stats.reject("near_candidate_supports_claim")  # "full" or "partial"
    stats.reject("near_all_candidates_support")
    return None


async def generate_items_v1(
    corpus: Corpus,
    gen_model: BaseChatModel,
    gen_name: str,
    filter_model: BaseChatModel,
    filter_name: str,
    embeddings: Embeddings,
    emb_name: str,
    usage: UsageTracker,
    *,
    seed: int = 13,
    n_case_bases: int = 50,
    n_wrong_code: int = 25,
    n_statute_none: int = 15,
    n_conditional_bases: int = 90,
    concurrency: int = 8,
    section_map: SectionMap | None = None,
) -> tuple[list[BenchItem], GenStatsV1]:
    from app.legal_india.benchmark.generate import GenStats

    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)
    stats = GenStatsV1()
    base_stats = GenStats()
    smap = section_map or load_section_map()

    case_bases = await make_case_bases(corpus, gen_model, gen_name, rng, n_case_bases, usage, sem, base_stats)
    statute_bases = await make_statute_bases(
        corpus, gen_model, gen_name, rng, n_wrong_code + n_statute_none, usage, sem, base_stats, smap
    )
    stats.attempted_bases = len(case_bases)
    stats.base_stats = base_stats.__dict__

    async def perturb(base: _Base):  # noqa: ANN202
        swap, alt, para, near = await asyncio.gather(
            _ask(gen_model, gen_name, SWAP_PROMPT, f"Statement: {base.proposition}", _Swap, usage, sem, "bench_swap"),
            _ask(gen_model, gen_name, ALTER_PROMPT_V1, f"Statement: {base.proposition}", _AlterV1, usage, sem, "bench_alter"),
            _ask(gen_model, gen_name, PARAPHRASE_PROMPT, f"Statement: {base.proposition}", _Paraphrase, usage, sem, "bench_paraphrase"),
            near_paragraph(base, embeddings, emb_name, filter_model, filter_name, usage, sem, stats),
        )
        return swap, alt, para, near

    async def overbroad_for(base: _Base) -> _Overbroad | None:
        user = f"Statement: {base.proposition}\nSource quote: {base.quote}"
        over = await _ask(gen_model, gen_name, OVERBROAD_PROMPT, user, _Overbroad, usage, sem, "bench_overbroad")
        if not check_overbroad(over, base):
            stats.reject("overbroad")
            return None
        # the broadened claim must not be FULLY supported by its source paragraph
        verdict = await _ask(
            filter_model, filter_name, SUPPORT_FILTER_PROMPT,
            f"Paragraph:\n{base.para.text}\n\nClaim: {over.overbroad}", _Support, usage, sem, "bench_filter", 60,
        )
        if verdict is None or verdict.supports.strip().lower() == "full":
            stats.reject("overbroad_still_supported")
            return None
        return over

    perturbed = await asyncio.gather(*(perturb(b) for b in case_bases))
    cond_bases = await make_conditional_bases(corpus, gen_model, gen_name, rng, n_conditional_bases, usage, sem, stats)
    cond_over = await asyncio.gather(*(overbroad_for(b) for b in cond_bases))

    splits = assign_splits(
        [b.case.case_id for b in case_bases] + [b.case.case_id for b in cond_bases] + [b.case.case_id for b in statute_bases], rng
    )
    items: list[BenchItem] = []

    def add(kind: str, base_id: str, split: str, **kw) -> None:  # noqa: ANN003
        kw["claim_text"] = end_with_period(kw["claim_text"])  # punctuation must not reveal the label
        items.append(BenchItem(id=f"{kind}-{base_id}", split=split, perturbation_type=kind, **kw))

    for k, (cb, over) in enumerate(zip(cond_bases, cond_over, strict=True)):
        if over is None:
            continue  # only bases whose overbroad version passed every check are used (control + corruption)
        authority = CitedAuthority(kind="case", citation=_citation_label(cb.case))
        common = dict(
            source_case_id=cb.case.case_id, source_para=cb.para.para_no, supporting_quote=cb.quote,
            original_claim=cb.proposition, judgment_date=cb.case.date, cited_para=cb.para.para_no,
        )
        add("none", f"k{k:03d}", splits[cb.case.case_id], should_flag=False, claim_text=cb.proposition,
            cited_authority=authority, meta={"variant": "conditional"}, **common)
        add("overbroad_holding", f"k{k:03d}", splits[cb.case.case_id], should_flag=True, claim_text=over.overbroad,
            cited_authority=authority, meta={"dropped": over.dropped}, **common)

    for i, (base, (swap, alt, para, near)) in enumerate(zip(case_bases, perturbed, strict=True)):
        case, src = base.case, base.para
        split = splits[case.case_id]
        bid = f"c{i:03d}"
        common = dict(
            source_case_id=case.case_id, source_para=src.para_no, supporting_quote=base.quote,
            original_claim=base.proposition, judgment_date=case.date,
        )
        authority = CitedAuthority(kind="case", citation=_citation_label(case))

        add("none", bid, split, should_flag=False, claim_text=base.proposition, cited_authority=authority,
            cited_para=src.para_no, **common)
        if check_paraphrase(para, base):
            add("none", bid + "p", split, should_flag=False, claim_text=para.paraphrase, cited_authority=authority,
                cited_para=src.para_no, meta={"variant": "paraphrase"}, **common)
        else:
            stats.reject("paraphrase")

        add("fabricated_case", bid, split, should_flag=True, claim_text=base.proposition,
            cited_authority=CitedAuthority(kind="case", citation=fabricated_citation(rng, corpus, case.date_obj.year)),
            cited_para=src.para_no, **common)

        wp = _pick_unrelated_paragraph(case, base.proposition + " " + base.quote, rng, {src.para_no})
        if wp is not None:
            add("wrong_paragraph", bid, split, should_flag=True, claim_text=base.proposition, cited_authority=authority,
                cited_para=wp.para_no, **common)

        if near is not None:
            np_para, sim = near
            add("near_paragraph", bid, split, should_flag=True, claim_text=base.proposition, cited_authority=authority,
                cited_para=np_para.para_no, meta={"similarity": f"{sim:.3f}"}, **common)

        if check_swap(swap, base):
            add("swapped_party_or_outcome", bid, split, should_flag=True, claim_text=swap.swapped, cited_authority=authority,
                cited_para=src.para_no, meta={"original": swap.original_element, "new": swap.new_element}, **common)
        else:
            stats.reject("swap")

        if check_alter(alt, base):
            add("altered_quote", bid, split, should_flag=True, claim_text=alt.altered, cited_authority=authority,
                cited_para=src.para_no, meta={"change": alt.change}, **common)
        else:
            stats.reject("alter")

    for j, sb in enumerate(statute_bases):
        split = splits[sb.case.case_id]
        common = dict(
            source_case_id=sb.case.case_id, source_para=sb.para.para_no, supporting_quote=sb.quote,
            original_claim=sb.proposition, judgment_date=sb.case.date, cited_para=sb.para.para_no,
        )
        if j < n_wrong_code:
            new_phrase = f"Section {sb.new_section} of the BNS"
            add("wrong_code", f"s{j:03d}", split, should_flag=True,
                claim_text=sb.proposition.replace(sb.phrase, new_phrase, 1),
                cited_authority=CitedAuthority(kind="statute", citation=new_phrase, act="BNS", provision=sb.new_section),
                meta={"ipc_section": sb.section}, **common)
        else:
            add("none", f"s{j:03d}", split, should_flag=False, claim_text=sb.proposition,
                cited_authority=CitedAuthority(kind="statute", citation=sb.phrase, act="IPC", provision=sb.section),
                meta={"ipc_section": sb.section}, **common)
    return items, stats
