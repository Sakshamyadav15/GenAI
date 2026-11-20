"""Generation of the adversarial citation benchmark from the judgment corpus.

Design: a *matched* design for case citations. Each base proposition (one sentence drawn from one
paragraph of one judgment) yields five items: the faithful control (``none``) and four corruptions
(fabricated_case, wrong_case, wrong_paragraph, altered_quote). Statute items come from paragraphs of
judgments decided before 1 July 2024, where any offence necessarily predates the BNS, so citing the
BNS is wrong by construction: ``wrong_code`` items cite the BNS, controls keep the IPC.

Labels are automatic by construction and have not been validated by lawyers.
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.structured import StructuredOutputError, structured_call
from app.core.usage import UsageTracker
from app.legal_india.benchmark.corpus import Case, Corpus, Paragraph, quote_in_text, substantive_paragraphs
from app.legal_india.benchmark.schema import BenchItem, CitedAuthority
from app.legal_india.statutes import extract_statutes
from app.legal_india.transition import SectionMap, load_section_map

CUTOFF = date(2024, 7, 1)

PROPOSITION_PROMPT = """You extract one checkable statement from a paragraph of an Indian court judgment.

Write ONE self-contained sentence stating a legal holding, rule, or fact that the paragraph itself states. It must make sense without the paragraph: name the parties, statute or issue instead of saying "the appellant" or "the said Act" alone. Do not add anything the paragraph does not say. Also copy one supporting quote of 12 to 40 words EXACTLY as it appears in the paragraph (keep its spelling, even if words look broken by the text extraction).

If the paragraph has no self-contained statement (only procedural history, headings, or a list of dates), return an empty proposition.

Return JSON: {"proposition": "...", "supporting_quote": "..."}"""

STATUTE_PROMPT = """You extract one checkable statement from a paragraph of an Indian court judgment that involves a specific statutory section.

Write ONE self-contained sentence stating something the paragraph says about the given section. The sentence MUST contain the given section phrase exactly, character for character. Do not add anything the paragraph does not say. Also copy one supporting quote of 12 to 40 words EXACTLY as it appears in the paragraph (keep its spelling, even if words look broken by the text extraction).

If you cannot do this faithfully, return an empty proposition.

Return JSON: {"proposition": "...", "supporting_quote": "..."}"""

ALTER_PROMPT = """You create a deliberately WRONG version of a legal statement, to test a citation checker.

Change the meaning of the statement so that it contradicts or no longer matches its source quote, in one of these ways: negate it, change a number, date or amount, reverse the outcome or who prevailed, or swap who did what. Keep it fluent, plausible and about the same length. Do not change any citation, section number or case name.

Return JSON: {"altered": "...", "change": "negation|number|outcome|actor"}"""


class _Prop(BaseModel):
    proposition: str = ""
    supporting_quote: str = ""


class _Alter(BaseModel):
    altered: str
    change: str = ""


@dataclass
class GenStats:
    attempted: int = 0
    rejected_empty: int = 0
    rejected_quote: int = 0
    rejected_phrase: int = 0
    rejected_alter: int = 0
    errors: int = 0
    notes: list[str] = field(default_factory=list)


_WORD = re.compile(r"[a-z]{4,}")


def overlap(a: str, b: str) -> float:
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(wa & wb) / max(1, len(wa | wb))


def end_with_period(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "?", "!")) else text + "."


def fabricated_citation(rng: random.Random, corpus: Corpus, year: int) -> str:
    """A plausible neutral citation that does not exist in the corpus (serial far above the real range)."""
    for _ in range(100):
        serial = rng.randint(1700, 9500)
        text = f"{year} INSC {serial}"
        if corpus.resolve(text) is None:
            return text
    raise RuntimeError("could not build a fabricated citation")


async def _proposition(
    model: BaseChatModel,
    model_name: str,
    prompt: str,
    paragraph: str,
    extra: str,
    usage: UsageTracker,
    sem: asyncio.Semaphore,
    stage: str,
) -> _Prop | None:
    async with sem:
        for attempt in range(4):
            try:
                return await structured_call(
                    model,
                    [SystemMessage(content=prompt), HumanMessage(content=f"{extra}Paragraph:\n{paragraph}")],
                    _Prop,
                    stage=stage,
                    model_name=model_name,
                    usage=usage,
                    max_tokens=400,
                )
            except StructuredOutputError:
                return None
            except Exception:  # noqa: BLE001 - transient network/API error: back off, then give up on this item
                if attempt == 3:
                    return None
                await asyncio.sleep(2.0 * (attempt + 1))
        return None


def _citation_label(case: Case) -> str:
    """Neutral citation form used in claims, e.g. '2023 INSC 650'."""
    return case.case_id


@dataclass
class _Base:
    case: Case
    para: Paragraph
    proposition: str
    quote: str


async def make_case_bases(
    corpus: Corpus,
    model: BaseChatModel,
    model_name: str,
    rng: random.Random,
    n: int,
    usage: UsageTracker,
    sem: asyncio.Semaphore,
    stats: GenStats,
) -> list[_Base]:
    case_ids = sorted(corpus.cases)
    rng.shuffle(case_ids)
    bases: list[_Base] = []
    cursor = 0
    while len(bases) < n and cursor < len(case_ids):
        batch_ids = case_ids[cursor : cursor + int((n - len(bases)) * 1.6) + 2]
        cursor += len(batch_ids)
        picks = []
        for cid in batch_ids:
            paras = substantive_paragraphs(corpus.cases[cid])
            if paras:
                picks.append((corpus.cases[cid], rng.choice(paras)))
        stats.attempted += len(picks)
        results = await asyncio.gather(
            *(_proposition(model, model_name, PROPOSITION_PROMPT, p.text, "", usage, sem, "bench_proposition") for _, p in picks)
        )
        for (case, para), res in zip(picks, results, strict=True):
            if res is None:
                stats.errors += 1
            elif not res.proposition.strip():
                stats.rejected_empty += 1
            elif not quote_in_text(res.supporting_quote, para.text):
                stats.rejected_quote += 1
            elif len(bases) < n:
                bases.append(_Base(case, para, res.proposition.strip(), res.supporting_quote.strip()))
    return bases


@dataclass
class _StatuteBase:
    case: Case
    para: Paragraph
    section: str  # IPC section, e.g. "302"
    phrase: str  # "Section 302 of the IPC"
    proposition: str
    quote: str
    new_section: str


def _section_phrase(proposition: str, section: str) -> str | None:
    """The exact text of a single-section IPC mention of ``section`` in the proposition, if there is one.

    Mentions that list several sections ("Sections 302 and 34 of the IPC") are rejected because replacing
    them with one BNS section would change the claim.
    """
    refs = extract_statutes(proposition)
    for ref in refs:
        if ref.act == "IPC" and ref.base == section and ref.raw in proposition:
            if sum(1 for r in refs if r.raw == ref.raw) == 1:  # one section per mention
                return ref.raw.rstrip(" .,;:")
    return None


async def make_statute_bases(
    corpus: Corpus,
    model: BaseChatModel,
    model_name: str,
    rng: random.Random,
    n: int,
    usage: UsageTracker,
    sem: asyncio.Semaphore,
    stats: GenStats,
    section_map: SectionMap,
) -> list[_StatuteBase]:
    candidates: list[tuple[Case, Paragraph, str]] = []
    for case in corpus.cases.values():
        if case.date_obj >= CUTOFF:
            continue  # offence date cannot be bounded below the cut-off
        per_case = 0
        paras = list(case.paragraphs.values())
        rng.shuffle(paras)
        for para in paras:
            if para.para_no == 0 or not 250 <= len(para.text) <= 2600 or per_case >= 4:
                continue
            refs = [r for r in extract_statutes(para.text) if r.act == "IPC" and section_map.to_new("IPC", r.base)]
            if refs:
                candidates.append((case, para, rng.choice(refs).base))
                per_case += 1
    rng.shuffle(candidates)
    bases: list[_StatuteBase] = []
    cursor = 0
    while len(bases) < n and cursor < len(candidates):
        batch = candidates[cursor : cursor + int((n - len(bases)) * 1.6) + 2]
        cursor += len(batch)
        stats.attempted += len(batch)
        results = await asyncio.gather(
            *(
                _proposition(
                    model, model_name, STATUTE_PROMPT, para.text,
                    f'Section phrase (use exactly): "Section {sec} of the IPC"\n\n', usage, sem, "bench_statute_proposition",
                )
                for _, para, sec in batch
            )
        )
        for (case, para, sec), res in zip(batch, results, strict=True):
            phrase = _section_phrase(res.proposition, sec) if res is not None else None
            if res is None:
                stats.errors += 1
            elif not res.proposition.strip():
                stats.rejected_empty += 1
            elif phrase is None:
                stats.rejected_phrase += 1
            elif not quote_in_text(res.supporting_quote, para.text):
                stats.rejected_quote += 1
            elif len(bases) < n:
                entry = section_map.to_new("IPC", sec)[0]
                bases.append(_StatuteBase(case, para, sec, phrase, res.proposition.strip(), res.supporting_quote.strip(), entry.new))
    return bases


async def _alter(
    model: BaseChatModel, model_name: str, base: _Base, usage: UsageTracker, sem: asyncio.Semaphore, stats: GenStats
) -> tuple[str, str] | None:
    async with sem:
        try:
            res = await structured_call(
                model,
                [
                    SystemMessage(content=ALTER_PROMPT),
                    HumanMessage(content=f"Statement: {base.proposition}\nSource quote: {base.quote}"),
                ],
                _Alter,
                stage="bench_alter",
                model_name=model_name,
                usage=usage,
                max_tokens=300,
            )
        except StructuredOutputError:
            stats.errors += 1
            return None
    altered = res.altered.strip()
    if not altered or altered.lower() == base.proposition.lower() or overlap(altered, base.proposition) < 0.35:
        stats.rejected_alter += 1
        return None
    return altered, res.change


def assign_splits(case_ids: Sequence[str], rng: random.Random, dev_fraction: float = 0.3) -> dict[str, str]:
    """Split by source case, so that no judgment contributes to both dev and test."""
    ids = sorted(set(case_ids))
    rng.shuffle(ids)
    cut = round(len(ids) * dev_fraction)
    return {cid: ("dev" if i < cut else "test") for i, cid in enumerate(ids)}


def _pick_unrelated_paragraph(case: Case, text: str, rng: random.Random, exclude: set[int]) -> Paragraph | None:
    options = [p for p in case.paragraphs.values() if p.para_no > 0 and p.para_no not in exclude and len(p.text) >= 200]
    scored = sorted(options, key=lambda p: overlap(p.text, text))
    low = [p for p in scored[:8] if overlap(p.text, text) < 0.12]
    return rng.choice(low) if low else None


async def generate_items(
    corpus: Corpus,
    model: BaseChatModel,
    model_name: str,
    usage: UsageTracker,
    *,
    seed: int = 13,
    n_case_bases: int = 50,
    n_wrong_code: int = 50,
    n_statute_none: int = 25,
    concurrency: int = 8,
    section_map: SectionMap | None = None,
) -> tuple[list[BenchItem], GenStats]:
    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)
    stats = GenStats()
    smap = section_map or load_section_map()

    case_bases = await make_case_bases(corpus, model, model_name, rng, n_case_bases, usage, sem, stats)
    statute_bases = await make_statute_bases(
        corpus, model, model_name, rng, n_wrong_code + n_statute_none, usage, sem, stats, smap
    )
    alters = await asyncio.gather(*(_alter(model, model_name, b, usage, sem, stats) for b in case_bases))

    splits = assign_splits([b.case.case_id for b in case_bases] + [b.case.case_id for b in statute_bases], rng)
    items: list[BenchItem] = []
    all_case_ids = sorted(corpus.cases)

    def add(kind: str, base_id: str, split: str, **kw) -> None:  # noqa: ANN003
        kw["claim_text"] = end_with_period(kw["claim_text"])  # punctuation must not reveal the label
        items.append(BenchItem(id=f"{kind}-{base_id}", split=split, perturbation_type=kind, **kw))

    for i, (base, alt) in enumerate(zip(case_bases, alters, strict=True)):
        case, para = base.case, base.para
        split = splits[case.case_id]
        common = dict(
            source_case_id=case.case_id,
            source_para=para.para_no,
            supporting_quote=base.quote,
            original_claim=base.proposition,
            judgment_date=case.date,
        )
        authority = CitedAuthority(kind="case", citation=_citation_label(case))
        add("none", f"c{i:03d}", split, should_flag=False, claim_text=base.proposition, cited_authority=authority, cited_para=para.para_no, **common)

        fake = fabricated_citation(rng, corpus, case.date_obj.year)
        add("fabricated_case", f"c{i:03d}", split, should_flag=True, claim_text=base.proposition,
            cited_authority=CitedAuthority(kind="case", citation=fake), cited_para=para.para_no, **common)

        for _ in range(20):
            other_id = rng.choice(all_case_ids)
            if other_id == case.case_id:
                continue
            other = corpus.cases[other_id]
            cands = [p for p in other.paragraphs.values() if p.para_no > 0 and len(p.text) >= 200 and overlap(p.text, base.proposition) < 0.12]
            if cands:
                op = rng.choice(cands)
                add("wrong_case", f"c{i:03d}", split, should_flag=True, claim_text=base.proposition,
                    cited_authority=CitedAuthority(kind="case", citation=_citation_label(other)), cited_para=op.para_no, **common)
                break

        wp = _pick_unrelated_paragraph(case, base.proposition + " " + base.quote, rng, {para.para_no})
        if wp is not None:
            add("wrong_paragraph", f"c{i:03d}", split, should_flag=True, claim_text=base.proposition,
                cited_authority=authority, cited_para=wp.para_no, **common)

        if alt is not None:
            altered, change = alt
            add("altered_quote", f"c{i:03d}", split, should_flag=True, claim_text=altered,
                cited_authority=authority, cited_para=para.para_no, meta={"change": change}, **{**common})

    for j, sb in enumerate(statute_bases):
        split = splits[sb.case.case_id]
        common = dict(
            source_case_id=sb.case.case_id,
            source_para=sb.para.para_no,
            supporting_quote=sb.quote,
            original_claim=sb.proposition,
            judgment_date=sb.case.date,
            cited_para=sb.para.para_no,
        )
        if j < n_wrong_code:
            new_phrase = f"Section {sb.new_section} of the BNS"
            add("wrong_code", f"s{j:03d}", split, should_flag=True, claim_text=sb.proposition.replace(sb.phrase, new_phrase, 1),
                cited_authority=CitedAuthority(kind="statute", citation=new_phrase, act="BNS", provision=sb.new_section),
                meta={"ipc_section": sb.section}, **common)
        else:
            add("none", f"s{j:03d}", split, should_flag=False, claim_text=sb.proposition,
                cited_authority=CitedAuthority(kind="statute", citation=sb.phrase, act="IPC", provision=sb.section),
                meta={"ipc_section": sb.section}, **common)
    return items, stats
