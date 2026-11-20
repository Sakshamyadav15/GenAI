"""Benchmark generation, baselines and metrics, with fake LLM/NLI (no network, no API)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from app.core.usage import UsageTracker
from app.legal_india.benchmark import baselines
from app.legal_india.benchmark.corpus import load_corpus, quote_in_text, squash
from app.legal_india.benchmark.generate import (
    ALTER_PROMPT,
    PROPOSITION_PROMPT,
    STATUTE_PROMPT,
    _section_phrase,
    assign_splits,
    end_with_period,
    fabricated_citation,
    generate_items,
    overlap,
)
from app.legal_india.benchmark.metrics import bootstrap_ci, evaluate, prf
from app.legal_india.benchmark.schema import BenchItem
from app.verification.nli import NLIResult
from tests.fakes import ScriptedChatModel


_WORDS = ("river", "contract", "tenant", "bail", "tariff", "pension", "forest", "arbitral", "mortgage", "survey",
          "teacher", "election", "limitation", "easement", "customs", "dowry", "railway", "patent", "tribunal", "gratuity")


def _case(i: int, year: int = 2023, month: int = 5) -> dict:
    """Paragraphs of a case use disjoint vocabulary so that "unrelated paragraph" selection is possible."""
    rng = random.Random(i)
    paras = [{"para_no": 0, "text": "HEADER"}]
    for n in range(1, 8):
        vocab = " ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(7)) for _ in range(40))
        paras.append(
            {
                "para_no": n,
                "text": (
                    f"Para {n} of case {i}. The appellant Ramesh{i} was convicted under Section 302 of the IPC by the trial "
                    f"court and the High Court confirmed the sentence of life imprisonment. {vocab}. "
                    f"Further reasoning is recorded here at length for the generator selection rules to accept this paragraph."
                ),
            }
        )
    return {
        "case_id": f"{year} INSC {100 + i}",
        "title": f"T{i}",
        "date": f"10-{month:02d}-{year}",
        "citation": f"[{year}] 9 S.C.R. {200 + i}",
        "paragraphs": paras,
    }


@pytest.fixture
def corpus(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(_case(i, 2023 if i % 2 else 2024, 3)) for i in range(40)), encoding="utf-8")
    return load_corpus(path)


def responder(messages) -> str:
    system, user = str(messages[0].content), str(messages[-1].content)
    para = user.split("Paragraph:\n", 1)[-1]
    sentences = para.split(". ")
    first = sentences[2] if len(sentences) > 2 else para  # the distinctive (vocabulary) part of the paragraph
    quote = " ".join(first.split()[:12]).rstrip(".")
    if system == PROPOSITION_PROMPT:
        return json.dumps({"proposition": f"The court held that {first[:90]}.", "supporting_quote": quote})
    if system == STATUTE_PROMPT:
        return json.dumps({"proposition": "The appellant was convicted under Section 302 of the IPC.", "supporting_quote": " ".join(para.split()[5:20]).rstrip(".")})
    if system == ALTER_PROMPT:
        statement = user.split("Statement: ", 1)[1].split("\n", 1)[0]
        return json.dumps({"altered": statement.replace("held", "did not hold", 1), "change": "negation"})
    raise AssertionError(system[:40])


async def _generate(corpus, **kw):
    model = ScriptedChatModel(responder=responder)
    return await generate_items(corpus, model, "fake", UsageTracker({}), seed=1, **kw)


def test_helpers() -> None:
    assert squash("Refere nce  To") == "referenceto"
    assert quote_in_text("the appellant was convicted under section 302", "The appel lant was convic ted under Section 302 of the IPC.")
    assert not quote_in_text("short", "short text")  # too short to be evidence
    assert end_with_period("abc") == "abc." and end_with_period("abc?") == "abc?"
    assert overlap("alpha beta gamma delta", "alpha beta gamma delta") == 1.0
    assert _section_phrase("He was convicted under Section 302 of the IPC.", "302") == "Section 302 of the IPC"
    assert _section_phrase("Sections 302 and 34 of the IPC apply.", "302") is None  # lists are rejected


def test_splits_are_by_case_and_deterministic() -> None:
    ids = [f"c{i}" for i in range(20)] * 3
    a = assign_splits(ids, random.Random(5))
    b = assign_splits(ids, random.Random(5))
    assert a == b and len(a) == 20
    assert sum(v == "dev" for v in a.values()) == 6  # 30%


def test_fabricated_citation_never_resolves(corpus) -> None:
    rng = random.Random(1)
    for _ in range(30):
        assert corpus.resolve(fabricated_citation(rng, corpus, 2023)) is None


async def test_generation_builds_matched_items_with_valid_labels(corpus) -> None:
    items, stats = await _generate(corpus, n_case_bases=8, n_wrong_code=4, n_statute_none=2)
    by_type = {t: [i for i in items if i.perturbation_type == t] for t in {i.perturbation_type for i in items}}
    assert len(by_type["none"]) == 8 + 2
    assert all(not i.should_flag for i in by_type["none"])
    for t in ("fabricated_case", "wrong_case", "wrong_paragraph", "altered_quote", "wrong_code"):
        assert by_type[t] and all(i.should_flag for i in by_type[t])
    # fabricated citations do not resolve; real ones do
    assert all(corpus.resolve(i.cited_authority.citation) is None for i in by_type["fabricated_case"])
    assert all(corpus.resolve(i.cited_authority.citation) is not None for i in by_type["none"] if i.cited_authority.kind == "case")
    # wrong_case cites another case; wrong_paragraph cites a different paragraph of the same case
    assert all(corpus.resolve(i.cited_authority.citation).case_id != i.source_case_id for i in by_type["wrong_case"])
    assert all(i.cited_para != i.source_para for i in by_type["wrong_paragraph"])
    # altered claims differ from the original, controls do not
    assert all(end_with_period(i.original_claim) != i.claim_text for i in by_type["altered_quote"])
    # wrong_code: BNS cited, judgment predates the cut-off (so the BNS cannot apply)
    for i in by_type["wrong_code"]:
        assert i.cited_authority.act == "BNS" and "BNS" in i.claim_text and "IPC" not in i.claim_text
        assert i.judgment_date.endswith("2023") or i.judgment_date.endswith("2024")
    # all supporting quotes really occur in the source paragraph
    for i in items:
        assert quote_in_text(i.supporting_quote, corpus.cases[i.source_case_id].paragraphs[i.source_para].text)
    # no judgment contributes to both dev and test
    cases_by_split: dict[str, set[str]] = {"dev": set(), "test": set()}
    for i in items:
        cases_by_split[i.split].add(i.source_case_id)
    assert not cases_by_split["dev"] & cases_by_split["test"]
    # identical output for the same seed
    again, _ = await _generate(corpus, n_case_bases=8, n_wrong_code=4, n_statute_none=2)
    assert [i.model_dump() for i in items] == [i.model_dump() for i in again]


async def test_quotes_not_in_paragraph_are_rejected(corpus) -> None:
    def bad(messages) -> str:
        return json.dumps({"proposition": "A claim.", "supporting_quote": "this quote does not occur anywhere in the paragraph at all"})

    model = ScriptedChatModel(responder=bad)
    items, stats = await generate_items(corpus, model, "fake", UsageTracker({}), seed=1, n_case_bases=3, n_wrong_code=1, n_statute_none=1)
    assert items == [] and stats.rejected_quote > 0


def _item(i: int, kind: str, flag: bool, split: str = "test", citation: str = "2023 INSC 101", para: int = 1, claim: str = "c.") -> BenchItem:
    return BenchItem(
        id=f"i{i}", split=split, perturbation_type=kind, should_flag=flag, claim_text=claim,
        cited_authority={"kind": "case", "citation": citation}, cited_para=para, source_case_id="2023 INSC 101", source_para=1,
        supporting_quote="q" * 30, original_claim=claim, judgment_date="10-03-2023",
    )


def test_b0_flags_only_unresolvable_citations(corpus) -> None:
    items = [_item(0, "none", False, citation="2023 INSC 101"), _item(1, "fabricated_case", True, citation="2023 INSC 9999")]
    d = baselines.check_b0(items, corpus)
    assert [x.flagged for x in d] == [False, True]


async def test_b1_uses_the_cited_paragraph_only(corpus) -> None:
    seen: list[str] = []

    def judge(messages) -> str:
        seen.append(str(messages[-1].content))
        return json.dumps({"verdict": "supported" if "Para 1 of case 1" in str(messages[-1].content) else "unsupported", "reason": "x"})

    model = ScriptedChatModel(responder=judge)
    case = next(iter(corpus.cases.values()))
    cit = case.case_id
    items = [_item(0, "none", False, citation=cit, para=1), _item(1, "wrong_paragraph", True, citation=cit, para=2), _item(2, "fabricated_case", True, citation="2023 INSC 9999")]
    d = await baselines.check_b1(items, corpus, model, "fake", UsageTracker({}))
    assert len(model.calls) == 2  # the unresolvable citation needs no LLM call
    assert d[2].flagged and d[2].reason == "no cited text to check against"
    assert all("Paragraph:" in s and "Claim:" in s for s in seen)


class _FakeNLI:
    name = "fake"

    def score(self, pairs):
        return [NLIResult(0.9 if h.split()[0] in p else 0.1, 0.05, 0.05) for p, h in pairs]


def test_b2_scores_and_dev_threshold(corpus) -> None:
    case = next(iter(corpus.cases.values()))
    ok = BenchItem(**{**_item(0, "none", False, citation=case.case_id, para=1).model_dump(), "claim_text": "Para supported."})
    bad = BenchItem(**{**_item(1, "altered_quote", True, citation=case.case_id, para=1).model_dump(), "claim_text": "Zzz unsupported."})
    scored = baselines.score_b2([ok, bad], corpus, _FakeNLI())
    assert scored[0].score > 0.8 and scored[1].score < 0.2
    t = baselines.tune_threshold(scored, [ok, bad])
    assert 0.1 < t <= 0.9
    flags = [d.flagged for d in baselines.apply_threshold(scored, t)]
    assert flags == [False, True]


def test_rule_check_flags_bns_for_pre_cutoff_judgments() -> None:
    base = _item(0, "wrong_code", True).model_dump()
    stat = {"kind": "statute", "citation": "Section 103 of the BNS", "act": "BNS", "provision": "103"}
    wrong = BenchItem(**{**base, "cited_authority": stat, "claim_text": "He was convicted under Section 103 of the BNS."})
    right = BenchItem(**{**base, "id": "r", "perturbation_type": "none", "should_flag": False,
                         "cited_authority": {**stat, "act": "IPC", "citation": "Section 302 of the IPC"},
                         "claim_text": "He was convicted under Section 302 of the IPC."})
    case_item = _item(2, "none", False)
    d = baselines.check_rule_wrong_code([wrong, right, case_item])
    assert [x.flagged for x in d] == [True, False, False]


def test_prf_and_bootstrap() -> None:
    assert prf(8, 2, 2) == pytest.approx((0.8, 0.8, 0.8))
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    rows = [(True, True)] * 40 + [(True, False)] * 10 + [(False, False)] * 50
    lo, hi = bootstrap_ci(rows, "recall", n_boot=300)
    assert lo <= 0.8 <= hi and hi - lo < 0.3


def test_evaluate_uses_only_the_requested_split_and_matched_controls() -> None:
    items = [
        _item(0, "none", False), _item(1, "none", False), _item(2, "altered_quote", True), _item(3, "altered_quote", True),
        _item(4, "none", False, split="dev"), _item(5, "altered_quote", True, split="dev"),
    ]
    decisions = [baselines.Decision("i0", False), baselines.Decision("i1", True), baselines.Decision("i2", True),
                 baselines.Decision("i3", False), baselines.Decision("i4", True), baselines.Decision("i5", True)]
    r = evaluate(items, decisions, split="test", n_boot=50)
    assert r["overall"]["n"] == 4
    assert r["overall"]["recall"]["value"] == 0.5 and r["overall"]["false_flag_rate"]["value"] == 0.5
    q = r["per_type"]["altered_quote"]
    assert q["n_type"] == 2 and q["n_controls"] == 2 and q["precision"]["value"] == 0.5 and q["recall"]["value"] == 0.5
    assert "wrong_code" not in r["per_type"]
