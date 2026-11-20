"""Benchmark v1: near-miss generators, mechanical label checks, confidence-bearing checker, NLI cache,
risk-coverage. Fake LLM / embeddings / NLI only (no network, no API)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from app.core.usage import UsageTracker
from app.legal_india.benchmark import baselines
from app.legal_india.benchmark import generate as v0
from app.legal_india.benchmark import generate_v1 as v1
from app.legal_india.benchmark.corpus import load_corpus, quote_in_text
from app.legal_india.benchmark.metrics import evaluate, risk_coverage
from app.legal_india.benchmark.schema import BenchItem
from app.verification.nli import NLIResult
from tests.fakes import HashingEmbeddings, ScriptedChatModel

CONDITION = "The notice stands only if the notice is served within 30 days, unless the court extends time."


def _case(i: int, year: int = 2023) -> dict:
    rng = random.Random(i)
    paras = [{"para_no": 0, "text": "HEADER"}]
    for n in range(1, 8):
        vocab = " ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(7)) for _ in range(40))
        paras.append(
            {
                "para_no": n,
                "text": (
                    f"Para {n} of case {i}. The appellant Ramesh{i} was convicted under Section 302 of the IPC by the trial "
                    f"court and the High Court confirmed the sentence of life imprisonment. {vocab}. {CONDITION} "
                    f"Further reasoning is recorded here at length for the generator selection rules to accept this paragraph."
                ),
            }
        )
    return {
        "case_id": f"{year} INSC {100 + i}",
        "title": f"T{i}",
        "date": f"10-03-{year}",
        "citation": f"[{year}] 9 S.C.R. {200 + i}",
        "paragraphs": paras,
    }


@pytest.fixture
def corpus(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(_case(i, 2023 if i % 2 else 2024)) for i in range(40)), encoding="utf-8")
    return load_corpus(path)


def responder(messages) -> str:
    system, user = str(messages[0].content), str(messages[-1].content)
    para = user.split("Paragraph:\n", 1)[-1]
    statement = user.split("Statement: ", 1)[-1].split("\n", 1)[0]
    if system == v0.PROPOSITION_PROMPT:
        first = para.split(". ")[2] if len(para.split(". ")) > 2 else para
        quote = " ".join(first.split()[:12]).rstrip(".")
        return json.dumps({"proposition": f"The court held that {first[:90]}.", "supporting_quote": quote})
    if system == v1.CONDITIONAL_PROPOSITION_PROMPT:
        return json.dumps({"proposition": CONDITION, "supporting_quote": CONDITION.rstrip(".")})
    if system == v0.STATUTE_PROMPT:
        return json.dumps({"proposition": "The appellant was convicted under Section 302 of the IPC.", "supporting_quote": " ".join(para.split()[5:20]).rstrip(".")})
    if system == v1.OVERBROAD_PROMPT:
        return json.dumps({"overbroad": "The notice stands only if the notice is served within some period.", "dropped": "unless the court extends time"})
    if system == v1.SWAP_PROMPT:
        return json.dumps({"swapped": statement.replace("held", "denied", 1), "original_element": "held", "new_element": "denied"})
    if system == v1.ALTER_PROMPT_V1:
        return json.dumps({"altered": statement.replace("held", "did not hold", 1), "change": "negation"})
    if system == v1.PARAPHRASE_PROMPT:
        return json.dumps({"paraphrase": statement.replace("The court held that", "According to the bench it was decided that", 1)})
    if system == v1.SUPPORT_FILTER_PROMPT:
        claim = user.rsplit("Claim: ", 1)[-1]
        return json.dumps({"supports": "full" if "FULLY" in claim else "no"})
    raise AssertionError(system[:50])


async def _gen(corpus, resp=responder, **kw):
    gen = ScriptedChatModel(responder=resp)
    filt = ScriptedChatModel(responder=resp)
    kw = {"n_case_bases": 6, "n_wrong_code": 3, "n_statute_none": 2, "n_conditional_bases": 12, **kw}
    return await v1.generate_items_v1(
        corpus, gen, "fake-gen", filt, "fake-filter", HashingEmbeddings(), "fake-emb", UsageTracker({}), seed=3, **kw
    )


# ----------------------------------------------------------------------------- mechanical checks


def _base(corpus) -> v0._Base:
    case = corpus.cases["2023 INSC 101"]
    para = case.paragraphs[1]
    return v0._Base(case, para, CONDITION, CONDITION.rstrip("."))


def test_numbers_in() -> None:
    assert v1.numbers_in("Rs. 3.52 crores in 2023, section 302.") == {"3.52", "2023", "302"}


def test_check_overbroad_requires_a_real_condition(corpus) -> None:
    base = _base(corpus)
    good = v1._Overbroad(overbroad="The notice stands only if the notice is served within some period.", dropped="unless the court extends time")
    assert v1.check_overbroad(good, base)
    stylistic = v1._Overbroad(overbroad="The notice stands only if the notice is served within some period.", dropped="by the trial court and the High Court")
    assert not v1.check_overbroad(stylistic, base)  # no qualifier word: not a condition
    leaked = v1._Overbroad(overbroad="The notice stands unless the court extends time.", dropped="unless the court extends time")
    assert not v1.check_overbroad(leaked, base)  # the dropped words are still in the new claim
    invented = v1._Overbroad(overbroad="The notice stands only if the notice is served within some period.", dropped="unless the Moon extends time")
    assert not v1.check_overbroad(invented, base)  # dropped words are not in the paragraph
    assert not v1.check_overbroad(None, base)


def test_check_swap_rejects_bare_negation_and_noops(corpus) -> None:
    base = v0._Base(corpus.cases["2023 INSC 101"], corpus.cases["2023 INSC 101"].paragraphs[1],
                    "The Supreme Court allowed the appeal and set aside the order of the High Court.", "q" * 30)
    ok = v1._Swap(swapped="The Supreme Court dismissed the appeal and set aside the order of the High Court.", original_element="allowed", new_element="dismissed")
    assert v1.check_swap(ok, base)
    negation = v1._Swap(swapped="The Supreme Court did not allow the appeal and set aside the order of the High Court.", original_element="allowed", new_element="did not allow")
    assert not v1.check_swap(negation, base)
    noop = v1._Swap(swapped=base.proposition, original_element="allowed", new_element="allowed")
    assert not v1.check_swap(noop, base)
    unchanged = v1._Swap(swapped="The Supreme Court allowed the appeal and quashed the order of the High Court.", original_element="allowed", new_element="quashed")
    assert not v1.check_swap(unchanged, base)  # the original element is still in the new claim


def test_check_paraphrase_keeps_numbers(corpus) -> None:
    base = v0._Base(corpus.cases["2023 INSC 101"], corpus.cases["2023 INSC 101"].paragraphs[1],
                    "The Bank transferred Rs. 3.52 crores to the Collector within 30 days of the order.", "q" * 30)
    assert v1.check_paraphrase(v1._Paraphrase(paraphrase="Within 30 days of the order, the Collector received Rs. 3.52 crores from the Bank."), base)
    assert not v1.check_paraphrase(v1._Paraphrase(paraphrase="The Bank paid the Collector a large sum soon after the order."), base)  # numbers lost
    assert not v1.check_paraphrase(v1._Paraphrase(paraphrase=base.proposition), base)  # identical


# ----------------------------------------------------------------------------- generation


async def test_v1_generation_types_labels_and_checks(corpus) -> None:
    items, stats = await _gen(corpus)
    by_type: dict[str, list[BenchItem]] = {}
    for i in items:
        by_type.setdefault(i.perturbation_type, []).append(i)
    assert {"none", "fabricated_case", "wrong_paragraph", "near_paragraph", "overbroad_holding",
            "swapped_party_or_outcome", "altered_quote", "wrong_code"} <= set(by_type)
    assert "wrong_court_binding" not in by_type and "wrong_case" not in by_type
    assert all(not i.should_flag for i in by_type["none"])
    assert all(i.should_flag for t, v in by_type.items() if t != "none" for i in v)
    assert any(i.meta.get("variant") == "paraphrase" for i in by_type["none"])  # paraphrase controls exist

    for i in by_type["near_paragraph"]:
        assert i.cited_para != i.source_para
        case = corpus.cases[i.source_case_id]
        assert not quote_in_text(i.supporting_quote, case.paragraphs[i.cited_para].text)
        assert float(i.meta["similarity"]) <= 0.75
        assert i.claim_text == v0.end_with_period(i.original_claim)  # the claim is faithful, the paragraph is wrong
    for i in by_type["overbroad_holding"]:
        assert i.meta["dropped"].lower() not in i.claim_text.lower() and i.claim_text != v0.end_with_period(i.original_claim)
    for i in by_type["swapped_party_or_outcome"]:
        assert i.meta["original"] != i.meta["new"] and i.claim_text != v0.end_with_period(i.original_claim)
    for i in items:
        assert quote_in_text(i.supporting_quote, corpus.cases[i.source_case_id].paragraphs[i.source_para].text)
        assert i.claim_text.endswith((".", "?", "!"))
    splits: dict[str, set[str]] = {"dev": set(), "test": set()}
    for i in items:
        splits[i.split].add(i.source_case_id)
    assert not splits["dev"] & splits["test"]
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids))
    assert stats.attempted_bases == 6


async def test_v1_generation_is_deterministic(corpus) -> None:
    a, _ = await _gen(corpus)
    b, _ = await _gen(corpus)
    assert [i.model_dump() for i in a] == [i.model_dump() for i in b]


async def test_near_paragraph_is_dropped_when_the_filter_says_it_supports_the_claim(corpus) -> None:
    def strict(messages) -> str:
        if str(messages[0].content) == v1.SUPPORT_FILTER_PROMPT:
            return json.dumps({"supports": "partial"})  # "partial" must also reject the candidate
        return responder(messages)

    items, stats = await _gen(corpus, strict)
    assert not [i for i in items if i.perturbation_type == "near_paragraph"]
    assert stats.rejected.get("near_candidate_supports_claim", 0) > 0


async def test_overbroad_is_dropped_when_the_source_still_fully_supports_it(corpus) -> None:
    def lenient(messages) -> str:
        if str(messages[0].content) == v1.OVERBROAD_PROMPT:
            return json.dumps({"overbroad": "The notice stands only if the notice is served within FULLY.", "dropped": "unless the court extends time"})
        return responder(messages)

    def filt(messages) -> str:
        if str(messages[0].content) == v1.SUPPORT_FILTER_PROMPT:
            return json.dumps({"supports": "full" if "FULLY" in str(messages[-1].content) else "no"})
        return responder(messages)

    items, stats = await v1.generate_items_v1(
        corpus, ScriptedChatModel(responder=lenient), "g", ScriptedChatModel(responder=filt), "f", HashingEmbeddings(), "e",
        UsageTracker({}), seed=3, n_case_bases=2, n_wrong_code=0, n_statute_none=0, n_conditional_bases=6,
    )
    assert not [i for i in items if i.perturbation_type == "overbroad_holding"]
    assert stats.rejected.get("overbroad_still_supported", 0) > 0


async def test_network_errors_are_retried_then_skipped(corpus, monkeypatch) -> None:
    calls = {"n": 0}

    async def flaky(*a, **k):  # noqa: ANN002, ANN003
        calls["n"] += 1
        raise ConnectionError("connection reset")

    async def no_sleep(_):  # noqa: ANN001
        return None

    monkeypatch.setattr(v1, "structured_call", flaky)
    monkeypatch.setattr(v1.asyncio, "sleep", no_sleep)
    import asyncio

    out = await v1._ask(None, "m", "s", "u", v1._Support, UsageTracker({}), asyncio.Semaphore(1), "t")
    assert out is None and calls["n"] == 4


# ----------------------------------------------------------------------------- checkers and metrics


def _item(i: int, kind: str, flag: bool, citation: str = "2023 INSC 101", para: int = 1) -> BenchItem:
    return BenchItem(
        id=f"i{i}", split="test", perturbation_type=kind, should_flag=flag, claim_text="c.",
        cited_authority={"kind": "case", "citation": citation}, cited_para=para, source_case_id="2023 INSC 101", source_para=1,
        supporting_quote="q" * 30, original_claim="c.", judgment_date="10-03-2023",
    )


async def test_b1_conf_score_is_the_support_probability(corpus) -> None:
    def judge(messages) -> str:
        claim = str(messages[-1].content).rsplit("Claim: ", 1)[-1]
        return json.dumps({"verdict": "supported" if claim.startswith("good") else "unsupported", "confidence": 0.9 if claim.startswith("good") else 0.8, "reason": "x"})

    items = [_item(0, "none", False), _item(1, "near_paragraph", True), _item(2, "fabricated_case", True, citation="2023 INSC 9999")]
    items[0] = items[0].model_copy(update={"claim_text": "good claim."})
    items[1] = items[1].model_copy(update={"claim_text": "bad claim."})
    d = await baselines.check_b1_conf(items, corpus, ScriptedChatModel(responder=judge), "fake", UsageTracker({}))
    assert (d[0].flagged, round(d[0].score, 2)) == (False, 0.9)
    assert (d[1].flagged, round(d[1].score, 2)) == (True, 0.2)  # unsupported with confidence 0.8 -> support probability 0.2
    assert d[2].flagged and d[2].score == 0.0  # unresolvable citation


class _CountingNLI:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, pairs):  # noqa: ANN001, ANN201
        self.calls += len(pairs)
        return [NLIResult(0.9 if "yes" in h else 0.1, 0.05, 0.05) for _, h in pairs]


def test_cached_nli_scores_each_pair_once_and_persists(tmp_path) -> None:
    path = tmp_path / "cache.json"
    inner = _CountingNLI()
    nli = baselines.CachedNLI(inner, path)
    pairs = [("premise one", "yes claim"), ("premise two", "no claim"), ("premise one", "yes claim")]
    first = nli.score(pairs)
    assert inner.calls == 2 and [round(r.entailment, 1) for r in first] == [0.9, 0.1, 0.9]
    nli.score(pairs)
    assert inner.calls == 2 and nli.hits >= 3  # second call is served from the cache
    reloaded = baselines.CachedNLI(_CountingNLI(), path)
    assert [round(r.entailment, 1) for r in reloaded.score(pairs)] == [0.9, 0.1, 0.9]
    assert reloaded.misses == 0  # a new process reuses the persisted scores


def test_risk_coverage_rewards_a_calibrated_checker() -> None:
    items = [_item(i, "none" if i % 2 else "near_paragraph", bool(i % 2 == 0)) for i in range(10)]
    # the checker is right on the 5 most confident items and wrong on the 5 least confident
    decisions = [baselines.Decision(i.id, flagged=(i.should_flag if int(i.id[1:]) < 5 else not i.should_flag)) for i in items]
    confidence = {i.id: 1.0 - int(i.id[1:]) / 10 for i in items}
    rc = risk_coverage(items, decisions, confidence, levels=(0.5, 1.0))
    assert rc["accuracy"] == [1.0, 0.5] and rc["n"] == 10
    flat = risk_coverage(items, decisions, {i.id: 0.5 for i in items}, levels=(0.5, 1.0))
    assert rc["aurc"] < flat["aurc"] or flat["aurc"] == rc["aurc"]
    assert risk_coverage([], [], {})["aurc"] is None


def test_evaluate_reports_the_new_types() -> None:
    items = [_item(0, "none", False), _item(1, "none", False), _item(2, "near_paragraph", True), _item(3, "overbroad_holding", True)]
    decisions = [baselines.Decision("i0", False), baselines.Decision("i1", True), baselines.Decision("i2", True), baselines.Decision("i3", False)]
    res = evaluate(items, decisions, n_boot=50)
    assert res["per_type"]["near_paragraph"]["recall"]["value"] == 1.0
    assert res["per_type"]["overbroad_holding"]["recall"]["value"] == 0.0
    assert res["per_type"]["near_paragraph"]["precision"]["value"] == 0.5  # one control also flagged
