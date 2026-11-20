"""Section parser, statute labels, local embeddings wrapper, cost-free accounting, presets and the budget guard."""

from __future__ import annotations

import numpy as np
import pytest
from langchain_core.messages import AIMessage

from app.core.config import PipelineConfig, Settings
from app.core.errors import InvalidConfigError
from app.core.usage import UsageTracker
from app.embeddings.local import LocalEmbeddings
from app.legal_acts.parser import chunk_sections, clean_pdf_text, parse_sections
from app.schemas.documents import Chunk
from app.workflows.budget import regeneration_cost, verification_cost, would_exceed

SYNTHETIC_ACT = """THE EXAMPLE ACT, 2020
ARRANGEMENT OF SECTIONS
CHAPTER I
PRELIMINARY
1. Short title and commencement.
2. Definitions.
CHAPTER II
OFFENCES
3. Theft.
4. Punishment for theft.
THE EXAMPLE ACT, 2020
An Act to provide for examples.
CHAPTER I
PRELIMINARY
1. Short title and commencement.—(1) This Act may be called the Example Act, 2020. (2) It extends to the whole of India.
2. Definitions.—In this Act, unless the context otherwise requires,—
(a) "property" means movable property;
(b) "owner" means the person entitled to the property.
CHAPTER II
OFFENCES
3. Theft.—Whoever intends to take dishonestly any movable property out of the possession of any person
without that person's consent, commits theft.
1. This is a numbered list item inside section three and must not start a section.
4. Punishment for theft.—Whoever commits theft shall be punished with imprisonment of either description
for a term which may extend to three years, or with fine, or with both.
5A. Inserted provision.—Section 5A text.
6. Final section.—The end.
"""


def test_parser_ignores_arrangement_table_and_list_items() -> None:
    sections, report = parse_sections(SYNTHETIC_ACT)
    assert [s.number for s in sections] == ["1", "2", "3", "4", "5A", "6"]
    assert report.sections == 6 and report.first == "1" and report.last == "6"
    theft = next(s for s in sections if s.number == "3")
    assert "must not start a section" in theft.text  # the list item stayed inside section 3
    assert theft.title == "Theft"


def test_parser_tracks_chapters_and_reports_gaps() -> None:
    sections, report = parse_sections(SYNTHETIC_ACT)
    assert sections[0].chapter and sections[0].chapter.startswith("Chapter I")
    assert next(s for s in sections if s.number == "4").chapter.startswith("Chapter II")
    assert report.gaps == [] and report.short_sections == 2  # the two one-line sections (5A and 6)
    _, gappy = parse_sections("1. One.—Text of one is here and is long enough to count.\n4. Four.—Text of four is also long enough here.")
    assert gappy.gaps == ["2", "3"] and gappy.warnings


def test_two_line_headings_and_mid_line_starts() -> None:
    text = (
        "14. Equality before law.—The State shall not deny equality before the law.\n"
        "15. Prohibition of discrimination on grounds of religion, race, caste,\n"
        "sex or place of birth.—(1) The State shall not discriminate against any citizen.\n"
        "16. Equality of opportunity.—There shall be equality of opportunity."
    )
    sections, _ = parse_sections(text)
    assert [s.number for s in sections] == ["14", "15", "16"]
    assert sections[1].title == "Prohibition of discrimination on grounds of religion, race, caste, sex or place of birth"


def test_double_hyphen_after_the_heading_is_accepted() -> None:
    text = "19. Appeal.--(1) Any person aggrieved may appeal.\n20. Penalties.--(1) Where the Commission finds a default, it may impose a penalty."
    sections, _ = parse_sections(text)
    assert [s.number for s in sections] == ["19", "20"] and sections[1].title == "Penalties"


def test_clean_pdf_text_drops_legacy_font_lines_and_page_numbers() -> None:
    raw = "1. Title.—Text here\n£ÉÉ®iÉ BÉEÉ ºÉÆÉÊ´ÉvÉÉxÉ\n42\nMore text"
    cleaned = clean_pdf_text(raw)
    assert "£ÉÉ" not in cleaned and "\n42\n" not in cleaned and "More text" in cleaned


def test_chunks_carry_section_metadata_and_label() -> None:
    sections, _ = parse_sections(SYNTHETIC_ACT)
    chunks = chunk_sections(sections, act="Example Act, 2020", doc_id="d1", corpus="indian_acts", filename="example.pdf")
    theft = next(c for c in chunks if c.section == "3")
    assert theft.section_label == "Example Act, 2020, Section 3"
    assert theft.text.startswith("Example Act, 2020, Section 3. Theft\n")
    assert theft.chapter and theft.act == "Example Act, 2020"
    # round trip through the vector-store metadata keeps the label
    restored = Chunk.from_vector_record(theft.chunk_id, theft.text, theft.vector_metadata())
    assert restored.section_label == theft.section_label and restored.section_title == "Theft"


def test_constitution_chunks_are_labelled_as_articles() -> None:
    sections, _ = parse_sections("14. Equality before law.—The State shall not deny to any person equality before the law.")
    chunk = chunk_sections(sections, act="Constitution of India", doc_id="d", corpus="c", filename="c.pdf")[0]
    assert chunk.section_label == "Constitution of India, Article 14"


def test_long_sections_are_split_with_header_on_every_chunk() -> None:
    body = " ".join(f"({i}) Clause number {i} says something sufficiently long to fill space in the section." for i in range(1, 40))
    sections, _ = parse_sections(f"9. Long section.—{body}\n10. Next.—Short.")
    chunks = chunk_sections(sections, act="Example Act", doc_id="d", corpus="c", filename="f.pdf", max_chars=500)
    long_chunks = [c for c in chunks if c.section == "9"]
    assert len(long_chunks) > 3
    assert all(len(c.text) <= 520 and c.text.startswith("Example Act, Section 9. Long section\n") for c in long_chunks)


class _FakeSentenceModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):  # noqa: ANN001
        self.calls.append(list(texts))
        return np.array([[float(len(t)), 1.0, 0.0] for t in texts]) / 10


def test_local_embeddings_wrapper_batches_and_prefixes_queries_for_bge() -> None:
    model = _FakeSentenceModel()
    emb = LocalEmbeddings("BAAI/bge-small-en-v1.5", model=model)
    docs = emb.embed_documents(["a", "bb"])
    query = emb.embed_query("what")
    assert len(docs) == 2 and len(docs[0]) == 3
    assert model.calls[0] == ["a", "bb"]  # documents are embedded as they are
    assert model.calls[1][0].startswith("Represent this sentence for searching relevant passages: ")
    assert query is not None and emb.embed_documents([]) == []
    plain = _FakeSentenceModel()
    LocalEmbeddings("sentence-transformers/all-MiniLM-L6-v2", model=plain).embed_query("x")
    assert plain.calls[0] == ["x"]  # no prefix for other models


def test_local_models_cost_nothing_and_do_not_make_total_cost_unknown() -> None:
    tracker = UsageTracker(PipelineConfig().pricing)
    tracker.record_embedding("query_embedding", "local:BAAI/bge-small-en-v1.5", ["some query text"])
    tracker.record_chat("generation", "gpt-4o-mini", AIMessage(content="x", usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}), "p")
    summary = tracker.summary()
    assert summary.stages[0].cost_usd == 0.0
    assert summary.cost_usd == pytest.approx((1000 * 0.15 + 100 * 0.60) / 1e6)
    assert tracker.spent_usd() == pytest.approx(summary.cost_usd)


def test_settings_embedding_label() -> None:
    assert Settings(_env_file=None).embedding_label == "text-embedding-3-small"
    local = Settings(_env_file=None, embedding_provider="local", local_embedding_model="BAAI/bge-small-en-v1.5")
    assert local.embedding_label == "local:BAAI/bge-small-en-v1.5"


def test_budget_preset_and_overrides() -> None:
    base = PipelineConfig()
    assert "budget" not in base.model_dump()  # unchanged serialisation for configs without a ceiling
    budget = base.with_overrides({"preset": "budget"})
    assert budget.budget is not None and budget.budget.max_cost_usd == 0.001
    assert (budget.proposed.adaptive, budget.proposed.verify) == (False, True)
    assert budget.baseline.top_k == 4 and budget.generation.max_answer_tokens == 250
    assert (budget.verification.use_llm_judge, budget.verification.max_retries, budget.verification.max_claims) == (False, 0, 6)
    assert budget.with_overrides({"baseline": {"top_k": 3}}).baseline.top_k == 3
    assert base.with_overrides({"preset": "budget", "baseline": {"top_k": 6}}).baseline.top_k == 6
    with pytest.raises(InvalidConfigError):
        base.with_overrides({"preset": "nope"})
    assert PipelineConfig.model_validate(budget.model_dump()) == budget  # recorded configs round-trip


def test_budget_estimates_fit_the_ceiling_for_a_typical_question() -> None:
    config = PipelineConfig().with_overrides({"preset": "budget"})
    tracker = UsageTracker(config.pricing)
    answer = "The Public Information Officer shall reply within thirty days of the receipt of the request [1]. " * 2
    extraction = verification_cost(config, "gpt-4o-mini", answer)
    assert 0 < extraction < 0.0004
    assert not would_exceed(config, tracker, extraction)
    assert would_exceed(config, tracker, 0.002)  # a step costing more than the whole ceiling is refused
    tracker.record_chat("generation", "gpt-4o-mini", AIMessage(content=answer, usage_metadata={"input_tokens": 1100, "output_tokens": 250, "total_tokens": 1350}), "p")
    retry = regeneration_cost(config, "gpt-4o-mini", tracker, answer)
    # After one generation (USD 0.000315) a retry (about USD 0.00043) still fits under USD 0.001 ...
    assert 0.0003 < retry < 0.0005 and not would_exceed(config, tracker, retry)
    # ... but not once the verification call and a second generation have been paid for.
    tracker.record_chat("claim_extraction", "gpt-4o-mini", AIMessage(content="x", usage_metadata={"input_tokens": 400, "output_tokens": 150, "total_tokens": 550}), "p")
    tracker.record_chat("generation", "gpt-4o-mini", AIMessage(content=answer, usage_metadata={"input_tokens": 1300, "output_tokens": 250, "total_tokens": 1550}), "p")
    assert would_exceed(config, tracker, regeneration_cost(config, "gpt-4o-mini", tracker, answer))
    unpriced = PipelineConfig().with_overrides({"budget": {"max_cost_usd": 0.001}})
    assert verification_cost(unpriced, "unknown-model", answer) == 0.0  # unpriced models cannot be estimated
