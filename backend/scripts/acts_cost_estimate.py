"""Offline estimate of the per-question cost of the Indian-Acts budget mode (no API calls).

    python -m scripts.acts_cost_estimate

Builds the real generation prompt from four real chunks of the downloaded RTI Act (or synthetic text when the
PDF is not present), counts tokens with tiktoken, and applies the price table. The worst case assumes the
model uses its whole answer budget.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import BUDGET_PRESET, PipelineConfig
from app.core.usage import estimate_tokens
from app.generation.prompts import build_generation_messages
from app.legal_acts.parser import chunk_sections, clean_pdf_text, parse_sections
from app.schemas.documents import Chunk
from app.verification.claim_extractor import EXTRACTION_PROMPT

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL = "gpt-4o-mini"
QUESTION = "Within what time must a Public Information Officer reply to a request under the Right to Information Act, 2005?"


def sample_chunks(k: int) -> list[Chunk]:
    pdf = BACKEND_DIR / "data_legal" / "raw" / "right_to_information_act_2005.pdf"
    if pdf.exists():
        from pypdf import PdfReader

        text = clean_pdf_text("\n".join((p.extract_text() or "") for p in PdfReader(str(pdf)).pages))
        sections, _ = parse_sections(text)
        chunks = chunk_sections(sections, act="Right to Information Act, 2005", doc_id="x", corpus="c", filename=pdf.name, max_chars=800)
        # the longest chunks are the worst case for prompt size
        return sorted(chunks, key=lambda c: -len(c.text))[:k]
    filler = "Every public authority shall designate Public Information Officers and shall provide information within thirty days. " * 8
    return [Chunk(chunk_id=f"x:{i}", doc_id="x", corpus="c", filename="f.pdf", chunk_index=i, text=filler[:790]) for i in range(k)]


def main() -> None:
    config = PipelineConfig().with_overrides({"preset": "budget"})
    price = config.pricing[MODEL]
    k = config.baseline.top_k
    messages = build_generation_messages(QUESTION, sample_chunks(k))
    gen_in = sum(estimate_tokens(str(m.content)) for m in messages) + 8  # + chat framing overhead
    gen_out = config.generation.max_answer_tokens
    answer_tokens = gen_out  # worst case: the whole answer budget is used
    ext_in = estimate_tokens(EXTRACTION_PROMPT) + answer_tokens + 40
    ext_out = min(config.verification.max_tokens, int(answer_tokens * 1.2) + 40)

    def cost(i: int, o: int) -> float:
        return (i * price.input_per_million + o * price.output_per_million) / 1e6

    generation, extraction = cost(gen_in, gen_out), cost(ext_in, ext_out)
    print(f"Preset: {BUDGET_PRESET}")
    print(f"Model {MODEL}: USD {price.input_per_million}/M input, USD {price.output_per_million}/M output")
    print(f"Embeddings: local model, USD 0. NLI claim check: local model, USD 0.")
    print(f"Generation  : {gen_in} in x {price.input_per_million}/1e6 + {gen_out} out x {price.output_per_million}/1e6 = USD {generation:.6f}")
    print(f"Extraction  : {ext_in} in x {price.input_per_million}/1e6 + {ext_out} out x {price.output_per_million}/1e6 = USD {extraction:.6f}")
    total = generation + extraction
    print(f"Worst case per question: USD {total:.6f}  (ceiling USD {config.budget.max_cost_usd})  ->  {'fits' if total <= config.budget.max_cost_usd else 'EXCEEDS'}")
    print(f"Typical (answer ~150 tokens): about USD {cost(gen_in, 150) + cost(estimate_tokens(EXTRACTION_PROMPT) + 190, 220):.6f}")


if __name__ == "__main__":
    main()
