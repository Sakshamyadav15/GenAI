"""End-to-end latency of the legal instance's pipeline with a FAKE LLM and the REAL local parts.

    python -m scripts.legal_latency

Real: local embedding model, ChromaDB search over the indexed Acts, evidence windows, the fast NLI model.
Fake: the chat model (instant, scripted answer = first sentence of the top passage, cited). The numbers are
therefore the pipeline's own cost; add the LLM provider's response time (about 1 to 3 s per call) for real use.
No API call is made. The index is copied to a temporary directory first.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import statistics
import tempfile
from pathlib import Path

from langchain_core.messages import BaseMessage

from app.core.config import FAST_NLI_MODEL, BACKEND_DIR, PipelineConfig, Settings
from app.core.container import build_pipelines
from app.embeddings.local import LocalEmbeddings
from app.schemas.query import PipelineName
from app.vectorstore.store import ChromaVectorStore
from app.verification.nli import NLIRegistry
from tests.fakes import ScriptedChatModel

QUESTIONS = [
    "Which article of the Constitution abolishes untouchability?",
    "What does Article 14 say about equality before the law?",
    "What is the right to freedom of religion under the Constitution?",
    "Who can appeal against a refusal to give information under the Right to Information Act?",
    "What does the Constitution say about the protection of life and personal liberty?",
    "Which article deals with the abolition of titles?",
]


def respond(messages: list[BaseMessage]) -> str:
    prompt = str(messages[-1].content)
    match = re.search(r"\[1\] \(source: [^)]*\)\n([^\n]+)", prompt)
    body = match.group(1) if match else "No passage."
    sentence = re.split(r"(?<=[.])\s+", body.split("\n", 1)[0], maxsplit=1)[-1][:260].rstrip(".")
    return f"{sentence} [1]."


async def main() -> None:
    source = BACKEND_DIR / "data_legal" / "chroma"
    tmp = Path(tempfile.mkdtemp()) / "chroma"
    shutil.copytree(source, tmp)
    settings = Settings(embedding_provider="local", chat_model="fake-chat")
    embeddings = LocalEmbeddings(settings.local_embedding_model)
    nli = NLIRegistry()
    store = ChromaVectorStore(tmp)
    pipelines = build_pipelines(
        settings=settings, store=store, chat_model=ScriptedChatModel(responder=respond), embeddings=embeddings, nli=nli
    )
    config = PipelineConfig().with_overrides({"preset": "budget"})
    print(f"NLI model: {config.verification.nli_model} (fast: {config.verification.nli_model == FAST_NLI_MODEL})")
    embeddings.warm_up()
    nli.get(config.verification.nli_model).load()  # type: ignore[attr-defined]
    await pipelines[PipelineName.PROPOSED].run(QUESTIONS[0], "indian_acts", config)  # discard: first-call effects

    rows = []
    for q in QUESTIONS:
        r = await pipelines[PipelineName.PROPOSED].run(q, "indian_acts", config)
        rows.append(r.timings_ms | {"claims": len(r.verification.claims) if r.verification else 0})
        print(f"{r.total_latency_ms:7.0f} ms total | retrieval {r.timings_ms.get('retrieval', 0):5.0f} | "
              f"verification {r.timings_ms.get('verification', 0):5.0f} | claims {rows[-1]['claims']} | {q[:60]}")
    totals = [row["total"] for row in rows]
    print(f"\nPipeline only (fake LLM, {len(rows)} questions): mean {statistics.mean(totals):.0f} ms, median "
          f"{statistics.median(totals):.0f} ms, max {max(totals):.0f} ms")
    for stage in ("retrieval", "generation", "claim_extraction", "verification"):
        values = [row.get(stage, 0.0) for row in rows]
        print(f"  {stage:17s} mean {statistics.mean(values):6.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
