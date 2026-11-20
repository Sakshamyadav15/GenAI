"""Indexes the Acts found in backend/data_legal/raw into the 'indian_acts' corpus with LOCAL embeddings.

    python -m scripts.acts_index [--raw backend/data_legal/raw] [--reindex]

No API calls: embeddings are computed on this machine. Documents are registered in the application database
(so they appear in the UI and the demo-cache fingerprint covers them) and chunks go into ChromaDB under
DATA_DIR (default backend/data_legal). Prints the counts and any parse problems.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

BACKEND_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_DIR", str(BACKEND_DIR / "data_legal"))
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("DEFAULT_CORPUS", "indian_acts")

from app.core.config import Settings  # noqa: E402
from app.database.models import DocumentRow  # noqa: E402
from app.database.repositories import DocumentRepository  # noqa: E402
from app.database.session import create_engine, create_sessionmaker, init_db  # noqa: E402
from app.embeddings.factory import create_embeddings  # noqa: E402
from app.legal_acts.parser import chunk_sections, clean_pdf_text, parse_sections  # noqa: E402
from app.vectorstore.store import ChromaVectorStore  # noqa: E402

CATALOG = BACKEND_DIR / "seed" / "acts_catalog.json"
CORPUS = "indian_acts"
MAX_CHUNK_CHARS = 800


async def index(raw: Path, reindex: bool) -> int:
    settings = Settings()
    assert settings.database_url and settings.chroma_dir
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.database_url)
    await init_db(engine)
    sessionmaker = create_sessionmaker(engine)
    store = ChromaVectorStore(settings.chroma_dir)
    embeddings = create_embeddings(settings)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["acts"]

    totals = {"acts": 0, "sections": 0, "chunks": 0}
    missing: list[str] = []
    problems: list[str] = []
    for act in catalog:
        pdf = raw / f"{act['slug']}.pdf"
        if not pdf.exists():
            missing.append(act["name"])
            continue
        data = pdf.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        async with sessionmaker() as session:
            repo = DocumentRepository(session)
            existing = await repo.get_by_hash(CORPUS, sha)
            if existing is not None and not reindex:
                print(f"skip (already indexed) {act['name']}")
                continue
            if existing is not None:
                store.delete_document(CORPUS, existing.id)
                await repo.delete(existing.id)
                await session.commit()

        reader = PdfReader(str(pdf))
        text = clean_pdf_text("\n".join((page.extract_text() or "") for page in reader.pages))
        sections, report = parse_sections(text)
        doc_id = uuid4().hex
        chunks = chunk_sections(sections, act=act["name"], doc_id=doc_id, corpus=CORPUS, filename=pdf.name, max_chars=MAX_CHUNK_CHARS)
        print(f"{act['name']}: {len(reader.pages)} pages -> {report.sections} sections ({report.first}..{report.last}), "
              f"{len(chunks)} chunks, {len(report.gaps)} gaps, {report.short_sections} very short")
        for warning in report.warnings:
            problems.append(f"{act['name']}: {warning}")
        if not chunks:
            problems.append(f"{act['name']}: nothing indexed")
            continue

        vectors = await embeddings.aembed_documents([c.text for c in chunks])
        store.add_chunks(chunks, vectors)
        async with sessionmaker() as session:
            await DocumentRepository(session).add(
                DocumentRow(
                    id=doc_id, corpus=CORPUS, filename=f"{act['name']}", file_type="pdf", file_size=len(data), sha256=sha,
                    storage_path=str(pdf), status="ready", num_pages=len(reader.pages), num_chunks=len(chunks),
                    chunk_size=MAX_CHUNK_CHARS, chunk_overlap=0,
                )
            )
            await session.commit()
        totals["acts"] += 1
        totals["sections"] += report.sections
        totals["chunks"] += len(chunks)

    await engine.dispose()
    print(f"\nIndexed {totals['acts']} Acts, {totals['sections']} sections, {totals['chunks']} chunks into corpus '{CORPUS}'.")
    if missing:
        print(f"Not indexed (no PDF in {raw}): {len(missing)} Acts")
    for problem in problems:
        print("PARSE PROBLEM:", problem)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=BACKEND_DIR / "data_legal" / "raw")
    parser.add_argument("--reindex", action="store_true", help="Delete and rebuild Acts that are already indexed")
    args = parser.parse_args()
    return asyncio.run(index(args.raw, args.reindex))


if __name__ == "__main__":
    sys.exit(main())
