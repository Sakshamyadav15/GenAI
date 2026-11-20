"""Embeddings computed on this machine with sentence-transformers: no API key, no cost."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# BGE retrieval models expect this prefix on *queries* (not on documents).
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str, *, batch_size: int = 32, device: str = "cpu", model=None) -> None:  # type: ignore[no-untyped-def]
        self.model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._model = model  # injectable for tests; loaded lazily otherwise
        self._lock = threading.Lock()

    def _load(self):  # type: ignore[no-untyped-def]
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self._device)
                logger.info("Loaded local embedding model %s on %s", self.model_name, self._device)
            return self._model

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            list(texts), batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def warm_up(self) -> None:
        """Loads the model and runs one encode so the first real query is not slow."""
        self._encode(["warm up"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        prefix = _BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self._encode([prefix + text])[0]
