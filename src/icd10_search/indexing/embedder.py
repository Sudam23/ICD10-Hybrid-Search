"""Thin wrapper around FastEmbed.

FastEmbed runs the sentence-transformer through ONNX Runtime on CPU, which is
what makes a 98k-document index buildable on a laptop and servable inside a
512 MB free-tier container - no PyTorch, no GPU.

Two details matter for quality:

* bge models are trained asymmetrically. Queries must carry the instruction
  prefix "Represent this sentence for searching relevant passages: " and
  documents must not. Getting this backwards silently costs several points of
  recall, so the prefix lives in config and is applied in exactly one place.
* The model is loaded lazily and shared, because loading it costs ~2 s and the
  API process must not pay that twice.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence

import numpy as np

from icd10_search.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, query_prefix


class Embedder:
    """Encodes text into L2-normalised float32 vectors."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, *, threads: int | None = None) -> None:
        self.model_name = model_name
        self._threads = threads
        self._model = None

    @property
    def model(self):  # noqa: ANN201 - FastEmbed does not export a stable type
        if self._model is None:
            from fastembed import TextEmbedding

            kwargs = {"model_name": self.model_name}
            if self._threads is not None:
                kwargs["threads"] = self._threads
            self._model = TextEmbedding(**kwargs)
        return self._model

    def warm_up(self) -> None:
        """Force model load and run one tiny inference, so the first real request is fast."""
        self.embed_queries(["warm up"])

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        progress_every: int = 10_000,
    ) -> np.ndarray:
        """Embed the document side (no instruction prefix)."""
        return self._embed(texts, batch_size=batch_size, progress_every=progress_every)

    def embed_queries(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        """Embed the query side, applying the model's instruction prefix."""
        prefix = query_prefix(self.model_name)
        prepared = [prefix + text for text in texts]
        return self._embed(prepared, batch_size=batch_size, progress_every=0)

    def _embed(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        progress_every: int,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        started = time.perf_counter()
        vectors: list[np.ndarray] = []
        stream: Iterable[np.ndarray] = self.model.embed(list(texts), batch_size=batch_size)
        for position, vector in enumerate(stream, start=1):
            vectors.append(vector)
            if progress_every and position % progress_every == 0:
                rate = position / (time.perf_counter() - started)
                print(f"[embed] {position:,}/{len(texts):,} ({rate:,.0f} docs/s)", flush=True)

        matrix = np.asarray(vectors, dtype=np.float32)
        return l2_normalise(matrix)


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length so a dot product is a cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32)
