"""The dense (embedding) half of the retriever."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from icd10_search.config import DENSE_SCALES_PATH, DENSE_VECTORS_PATH
from icd10_search.indexing.quantize import score_all, top_k


class DenseIndex:
    """Exact brute-force cosine search over int8-quantised document vectors.

    The matrix is memory-mapped: the container's resident memory stays low and
    the OS page cache keeps the hot pages around after the first few queries.
    """

    def __init__(self, vectors: np.ndarray, scales: np.ndarray) -> None:
        if vectors.shape[0] != scales.shape[0]:
            raise ValueError(
                f"vector/scale count mismatch: {vectors.shape[0]} vs {scales.shape[0]}"
            )
        self.vectors = vectors
        self.scales = scales

    @classmethod
    def load(
        cls,
        vectors_path: Path = DENSE_VECTORS_PATH,
        scales_path: Path = DENSE_SCALES_PATH,
    ) -> DenseIndex:
        if not vectors_path.exists():
            raise FileNotFoundError(f"dense index not found at {vectors_path} - run `make index`")
        return cls(
            vectors=np.load(vectors_path, mmap_mode="r"),
            scales=np.load(scales_path),
        )

    @property
    def num_docs(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def score(self, query_vector: np.ndarray) -> np.ndarray:
        return score_all(self.vectors, self.scales, query_vector)

    def search(self, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return top_k(self.score(query_vector), k)
