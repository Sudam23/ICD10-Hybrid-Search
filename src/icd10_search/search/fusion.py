"""Combining two ranked lists.

Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009):

    RRF(d) = sum over retrievers of  weight / (k + rank(d))

Rank, not score, is what is fused. That matters here because the two
retrievers produce numbers on incompatible scales - a cosine similarity sits
in [-1, 1] while a BM25 score is unbounded and depends on corpus statistics.
Normalising those into a shared range needs per-query calibration that is
fragile; using rank sidesteps the problem entirely.

The constant k (60 by default) damps the influence of the very top ranks, so
one retriever being confidently wrong cannot dominate the fused list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from icd10_search.config import RRF_K


@dataclass(frozen=True)
class RankedList:
    """One retriever's output: document ids best-first, with their raw scores."""

    name: str
    doc_ids: np.ndarray
    scores: np.ndarray
    weight: float = 1.0

    def rank_of(self) -> dict[int, int]:
        """Map document id -> 1-based rank."""
        return {int(doc_id): rank for rank, doc_id in enumerate(self.doc_ids, start=1)}


def reciprocal_rank_fusion(
    lists: list[RankedList],
    *,
    k: int = RRF_K,
    top_n: int | None = None,
) -> list[tuple[int, float, dict[str, int]]]:
    """Fuse ranked lists.

    Returns ``(doc_id, fused_score, {retriever: rank})`` best-first. The
    per-retriever ranks are carried through so the UI can show *why* a result
    is where it is.
    """
    fused: dict[int, float] = {}
    provenance: dict[int, dict[str, int]] = {}

    for ranked in lists:
        for doc_id, rank in ranked.rank_of().items():
            fused[doc_id] = fused.get(doc_id, 0.0) + ranked.weight / (k + rank)
            provenance.setdefault(doc_id, {})[ranked.name] = rank

    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    if top_n is not None:
        ordered = ordered[:top_n]
    return [(doc_id, score, provenance[doc_id]) for doc_id, score in ordered]
