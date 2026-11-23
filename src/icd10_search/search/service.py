"""The search service: one object that owns the corpus, both indexes and the model.

Pipeline for a query
--------------------
1. Normalise and validate the query string.
2. Dense arm - embed the query (with the bge instruction prefix) and score it
   against every int8 document vector.
3. Lexical arm - BM25 score against the inverted index.
4. Filters - any active filter (billable only, chapter) is applied to the score
   vectors *before* the top-k cut, so a filtered search still returns a full
   page of results instead of a short one.
5. Fusion - take the top ``CANDIDATE_POOL`` from each arm and fuse by
   reciprocal rank.
6. Hydrate - attach the human-readable record and the per-arm evidence, so the
   UI can show why each hit was returned.
"""

from __future__ import annotations

import bisect
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from icd10_search.config import (
    CANDIDATE_POOL,
    CODE_LOOKUP_LIMIT,
    DENSE_SCALES_PATH,
    DENSE_VECTORS_PATH,
    EMBEDDING_MODEL,
    INDEX_DIR,
    INDEX_MANIFEST_PATH,
    LEXICAL_INDEX_PATH,
    MAX_QUERY_CHARS,
    MAX_TOP_K,
    RRF_K,
    RRF_WEIGHT_DENSE,
    RRF_WEIGHT_LEXICAL,
)
from icd10_search.data.parse import CodeRecord, load_corpus, undot_code
from icd10_search.data.text import lexical_text
from icd10_search.indexing.embedder import Embedder
from icd10_search.indexing.quantize import top_k
from icd10_search.search.dense import DenseIndex
from icd10_search.search.fusion import RankedList, reciprocal_rank_fusion
from icd10_search.search.lexical import LexicalIndex, tokenize

SEARCH_MODES = ("hybrid", "dense", "lexical")

CODE_QUERY_RE = re.compile(r"^[A-Za-z][0-9][0-9A-Za-z]\.?[0-9A-Za-z]{0,4}$")


class IndexMismatchError(RuntimeError):
    """Raised when the built index does not match the corpus or model in use."""


@dataclass(slots=True)
class SearchResult:
    rank: int
    code: str
    code_dotted: str
    description: str
    short_description: str
    billable: bool
    chapter: int
    chapter_title: str
    block: str
    block_title: str
    synonyms: list[str]
    score: float | None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    matched_terms: list[str] = field(default_factory=list)
    match_reason: str = "fused"


@dataclass(slots=True)
class SearchResponse:
    query: str
    mode: str
    results: list[SearchResult]
    total_candidates: int
    timings_ms: dict[str, float]
    filters: dict[str, object]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "results": [asdict(result) for result in self.results],
            "total_candidates": self.total_candidates,
            "timings_ms": self.timings_ms,
            "filters": self.filters,
        }


class SearchService:
    def __init__(
        self,
        records: list[CodeRecord],
        dense: DenseIndex,
        lexical: LexicalIndex,
        embedder: Embedder,
        manifest: dict,
    ) -> None:
        self.records = records
        self.dense = dense
        self.lexical = lexical
        self.embedder = embedder
        self.manifest = manifest

        self._by_code = {record.code: position for position, record in enumerate(records)}
        self._sorted_codes = sorted(self._by_code)
        self._billable = np.array([record.billable for record in records], dtype=bool)
        self._chapters = np.array([record.chapter for record in records], dtype=np.int16)

    @classmethod
    def load(
        cls,
        index_dir: Path = INDEX_DIR,
        *,
        model_name: str = EMBEDDING_MODEL,
        warm_up: bool = True,
    ) -> SearchService:
        manifest_path = index_dir / INDEX_MANIFEST_PATH.name
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no index manifest at {manifest_path} - run `make index` first"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        records = load_corpus()
        dense = DenseIndex.load(
            index_dir / DENSE_VECTORS_PATH.name, index_dir / DENSE_SCALES_PATH.name
        )
        lexical = LexicalIndex.load(index_dir / LEXICAL_INDEX_PATH.name)

        if not (len(records) == dense.num_docs == lexical.num_docs):
            raise IndexMismatchError(
                "corpus and index disagree on document count: "
                f"corpus={len(records):,}, dense={dense.num_docs:,}, bm25={lexical.num_docs:,}. "
                "Rebuild with `make index`."
            )
        if manifest.get("embedding_model") != model_name:
            raise IndexMismatchError(
                f"index was built with '{manifest.get('embedding_model')}' but the service is "
                f"configured for '{model_name}'. Rebuild the index or set ICD10_EMBEDDING_MODEL."
            )

        embedder = Embedder(model_name)
        if warm_up:
            embedder.warm_up()
        return cls(records, dense, lexical, embedder, manifest)

    def _mask(self, billable_only: bool, chapter: int | None) -> np.ndarray | None:
        mask: np.ndarray | None = None
        if billable_only:
            mask = self._billable
        if chapter is not None:
            chapter_mask = self._chapters == chapter
            mask = chapter_mask if mask is None else (mask & chapter_mask)
        return mask

    @staticmethod
    def _apply_mask(scores: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            return scores
        filtered = scores.copy()
        filtered[~mask] = -np.inf
        return filtered

    def normalise_query(self, query: str) -> str:
        cleaned = " ".join((query or "").split())
        if not cleaned:
            raise ValueError("query must not be empty")
        if len(cleaned) > MAX_QUERY_CHARS:
            raise ValueError(f"query must be at most {MAX_QUERY_CHARS} characters")
        return cleaned

    def code_candidates(self, query: str, limit: int = CODE_LOOKUP_LIMIT) -> list[int]:
        """Document positions for a query that is itself a code or a code prefix.

        ``E119`` and ``E11.9`` both resolve to E11.9; ``E11`` additionally
        returns the codes beneath it, which is how a coder drills down from a
        category to a billable child. A fully specified code such as E11.9 has
        nothing beneath it, so its siblings are offered instead: those are the
        alternatives the coder is actually choosing between, and they are far
        more useful than whatever a code string happens to embed near.
        """
        candidate = query.strip()
        if not CODE_QUERY_RE.match(candidate):
            return []
        token = undot_code(candidate)
        if len(token) < 3:
            return []

        positions: list[int] = []
        exact = self._by_code.get(token)
        if exact is not None:
            positions.append(exact)

        self._extend_with_prefix(positions, token, limit)
        if exact is not None:
            for prefix in (self.records[exact].parent_code, token[:3], token[:2]):
                if len(positions) >= limit:
                    break
                if prefix and len(prefix) < len(token):
                    self._extend_with_prefix(positions, prefix, limit)
        return positions

    def _extend_with_prefix(self, positions: list[int], prefix: str, limit: int) -> None:
        """Append codes beginning with ``prefix``, in code order, up to ``limit``."""
        start = bisect.bisect_left(self._sorted_codes, prefix)
        for code in self._sorted_codes[start:]:
            if not code.startswith(prefix):
                break
            position = self._by_code[code]
            if position not in positions:
                positions.append(position)
            if len(positions) >= limit:
                break

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: str = "hybrid",
        billable_only: bool = False,
        chapter: int | None = None,
        candidate_pool: int = CANDIDATE_POOL,
        weight_dense: float = RRF_WEIGHT_DENSE,
        weight_lexical: float = RRF_WEIGHT_LEXICAL,
        code_lookup: bool = True,
    ) -> SearchResponse:
        if mode not in SEARCH_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {SEARCH_MODES}")
        query = self.normalise_query(query)
        k = max(1, min(k, MAX_TOP_K))
        mask = self._mask(billable_only, chapter)
        pool = max(k, candidate_pool)

        pinned: list[int] = []
        if code_lookup:
            pinned = [
                position
                for position in self.code_candidates(query)
                if mask is None or bool(mask[position])
            ][:k]

        timings: dict[str, float] = {}
        ranked_lists: list[RankedList] = []
        dense_scores: np.ndarray | None = None
        lexical_scores: np.ndarray | None = None

        if mode in ("hybrid", "dense"):
            started = time.perf_counter()
            query_vector = self.embedder.embed_queries([query])[0]
            timings["embed"] = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            dense_scores = self._apply_mask(self.dense.score(query_vector), mask)
            doc_ids, scores = top_k(dense_scores, pool)
            timings["dense"] = (time.perf_counter() - started) * 1000
            keep = np.isfinite(scores)
            ranked_lists.append(
                RankedList("dense", doc_ids[keep], scores[keep], weight=weight_dense)
            )

        if mode in ("hybrid", "lexical"):
            started = time.perf_counter()
            lexical_scores = self._apply_mask(self.lexical.score(query), mask)
            doc_ids, scores = top_k(lexical_scores, pool)
            timings["lexical"] = (time.perf_counter() - started) * 1000
            keep = np.isfinite(scores) & (scores > 0)
            ranked_lists.append(
                RankedList("lexical", doc_ids[keep], scores[keep], weight=weight_lexical)
            )

        started = time.perf_counter()
        if mode == "hybrid":
            fused = reciprocal_rank_fusion(ranked_lists, k=RRF_K, top_n=k)
        else:
            only = ranked_lists[0]
            fused = [
                (int(doc_id), float(score), {only.name: rank})
                for rank, (doc_id, score) in enumerate(
                    zip(only.doc_ids[:k], only.scores[:k], strict=True), start=1
                )
            ]
        if pinned:
            pinned_set = set(pinned)
            fused = [
                (position, float("inf"), {"code": rank})
                for rank, position in enumerate(pinned, start=1)
            ] + [entry for entry in fused if entry[0] not in pinned_set]
            fused = fused[:k]
        timings["fuse"] = (time.perf_counter() - started) * 1000
        timings["total"] = round(sum(timings.values()), 2)
        timings = {name: round(value, 2) for name, value in timings.items()}

        query_tokens = set(tokenize(query))
        results: list[SearchResult] = []
        for position, (doc_id, score, provenance) in enumerate(fused, start=1):
            record = self.records[doc_id]
            is_pinned = "code" in provenance
            if is_pinned:
                token = undot_code(query)
                if record.code == token:
                    reason = "exact_code"
                elif record.code.startswith(token):
                    reason = "code_prefix"
                else:
                    reason = "code_sibling"
            else:
                reason = mode
            document_tokens = set(tokenize(lexical_text(record)))
            results.append(
                SearchResult(
                    rank=position,
                    code=record.code,
                    code_dotted=record.code_dotted,
                    description=record.long_desc,
                    short_description=record.short_desc,
                    billable=record.billable,
                    chapter=record.chapter,
                    chapter_title=record.chapter_title,
                    block=record.block,
                    block_title=record.block_title,
                    synonyms=[*record.inclusion_terms, *record.inherited_terms][:5],
                    score=None if is_pinned else round(float(score), 6),
                    match_reason=reason,
                    dense_rank=provenance.get("dense"),
                    lexical_rank=provenance.get("lexical"),
                    dense_score=(
                        round(float(dense_scores[doc_id]), 4)
                        if dense_scores is not None and np.isfinite(dense_scores[doc_id])
                        else None
                    ),
                    lexical_score=(
                        round(float(lexical_scores[doc_id]), 4)
                        if lexical_scores is not None and np.isfinite(lexical_scores[doc_id])
                        else None
                    ),
                    matched_terms=sorted(query_tokens & document_tokens),
                )
            )

        return SearchResponse(
            query=query,
            mode=mode,
            results=results,
            total_candidates=len({int(d) for lst in ranked_lists for d in lst.doc_ids}),
            timings_ms=timings,
            filters={"billable_only": billable_only, "chapter": chapter},
        )

    def get_code(self, code: str) -> CodeRecord | None:
        position = self._by_code.get(undot_code(code))
        return self.records[position] if position is not None else None

    def chapters(self) -> list[dict[str, object]]:
        seen: dict[int, str] = {}
        counts: dict[int, int] = {}
        for record in self.records:
            if not record.chapter:
                continue
            seen.setdefault(record.chapter, record.chapter_title)
            counts[record.chapter] = counts.get(record.chapter, 0) + 1
        return [
            {"chapter": number, "title": seen[number], "codes": counts[number]}
            for number in sorted(seen)
        ]

    def stats(self) -> dict[str, object]:
        return {
            "total_codes": len(self.records),
            "billable_codes": int(self._billable.sum()),
            "chapters": len({r.chapter for r in self.records if r.chapter}),
            "embedding_model": self.manifest.get("embedding_model"),
            "embedding_dim": self.dense.dim,
            "doc_text_variant": self.manifest.get("doc_text_variant"),
            "quantization": self.manifest.get("quantization"),
            "bm25_vocabulary": self.manifest.get("bm25", {}).get("vocabulary_terms"),
            "index_built_at": self.manifest.get("built_at"),
            "index_bytes": self.manifest.get("artifact_bytes"),
        }
