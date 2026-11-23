"""Request and response models.

Pydantic is doing real work here: it is the input guardrail. Every query is
length-checked and every knob is range-checked before it reaches the engine,
so a malformed or hostile request is rejected with a 422 rather than being
passed down into NumPy.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from icd10_search.config import DEFAULT_TOP_K, MAX_QUERY_CHARS, MAX_TOP_K


class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS, description="Free-text query")
    k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="Number of results")
    mode: Literal["hybrid", "dense", "lexical"] = "hybrid"
    billable_only: bool = False
    chapter: int | None = Field(None, ge=1, le=22)


class SearchResultModel(BaseModel):
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
    score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    matched_terms: list[str] = []
    match_reason: str = "fused"


class SearchResponseModel(BaseModel):
    query: str
    mode: str
    results: list[SearchResultModel]
    total_candidates: int
    timings_ms: dict[str, float]
    filters: dict[str, object]
    disclaimer: str


class CodeDetailModel(BaseModel):
    code: str
    code_dotted: str
    description: str
    short_description: str
    billable: bool
    chapter: int
    chapter_title: str
    block: str
    block_title: str
    parent_code: str | None
    parent_description: str | None
    synonyms: list[str]
    inherited_synonyms: list[str]


class HealthModel(BaseModel):
    status: Literal["ok", "loading", "error"]
    version: str
    index_ready: bool
    detail: str | None = None


class ErrorModel(BaseModel):
    error: str
    detail: str
