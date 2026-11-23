"""FastAPI application.

Design notes
------------
* The index and the embedding model are loaded once during the lifespan
  startup and shared by every request. Loading them per request would add
  seconds of latency and gigabytes of churn.
* ``/api/health`` reports readiness separately from liveness, so a platform
  health check does not kill the container while the model is still loading.
* A small in-process rate limiter protects the public demo. It is per-process
  and therefore only correct for a single instance - which is exactly what the
  free tier runs. That limitation is stated rather than hidden.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from icd10_search import __version__
from icd10_search.api.schemas import (
    CodeDetailModel,
    HealthModel,
    SearchResponseModel,
)
from icd10_search.config import (
    DEFAULT_TOP_K,
    DISCLAIMER,
    EVAL_REPORT_PATH,
    MAX_QUERY_CHARS,
    MAX_TOP_K,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    WEB_DIR,
)
from icd10_search.search.service import SearchService

logger = logging.getLogger("icd10_search.api")

EXAMPLE_QUERIES = [
    "ringing in the ears",
    "nosebleed",
    "high blood pressure",
    "type 2 diabetes with no complications",
    "stage 3 chronic kidney disease",
    "shortness of breath on exertion",
    "loss of smell",
    "E11.9",
]

STATE: dict[str, object] = {"service": None, "error": None, "loaded_at": None}


class RateLimiter:
    """Fixed-window-per-client request counter, kept in memory."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        hits = self._hits[client]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True


rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


def warm_up(service: SearchService) -> None:
    """Run one throwaway query so the first real visitor does not pay for it.

    The embedding model is loaded lazily and ONNX Runtime builds its execution
    plan on first use. On a free-tier instance that sleeps, that cost would
    otherwise land on whoever clicks the link after an idle period.
    """
    try:
        started = time.perf_counter()
        service.search("warm up the embedding model", k=1)
        logger.info("warm-up query took %.0fms", (time.perf_counter() - started) * 1000)
    except Exception:  # pragma: no cover - warm-up must never block startup
        logger.exception("warm-up query failed; serving anyway")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    started = time.perf_counter()
    try:
        logger.info("loading search service ...")
        service = SearchService.load()
        warm_up(service)
        STATE["service"] = service
        STATE["loaded_at"] = time.time()
        logger.info("search service ready in %.1fs", time.perf_counter() - started)
    except Exception as exc:  # pragma: no cover - startup failure path
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("failed to load the search service")
    yield
    STATE["service"] = None


app = FastAPI(
    title="ICD-10-CM Hybrid Code Finder",
    version=__version__,
    description=(
        "Hybrid semantic + lexical search over the public CMS ICD-10-CM 2026 code set. "
        + DISCLAIMER
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):  # noqa: ANN201
    if request.url.path.startswith("/api/"):
        client = request.client.host if request.client else "unknown"
        if not rate_limiter.allow(client):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "detail": (
                        f"Limit is {RATE_LIMIT_REQUESTS} requests per "
                        f"{RATE_LIMIT_WINDOW_SECONDS}s. Please slow down."
                    ),
                },
            )
    return await call_next(request)


def require_service() -> SearchService:
    service = STATE.get("service")
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=STATE.get("error") or "search index is still loading, try again shortly",
        )
    return service  # type: ignore[return-value]


@app.get("/api/health", response_model=HealthModel, tags=["meta"])
def health() -> HealthModel:
    if STATE.get("error"):
        return HealthModel(
            status="error", version=__version__, index_ready=False, detail=str(STATE["error"])
        )
    ready = STATE.get("service") is not None
    return HealthModel(
        status="ok" if ready else "loading",
        version=__version__,
        index_ready=ready,
        detail=None if ready else "index loading",
    )


@app.get("/api/search", response_model=SearchResponseModel, tags=["search"])
def search(
    q: str = Query(..., min_length=1, max_length=MAX_QUERY_CHARS, description="Free-text query"),
    k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
    mode: Literal["hybrid", "dense", "lexical"] = "hybrid",
    billable_only: bool = False,
    chapter: int | None = Query(None, ge=1, le=22),
) -> SearchResponseModel:
    service = require_service()
    try:
        response = service.search(q, k=k, mode=mode, billable_only=billable_only, chapter=chapter)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SearchResponseModel(**response.to_dict(), disclaimer=DISCLAIMER)


@app.get("/api/code/{code}", response_model=CodeDetailModel, tags=["search"])
def get_code(code: str) -> CodeDetailModel:
    service = require_service()
    record = service.get_code(code)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ICD-10-CM code '{code}' not found")
    return CodeDetailModel(
        code=record.code,
        code_dotted=record.code_dotted,
        description=record.long_desc,
        short_description=record.short_desc,
        billable=record.billable,
        chapter=record.chapter,
        chapter_title=record.chapter_title,
        block=record.block,
        block_title=record.block_title,
        parent_code=record.parent_code,
        parent_description=record.parent_desc,
        synonyms=list(record.inclusion_terms),
        inherited_synonyms=list(record.inherited_terms),
    )


@app.get("/api/stats", tags=["meta"])
def stats() -> dict:
    service = require_service()
    return {**service.stats(), "disclaimer": DISCLAIMER}


@app.get("/api/chapters", tags=["meta"])
def chapters() -> list[dict]:
    return require_service().chapters()


@app.get("/api/examples", tags=["meta"])
def examples() -> dict:
    return {"examples": EXAMPLE_QUERIES}


@app.get("/api/evaluation", tags=["meta"])
def evaluation() -> dict:
    """Serve the measured retrieval quality report produced by `make eval`."""
    if not EVAL_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="no evaluation report has been generated yet")
    return json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))


if WEB_DIR.exists():

    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
