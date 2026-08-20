import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated, cast

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.config import Settings, get_settings, validate_settings
from app.core.logging import configure_logging
from app.core.metrics import increment, observe_query, render_prometheus
from app.core.middleware import RateLimitMiddleware, RequestIdMiddleware, RequestSizeLimitMiddleware
from app.core.schemas import (
    Citation,
    ErrorResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievalMetadata,
)
from app.services.answering import answer_from_retrieval
from app.services.ingest import ingest_document, normalize_source_name
from app.services.retrieval import RetrievalStrategy, retrieve
from app.services.storage import BaseStore, get_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    validate_settings(settings)
    store = get_store(settings)
    await run_in_threadpool(store.ensure_collection)
    app.state.settings = settings
    app.state.store = store
    try:
        yield
    finally:
        await run_in_threadpool(store.close)


app = FastAPI(
    title="RAG API Eval Starter",
    version="0.2.0",
    description="Explainable dense + lexical retrieval with deterministic quality gates.",
    lifespan=lifespan,
)

# Starlette executes the last-added middleware first. Request IDs therefore wrap
# rate/size-limit responses as well as successful endpoint responses.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(RequestIdMiddleware)


def _store(request: Request) -> BaseStore:
    return cast(BaseStore, request.app.state.store)


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def require_ingest_access(request: Request) -> None:
    expected = _settings(request).ingest_api_key
    if expected is None:
        return
    supplied = request.headers.get("x-api-key", "")
    if not secrets.compare_digest(
        supplied.encode("utf-8"), expected.get_secret_value().encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="A valid X-API-Key is required")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    code = "request_too_large" if exc.status_code == 413 else "http_error"
    payload = ErrorResponse(code=code, message=str(exc.detail), request_id=request_id)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception", exc_info=exc)
    increment("errors")
    request_id = getattr(request.state, "request_id", None)
    payload = ErrorResponse(
        code="internal_error",
        message="Internal server error",
        request_id=request_id,
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_ingest_access)])
async def ingest(request: Request, file: Annotated[UploadFile, File(...)]) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    settings = _settings(request)
    try:
        source_name = normalize_source_name(file.filename, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not source_name.lower().endswith((".txt", ".md", ".pdf")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    limit_bytes = settings.request_size_limit_mb * 1024 * 1024
    data = await file.read(limit_bytes + 1)
    if len(data) > limit_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Document exceeds the {settings.request_size_limit_mb} MB limit",
        )

    try:
        result = await run_in_threadpool(
            ingest_document,
            source_name,
            data,
            _store(request),
            settings,
        )
    except ValueError as exc:
        increment("errors")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    increment("ingest_requests")
    logger.info(
        "ingest_complete",
        extra={"doc_id": result["doc_id"], "chunks": result["chunks"]},
    )
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    retrieval = await run_in_threadpool(
        partial(
            retrieve,
            payload.question,
            _store(request),
            strategy=cast(RetrievalStrategy | None, payload.strategy),
            top_k=payload.top_k,
            settings=_settings(request),
        )
    )
    answer = answer_from_retrieval(retrieval, settings=_settings(request))
    latency_ms = (time.perf_counter() - started) * 1000.0

    observe_query(
        strategy=retrieval.strategy,
        latency_ms=latency_ms,
        abstained=answer.abstained,
        citations=len(answer.citations),
    )
    logger.info(
        "query_complete",
        extra={
            "strategy": retrieval.strategy,
            "top_k": len(retrieval.hits),
            "confidence": answer.confidence,
            "abstained": answer.abstained,
            "latency_ms": latency_ms,
        },
    )
    return QueryResponse(
        answer=answer.answer,
        citations=[Citation(**item) for item in answer.citations],
        retrieval=RetrievalMetadata(
            strategy=retrieval.strategy,
            confidence=answer.confidence,
            abstained=answer.abstained,
            dense_candidates=retrieval.dense_candidates,
            lexical_candidates=retrieval.lexical_candidates,
            latency_ms=latency_ms,
        ),
    )


@app.get("/health")
async def health(request: Request) -> dict:
    """Liveness check: the API process is running."""

    return {"status": "ok", "environment": _settings(request).environment}


@app.get("/ready")
async def ready(request: Request) -> dict:
    """Readiness check: the configured retrieval store is reachable."""

    if not await run_in_threadpool(_store(request).is_ready):
        raise HTTPException(status_code=503, detail="Retrieval store is unavailable")
    return {"status": "ready", "backend": _settings(request).vector_backend}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return render_prometheus()
