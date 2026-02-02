import logging

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.config import get_settings, validate_settings
from app.core.logging import configure_logging
from app.core.metrics import increment, render_prometheus
from app.core.middleware import RateLimitMiddleware, RequestIdMiddleware, RequestSizeLimitMiddleware
from app.core.schemas import ErrorResponse, IngestResponse, QueryRequest, QueryResponse
from app.services.ingest import ingest_document
from app.services.retrieval import hybrid_search
from app.services.storage import get_store

logger = logging.getLogger(__name__)
app = FastAPI(title="RAG API Eval Starter")
settings = get_settings()
store = get_store()


@app.on_event("startup")
async def startup() -> None:
    configure_logging(settings.log_level)
    validate_settings(settings)
    store.ensure_collection()


app.add_middleware(RequestIdMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(RateLimitMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    payload = ErrorResponse(code="http_error", message=str(exc.detail), request_id=request_id)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception", exc_info=exc)
    increment("errors")
    request_id = getattr(request.state, "request_id", None)
    payload = ErrorResponse(code="internal_error", message="Internal server error", request_id=request_id)
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    if not file.filename.lower().endswith((".txt", ".md", ".pdf")):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    try:
        data = await file.read()
        result = ingest_document(file.filename, data, store)
        increment("ingest_requests")
        logger.info("ingest_complete", extra={"doc_id": result["doc_id"], "chunks": result["chunks"]})
        return IngestResponse(**result)
    except ValueError as exc:
        increment("errors")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    results = hybrid_search(payload.question, store)
    increment("query_requests")
    logger.info("query_complete", extra={"top_k": len(results)})
    answer = "\n".join(
        [
            "Answer (extractive):",
            *[f"- {item['snippet']}" for item in results[:2]],
        ]
    )
    return QueryResponse(answer=answer, citations=results)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "environment": settings.environment}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return render_prometheus()
