"""FastAPI application entry point.

Owns app assembly, CORS, global exception handling and startup warm-up.
All handlers return the same JSON error envelope and never leak stack traces.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import api_router
from app.api.deps import get_container
from app.config import configure_logging, get_settings
from app.errors import AppError
from app.prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings)

API_VERSION = "1.0.0"
API_PREFIX = "/api"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up services so the first request is not slowed by model loading."""
    settings.ensure_directories()
    logger.info(
        "Starting RAG Document Chatbot (env=%s, model=%s, prompts=%s)",
        settings.app_env,
        settings.groq_model,
        PROMPT_VERSION,
    )
    if not settings.groq_configured:
        logger.warning(
            "GROQ_API_KEY is not set. Uploads and indexing will work, but "
            "summarization and chat will return a configuration error."
        )
    try:
        container = get_container()
        # Touch the vector store so a misconfigured path fails fast at boot.
        container.vector_store.count()
        logger.info("Services initialised")
    except Exception as exc:
        # Never crash the process on warm-up: /health must stay reachable so the
        # operator can see what is misconfigured.
        logger.error("Service warm-up failed: %s", type(exc).__name__)

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="RAG Document Chatbot API",
    description=(
        "Upload documents, get automatic summaries, and ask questions answered "
        "strictly from document content with verifiable source citations. "
        "LLM inference is served by the Groq API."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Accept"],
    max_age=600,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request ID and log timing. Document contents are never logged."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - started) * 1000
        logger.exception(
            "Unhandled error | id=%s %s %s | %.0fms",
            request_id,
            request.method,
            request.url.path,
            elapsed,
        )
        raise

    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    if request.url.path not in {"/health", f"{API_PREFIX}/health"}:
        logger.info(
            "%s %s -> %s | %.0fms | id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            request_id,
        )
    return response


# --------------------------------------------------------------------------
# Exception handlers
# --------------------------------------------------------------------------
@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """Known application errors: safe message, correct status code."""
    request_id = getattr(request.state, "request_id", None)
    log = logger.warning if exc.status_code < 500 else logger.error
    log("AppError | %s | %s | id=%s", exc.code, exc.status_code, request_id)

    payload = exc.to_payload()
    if request_id:
        payload["error"]["request_id"] = request_id
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Pydantic validation failures, reduced to field-level messages."""
    fields = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        fields.append({"field": location or "body", "issue": error.get("msg", "invalid")})

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "The request payload is invalid.",
                "details": {"fields": fields},
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Framework-level HTTP errors, mapped into the standard envelope."""
    codes = {
        404: "not_found",
        405: "method_not_allowed",
        413: "file_too_large",
        415: "unsupported_file_type",
        429: "rate_limited",
    }
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": codes.get(exc.status_code, "http_error"),
                "message": str(exc.detail),
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: log internally, return a generic message externally."""
    request_id = getattr(request.state, "request_id", None)
    logger.exception("Unhandled exception | id=%s", request_id)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred. Please try again.",
                "request_id": request_id,
            }
        },
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
app.include_router(api_router, prefix=API_PREFIX)
# Also mount unprefixed so `GET /health` works for load balancers.
app.include_router(api_router)


@app.get("/", tags=["meta"], summary="API metadata")
async def root() -> dict:
    return {
        "name": "RAG Document Chatbot API",
        "version": API_VERSION,
        "prompt_version": PROMPT_VERSION,
        "docs": "/docs",
        "health": f"{API_PREFIX}/health",
        "endpoints": {
            "upload": f"POST {API_PREFIX}/documents",
            "list": f"GET {API_PREFIX}/documents",
            "get": f"GET {API_PREFIX}/documents/{{document_id}}",
            "delete": f"DELETE {API_PREFIX}/documents/{{document_id}}",
            "summarize": f"POST {API_PREFIX}/documents/{{document_id}}/summarize",
            "sources": f"GET {API_PREFIX}/documents/{{document_id}}/sources",
            "chat": f"POST {API_PREFIX}/chat",
            "chat_stream": f"POST {API_PREFIX}/chat/stream",
            "chat_history": f"GET {API_PREFIX}/chat/history",
            "chat_clear": f"POST {API_PREFIX}/chat/clear",
        },
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production,
        log_level=settings.log_level.lower(),
    )
