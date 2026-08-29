"""Health and capability endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import ServiceContainer, container_dep
from app.schemas.api import HealthResponse
from app.services.extraction import registry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service health")
async def health(
    container: ServiceContainer = Depends(container_dep),
) -> HealthResponse:
    """Report configuration and readiness. Never returns secret values."""
    settings = container.settings

    try:
        documents_indexed = container.processor.registry.count()
    except Exception:
        documents_indexed = 0

    return HealthResponse(
        status="ok",
        app_env=settings.app_env,
        # Reports only whether a key is present — never the key itself.
        llm_configured=container.groq.is_configured,
        llm_model=settings.groq_model,
        embedding_model=(
            "hashing-offline"
            if settings.embedding_offline_fallback
            else settings.embedding_model
        ),
        vector_store=settings.vector_store,
        documents_indexed=documents_indexed,
        supported_extensions=registry.supported_extensions,
    )


@router.get("/health/llm", summary="Verify Groq connectivity")
async def llm_health(container: ServiceContainer = Depends(container_dep)) -> dict:
    """Make one minimal Groq call to confirm the key and model work."""
    ok, detail = container.groq.health_check()
    return {
        "ok": ok,
        "detail": detail,
        "model": container.settings.groq_model,
        "configured": container.groq.is_configured,
    }
