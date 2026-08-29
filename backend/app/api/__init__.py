"""API layer: routers and dependencies."""

from fastapi import APIRouter

from app.api.routes import chat, documents, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)

__all__ = ["api_router"]
