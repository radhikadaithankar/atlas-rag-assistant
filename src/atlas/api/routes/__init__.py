"""Routers included by the application factory."""

from atlas.api.routes.ask import router as ask_router
from atlas.api.routes.documents import router as documents_router
from atlas.api.routes.health import router as health_router
from atlas.api.routes.ready import router as ready_router
from atlas.api.routes.search import router as search_router

__all__ = [
    "ask_router",
    "documents_router",
    "health_router",
    "ready_router",
    "search_router",
]
