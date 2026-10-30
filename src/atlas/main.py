"""Application factory. Uvicorn serves ``atlas.main:app``."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI

from atlas import __version__
from atlas.api.routes import (
    ask_router,
    documents_router,
    health_router,
    ready_router,
    search_router,
)
from atlas.config import Settings, load_settings
from atlas.embeddings.base import Embedder
from atlas.generation.base import Generator
from atlas.retrieval.hits import Hit
from atlas.retrieval.reranker import Reranker

Retriever = Callable[..., list[Hit]]


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    generator: Generator | None = None,
    reranker: Reranker | None = None,
    retriever: Retriever | None = None,
) -> FastAPI:
    """Build one FastAPI app. Invalid settings fail here, before a handler runs."""

    active = settings if settings is not None else load_settings()
    app = FastAPI(title="Atlas", version=__version__)
    app.state.settings = active
    app.state.embedder = embedder
    app.state.generator = generator
    app.state.reranker = reranker
    app.state.retriever = retriever
    app.include_router(health_router)
    app.include_router(ready_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(ask_router)
    return app


app = create_app()
