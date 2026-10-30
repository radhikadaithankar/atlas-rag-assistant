"""Resolve providers and the retriever without ranking in the route."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

from fastapi import Request
from sqlalchemy.orm import Session

from atlas.config import MISSING_PROVIDER_MESSAGE, Settings, select_provider
from atlas.embeddings.apply import embed_document
from atlas.embeddings.base import Embedder
from atlas.embeddings.errors import EmbeddingProviderError, EmbeddingShapeError
from atlas.embeddings.factory import build_embedder
from atlas.generation.base import Generator
from atlas.generation.errors import GenerationProviderError, UnavailableError
from atlas.generation.factory import build_generator
from atlas.retrieval.hits import Hit
from atlas.retrieval.reranker import (
    Reranker,
    RerankUnavailableError,
    apply_rerank,
    load_cross_encoder,
)
from atlas.retrieval.service import retrieve

logger = logging.getLogger(__name__)

Retriever = Callable[..., list[Hit]]


def settings_of(request: Request) -> Settings:
    value = request.app.state.settings
    if not isinstance(value, Settings):
        raise UnavailableError("Settings are not loaded.")
    return value


def require_embedder(request: Request) -> Embedder:
    injected = request.app.state.embedder
    if injected is not None:
        return cast(Embedder, injected)
    settings = settings_of(request)
    if select_provider(settings) == "none":
        raise UnavailableError(MISSING_PROVIDER_MESSAGE)
    try:
        return build_embedder(settings)
    except (EmbeddingProviderError, UnavailableError):
        raise
    except Exception as exc:
        logger.error("%s", type(exc).__name__)
        raise EmbeddingProviderError() from None


def require_generator(request: Request) -> Generator:
    injected = request.app.state.generator
    if injected is not None:
        return cast(Generator, injected)
    settings = settings_of(request)
    if select_provider(settings) == "none":
        raise UnavailableError(MISSING_PROVIDER_MESSAGE)
    try:
        return build_generator(settings)
    except UnavailableError:
        raise
    except Exception as exc:
        logger.error("%s", type(exc).__name__)
        raise GenerationProviderError() from None


def require_reranker(request: Request) -> Reranker:
    injected = request.app.state.reranker
    if injected is not None:
        return cast(Reranker, injected)
    settings = settings_of(request)
    if settings.reranker not in {None, "cross_encoder"}:
        raise RerankUnavailableError(
            "Cross-encoder reranking is unavailable because the rerank extra is not installed."
        )
    try:
        return load_cross_encoder()
    except RerankUnavailableError:
        raise
    except Exception as exc:
        logger.error("%s", type(exc).__name__)
        raise RerankUnavailableError(
            "Cross-encoder reranking is unavailable because the rerank extra is not installed."
        ) from None


def run_retrieve(
    request: Request,
    session: Session | None,
    *,
    query: str,
    mode: str,
    top_k: int,
    document_id: object,
    embedder: Embedder | None,
    vector: list[float] | None = None,
) -> list[Hit]:
    settings = settings_of(request)
    injected = request.app.state.retriever
    if injected is not None:
        hits = injected(
            session=session,
            query=query,
            mode=mode,
            top_k=top_k,
            document_id=document_id,
            embedder=embedder,
            rrf_k=settings.rrf_k,
            vector=vector,
        )
        return list(hits)
    if session is None:
        raise UnavailableError(MISSING_PROVIDER_MESSAGE)
    from uuid import UUID

    parsed_id = document_id if isinstance(document_id, UUID) else None
    return retrieve(
        session,
        query=query,
        mode=mode,
        top_k=top_k,
        document_id=parsed_id,
        embedder=embedder,
        rrf_k=settings.rrf_k,
        vector=vector,
    )


def maybe_rerank(
    request: Request,
    *,
    enabled: bool,
    query: str,
    hits: list[Hit],
    top_k: int,
) -> list[Hit]:
    if not enabled:
        return hits
    reranker = require_reranker(request)
    return apply_rerank(reranker, query, hits, top_k=top_k)


def pool_limit(settings: Settings, *, rerank: bool, top_k: int) -> int:
    if rerank and top_k > settings.rerank_pool:
        raise ValueError("top_k cannot exceed RERANK_POOL when rerank is true.")
    if rerank:
        return settings.rerank_pool
    return top_k


def fill_embeddings(
    session: Session,
    document_id: object,
    embedder: Embedder,
    *,
    replace: bool,
) -> int:
    from uuid import UUID

    if not isinstance(document_id, UUID):
        raise EmbeddingProviderError()
    try:
        return embed_document(session, document_id, embedder, replace=replace)
    except EmbeddingShapeError:
        raise
    except EmbeddingProviderError:
        raise
    except Exception as exc:
        logger.error("%s", type(exc).__name__)
        raise EmbeddingProviderError() from None
