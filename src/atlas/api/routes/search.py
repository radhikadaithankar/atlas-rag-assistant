"""Search. The handler embeds when needed and does not rank."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from atlas.api.deps import (
    maybe_rerank,
    pool_limit,
    require_embedder,
    run_retrieve,
    settings_of,
)
from atlas.api.http import error_json
from atlas.api.schemas.search import SearchHitOut, SearchRequest, SearchResponse
from atlas.config import AtlasConfigError, Settings
from atlas.database.session import session_scope
from atlas.embeddings.base import Embedder
from atlas.embeddings.errors import EmbeddingProviderError, EmbeddingShapeError
from atlas.generation.errors import UnavailableError
from atlas.retrieval.hits import Hit
from atlas.retrieval.reranker import RerankUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.post("/search")
def search(body: SearchRequest, request: Request) -> JSONResponse:
    settings = settings_of(request)
    try:
        limit = pool_limit(settings, rerank=body.rerank, top_k=body.top_k)
    except ValueError as exc:
        return error_json(422, "invalid_request", str(exc))
    embedder = None
    vector: list[float] | None = None
    if body.mode != "keyword":
        try:
            embedder = require_embedder(request)
            embedded = embedder.embed([body.query])
            vector = embedded[0]
        except UnavailableError as exc:
            return error_json(503, "unavailable", exc.message)
        except EmbeddingShapeError as exc:
            logger.error("%s", type(exc).__name__)
            return error_json(502, "provider", exc.message)
        except EmbeddingProviderError as exc:
            logger.error("%s", type(exc).__name__)
            return error_json(502, "provider", "the embedding provider failed")
        except Exception as exc:
            logger.error("%s", type(exc).__name__)
            return error_json(502, "provider", "the embedding provider failed")
    if body.rerank:
        try:
            require_reranker_ready(request)
        except RerankUnavailableError as exc:
            return error_json(503, "unavailable", exc.message)
    try:
        hits = _collect(
            request,
            settings,
            query=body.query,
            mode=body.mode,
            top_k=limit,
            document_id=body.document_id,
            embedder=embedder,
            vector=vector,
        )
    except AtlasConfigError as exc:
        logger.error("%s", type(exc).__name__)
        return error_json(503, "unavailable", str(exc))
    except Exception as exc:
        logger.error("%s", type(exc).__name__)
        if body.mode == "keyword":
            return error_json(500, "internal", "The request failed.")
        return error_json(502, "provider", "the embedding provider failed")
    try:
        hits = maybe_rerank(
            request,
            enabled=body.rerank,
            query=body.query,
            hits=hits,
            top_k=body.top_k,
        )
    except RerankUnavailableError as exc:
        return error_json(503, "unavailable", exc.message)
    except Exception as exc:
        logger.error("%s", type(exc).__name__)
        return error_json(503, "unavailable", "Cross-encoder reranking is unavailable.")
    payload = SearchResponse(
        mode=body.mode,
        hits=[
            SearchHitOut(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                filename=hit.filename,
                page=hit.page,
                section=hit.section,
                text=hit.text,
                score=hit.score,
                distance=hit.distance,
                keyword_rank=hit.keyword_rank,
                vector_rank=hit.vector_rank,
            )
            for hit in hits
        ],
    )
    return JSONResponse(status_code=200, content=payload.model_dump(mode="json"))


def _collect(
    request: Request,
    settings: Settings,
    *,
    query: str,
    mode: str,
    top_k: int,
    document_id: object,
    embedder: Embedder | None,
    vector: list[float] | None,
) -> list[Hit]:
    if request.app.state.retriever is not None:
        return run_retrieve(
            request,
            None,
            query=query,
            mode=mode,
            top_k=top_k,
            document_id=document_id,
            embedder=embedder,
            vector=vector,
        )
    with session_scope(settings) as session:
        return run_retrieve(
            request,
            session,
            query=query,
            mode=mode,
            top_k=top_k,
            document_id=document_id,
            embedder=embedder,
            vector=vector,
        )


def require_reranker_ready(request: Request) -> None:
    """Fail before a database socket when the extra is missing and no double is set."""

    if request.app.state.reranker is not None:
        return
    from atlas.api.deps import require_reranker

    require_reranker(request)
