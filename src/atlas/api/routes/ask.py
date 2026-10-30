"""Grounded question answering. Citation checks run after the model returns."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from atlas.api.deps import (
    maybe_rerank,
    pool_limit,
    require_embedder,
    require_generator,
    settings_of,
)
from atlas.api.http import error_json
from atlas.api.routes.search import require_reranker_ready
from atlas.api.schemas.ask import AskRequest, AskResponse, CitationOut
from atlas.config import MISSING_PROVIDER_MESSAGE, AtlasConfigError, select_provider
from atlas.embeddings.errors import EmbeddingProviderError, EmbeddingShapeError
from atlas.generation.errors import GenerationProviderError, UnavailableError
from atlas.generation.service import Answer, answer_from_passages
from atlas.retrieval.reranker import RerankUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


@router.post("/ask")
def ask(body: AskRequest, request: Request) -> JSONResponse:
    settings = settings_of(request)
    try:
        limit = pool_limit(settings, rerank=body.rerank, top_k=body.top_k)
    except ValueError as exc:
        return error_json(422, "invalid_request", str(exc))
    if request.app.state.generator is None and select_provider(settings) == "none":
        return error_json(503, "unavailable", MISSING_PROVIDER_MESSAGE)
    if body.rerank:
        try:
            require_reranker_ready(request)
        except RerankUnavailableError as exc:
            return error_json(503, "unavailable", exc.message)
    try:
        generator = require_generator(request)
    except UnavailableError as exc:
        return error_json(503, "unavailable", exc.message)
    except GenerationProviderError as exc:
        return error_json(502, "provider", exc.message)
    embedder = None
    vector: list[float] | None = None
    if body.mode != "keyword":
        try:
            embedder = require_embedder(request)
            embedded = embedder.embed([body.question])
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
    try:
        from atlas.api.routes.search import _collect

        hits = _collect(
            request,
            settings,
            query=body.question,
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
        return error_json(502, "provider", "the embedding provider failed")
    try:
        hits = maybe_rerank(
            request,
            enabled=body.rerank,
            query=body.question,
            hits=hits,
            top_k=body.top_k,
        )
    except RerankUnavailableError as exc:
        return error_json(503, "unavailable", exc.message)
    if not hits:
        answer = answer_from_passages(body.question, [], generator)
        return _response(answer)
    try:
        answer = answer_from_passages(body.question, hits, generator)
    except GenerationProviderError as exc:
        logger.error("%s", type(exc).__name__)
        return error_json(502, "provider", exc.message)
    return _response(answer)


def _response(answer: Answer) -> JSONResponse:
    payload = AskResponse(
        answer=answer.answer,
        abstained=answer.abstained,
        citations=[
            CitationOut(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                filename=item.filename,
                page=item.page,
                section=item.section,
                quote=item.quote,
            )
            for item in answer.citations
        ],
        latency_ms=answer.latency_ms,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        estimated_cost=None,
    )
    return JSONResponse(status_code=200, content=payload.model_dump(mode="json"))
