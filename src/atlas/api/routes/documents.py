"""Upload, list, fetch, delete, and embed documents."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, File, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from atlas.api.deps import fill_embeddings, require_embedder, settings_of
from atlas.api.http import error_json
from atlas.api.schemas.documents import DocumentList, DocumentSummary, EmbedBody
from atlas.config import AtlasConfigError
from atlas.database.repository import (
    chunk_count_for,
    count_documents,
    delete_document,
    get_document,
    list_documents,
)
from atlas.database.session import session_scope
from atlas.embeddings.errors import EmbeddingProviderError, EmbeddingShapeError
from atlas.generation.errors import UnavailableError
from atlas.ingestion.errors import IngestionError, UploadTooLargeError
from atlas.ingestion.pipeline import prepare_upload, store_prepared

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> JSONResponse:
    settings = settings_of(request)
    raw = await file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        return error_json(413, "too_large", "The file exceeds MAX_UPLOAD_BYTES.")
    try:
        prepared = prepare_upload(raw, file.filename, settings)
    except UploadTooLargeError as exc:
        return error_json(413, exc.code, exc.message)
    except IngestionError as exc:
        return error_json(422, exc.code, exc.message)
    vectors: list[list[float]] | None
    injected = request.app.state.embedder
    if injected is not None or _provider_configured(request):
        try:
            embedder = require_embedder(request)
            texts = [chunk.text for chunk in prepared.chunks]
            vectors = embedder.embed(texts) if texts else []
        except UnavailableError as exc:
            return error_json(503, "unavailable", exc.message)
        except EmbeddingShapeError as exc:
            logger.error("%s", type(exc).__name__)
            return error_json(502, "provider", exc.message)
        except EmbeddingProviderError as exc:
            logger.error("%s", type(exc).__name__)
            return error_json(502, "provider", exc.message)
        except Exception as exc:
            logger.error("%s", type(exc).__name__)
            return error_json(502, "provider", "the embedding provider failed")
    else:
        vectors = None
    try:
        with session_scope(settings) as session:
            document_id, created = store_prepared(session, prepared, vectors)
            document = get_document(session, document_id)
            if document is None:
                return error_json(422, "conflict", "The document could not be stored.")
            summary = DocumentSummary(
                id=document.id,
                filename=document.filename,
                source_format=document.source_format,
                created_at=document.created_at,
                chunk_count=chunk_count_for(session, document.id),
            )
    except IngestionError as exc:
        status = 409 if exc.code == "duplicate" else 422
        return error_json(status, exc.code, exc.message)
    except AtlasConfigError as exc:
        logger.error("%s", type(exc).__name__)
        return error_json(
            503, "unavailable", exc.args[0] if exc.args else "DATABASE_URL is not set."
        )
    except Exception as exc:
        logger.error("%s", type(exc).__name__)
        return error_json(422, "unreadable", "The document could not be stored.")
    body = summary.model_dump(mode="json")
    if not created:
        return JSONResponse(
            status_code=409,
            content={
                "error": "duplicate",
                "message": "A document with the same bytes is already stored.",
                "document": body,
            },
        )
    return JSONResponse(status_code=201, content=body)


@router.get("", response_model=DocumentList)
def list_stored(
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> DocumentList | JSONResponse:
    settings = settings_of(request)
    try:
        with session_scope(settings) as session:
            total = count_documents(session)
            rows = list_documents(session, offset=offset, limit=limit)
            items = [
                DocumentSummary(
                    id=row.id,
                    filename=row.filename,
                    source_format=row.source_format,
                    created_at=row.created_at,
                    chunk_count=chunk_count_for(session, row.id),
                )
                for row in rows
            ]
    except AtlasConfigError as exc:
        logger.error("%s", type(exc).__name__)
        return error_json(503, "unavailable", str(exc))
    return DocumentList(items=items, total=total, offset=offset, limit=limit)


@router.get("/{document_id}")
def fetch_document(request: Request, document_id: uuid.UUID) -> JSONResponse:
    settings = settings_of(request)
    try:
        with session_scope(settings) as session:
            document = get_document(session, document_id)
            if document is None:
                return error_json(404, "not_found", "The document was not found.")
            summary = DocumentSummary(
                id=document.id,
                filename=document.filename,
                source_format=document.source_format,
                created_at=document.created_at,
                chunk_count=chunk_count_for(session, document.id),
            )
    except AtlasConfigError as exc:
        logger.error("%s", type(exc).__name__)
        return error_json(503, "unavailable", str(exc))
    return JSONResponse(status_code=200, content=summary.model_dump(mode="json"))


@router.delete("/{document_id}", response_model=None)
def remove_document(request: Request, document_id: uuid.UUID) -> Response:
    settings = settings_of(request)
    try:
        with session_scope(settings) as session:
            deleted = delete_document(session, document_id)
    except AtlasConfigError as exc:
        logger.error("%s", type(exc).__name__)
        return error_json(503, "unavailable", str(exc))
    if not deleted:
        return error_json(404, "not_found", "The document was not found.")
    return Response(status_code=204)


@router.post("/{document_id}/embed")
def embed_existing(
    request: Request,
    document_id: uuid.UUID,
    replace: bool = Query(default=False),
) -> JSONResponse:
    try:
        embedder = require_embedder(request)
    except UnavailableError as exc:
        return error_json(503, "unavailable", exc.message)
    except EmbeddingProviderError as exc:
        return error_json(502, "provider", exc.message)
    settings = settings_of(request)
    try:
        with session_scope(settings) as session:
            document = get_document(session, document_id)
            if document is None:
                return error_json(404, "not_found", "The document was not found.")
            try:
                updated = fill_embeddings(
                    session, document_id, embedder, replace=replace
                )
            except EmbeddingShapeError as exc:
                logger.error("%s", type(exc).__name__)
                return error_json(502, "provider", exc.message)
            except EmbeddingProviderError as exc:
                logger.error("%s", type(exc).__name__)
                return error_json(502, "provider", "the embedding provider failed")
    except AtlasConfigError as exc:
        logger.error("%s", type(exc).__name__)
        return error_json(503, "unavailable", str(exc))
    body = EmbedBody(document_id=document_id, updated=updated)
    return JSONResponse(status_code=200, content=body.model_dump(mode="json"))


def _provider_configured(request: Request) -> bool:
    from atlas.config import select_provider

    return select_provider(settings_of(request)) != "none"
