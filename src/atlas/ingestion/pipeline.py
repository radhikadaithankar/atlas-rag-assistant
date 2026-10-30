"""Parse, hash, chunk, and store one upload in a single commit."""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from atlas.config import Settings
from atlas.database.repository import DocumentConflict, NewChunk, insert_document
from atlas.ingestion.chunker import TextChunk, chunk_parsed
from atlas.ingestion.errors import IngestionError, UploadTooLargeError
from atlas.ingestion.parser import ParsedDocument, parse_bytes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedUpload:
    filename: str
    content_sha256: str
    source_format: str
    parsed: ParsedDocument
    chunks: list[TextChunk]


def safe_filename(name: str | None) -> str:
    """Keep the base name. A path is not created on disk."""

    if name is None or name == "":
        return "upload"
    base = Path(name).name
    if base in {"", ".", ".."}:
        return "upload"
    return base


def prepare_upload(
    data: bytes, filename: str | None, settings: Settings
) -> PreparedUpload:
    if len(data) > settings.max_upload_bytes:
        raise UploadTooLargeError()
    parsed = parse_bytes(data)
    chunks = chunk_parsed(
        parsed, size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    return PreparedUpload(
        filename=safe_filename(filename),
        content_sha256=hashlib.sha256(data).hexdigest(),
        source_format=parsed.source_format,
        parsed=parsed,
        chunks=chunks,
    )


def store_prepared(
    session: Session,
    prepared: PreparedUpload,
    vectors: list[list[float]] | None,
) -> tuple[uuid.UUID, bool]:
    """Insert the document and its chunks. ``vectors`` align with chunks."""

    if vectors is not None and len(vectors) != len(prepared.chunks):
        raise IngestionError("unreadable", "Embeddings did not match the chunks.")
    created_at = datetime.now(UTC)
    document_id = uuid.uuid4()
    rows: list[NewChunk] = []
    for index, chunk in enumerate(prepared.chunks):
        embedding: list[float] | None = None if vectors is None else vectors[index]
        rows.append(
            NewChunk(
                id=uuid.uuid4(),
                chunk_index=index,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
                embedding=embedding,
            )
        )
    try:
        document, created = insert_document(
            session,
            document_id=document_id,
            filename=prepared.filename,
            content_sha256=prepared.content_sha256,
            source_format=prepared.source_format,
            created_at=created_at,
            chunks=rows,
        )
    except DocumentConflict:
        logger.error("%s", "DocumentConflict")
        raise IngestionError("conflict", "The document could not be stored.") from None
    except Exception as exc:
        logger.error("%s", type(exc).__name__)
        raise IngestionError(
            "unreadable", "The document could not be stored."
        ) from None
    return document.id, created
