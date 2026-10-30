"""Fill null chunk embeddings, or replace them when asked."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from atlas.database.repository import list_chunks, set_embeddings
from atlas.embeddings.base import Embedder
from atlas.ingestion.errors import IngestionError


def embed_document(
    session: Session,
    document_id: uuid.UUID,
    embedder: Embedder,
    *,
    replace: bool = False,
) -> int:
    chunks = list_chunks(session, document_id)
    if not chunks and replace:
        return 0
    pending = (
        chunks if replace else [chunk for chunk in chunks if chunk.embedding is None]
    )
    if not pending:
        return 0
    vectors = embedder.embed([chunk.text for chunk in pending])
    if len(vectors) != len(pending):
        raise IngestionError("unreadable", "Embeddings did not match the chunks.")
    return set_embeddings(
        session, list(zip((chunk.id for chunk in pending), vectors, strict=True))
    )
