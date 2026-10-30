"""Parameterised reads and writes for documents and chunks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from atlas.database.models import Chunk, Document


class DocumentConflict(Exception):
    """The unique hash fired and the winning row is not visible."""


def get_document(session: Session, document_id: uuid.UUID) -> Document | None:
    return session.get(Document, document_id)


def get_by_hash(session: Session, content_sha256: str) -> Document | None:
    statement = select(Document).where(Document.content_sha256 == content_sha256)
    return session.scalars(statement).first()


def count_documents(session: Session) -> int:
    value = session.scalar(select(func.count()).select_from(Document))
    return int(value or 0)


def count_chunks(session: Session, document_id: uuid.UUID | None = None) -> int:
    statement = select(func.count()).select_from(Chunk)
    if document_id is not None:
        statement = statement.where(Chunk.document_id == document_id)
    value = session.scalar(statement)
    return int(value or 0)


def list_documents(session: Session, *, offset: int, limit: int) -> list[Document]:
    statement = (
        select(Document)
        .order_by(Document.created_at.desc(), Document.id)
        .offset(offset)
        .limit(limit)
    )
    return list(session.scalars(statement).all())


def chunk_count_for(session: Session, document_id: uuid.UUID) -> int:
    return count_chunks(session, document_id)


def list_chunks(session: Session, document_id: uuid.UUID) -> list[Chunk]:
    statement = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    )
    return list(session.scalars(statement).all())


def delete_document(session: Session, document_id: uuid.UUID) -> bool:
    document = get_document(session, document_id)
    if document is None:
        return False
    session.execute(delete(Document).where(Document.id == document_id))
    return True


@dataclass(frozen=True)
class NewChunk:
    id: uuid.UUID
    chunk_index: int
    text: str
    page: int | None
    section: str | None
    embedding: list[float] | None


def insert_document(
    session: Session,
    *,
    document_id: uuid.UUID,
    filename: str,
    content_sha256: str,
    source_format: str,
    created_at: datetime,
    chunks: list[NewChunk],
) -> tuple[Document, bool]:
    """Insert a document. A visible duplicate hash returns the stored row."""

    existing = get_by_hash(session, content_sha256)
    if existing is not None:
        return existing, False
    document = Document(
        id=document_id,
        filename=filename,
        content_sha256=content_sha256,
        source_format=source_format,
        created_at=created_at,
    )
    session.add(document)
    for row in chunks:
        session.add(
            Chunk(
                id=row.id,
                document_id=document_id,
                chunk_index=row.chunk_index,
                text=row.text,
                page=row.page,
                section=row.section,
                embedding=row.embedding,
                created_at=created_at,
            )
        )
    try:
        session.flush()
    except IntegrityError:
        session.rollback()
        existing = get_by_hash(session, content_sha256)
        if existing is not None:
            return existing, False
        raise DocumentConflict() from None
    return document, True


def set_embeddings(
    session: Session,
    pairs: list[tuple[uuid.UUID, list[float]]],
) -> int:
    updated = 0
    for chunk_id, vector in pairs:
        chunk = session.get(Chunk, chunk_id)
        if chunk is None:
            continue
        chunk.embedding = vector
        updated += 1
    session.flush()
    return updated
