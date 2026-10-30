"""Exact cosine scan. Null embeddings are not directions, so they are skipped."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from atlas.database.models import Chunk, Document
from atlas.retrieval.hits import Hit


def search_vectors(
    session: Session,
    vector: list[float],
    *,
    top_k: int,
    document_id: uuid.UUID | None = None,
) -> list[Hit]:
    distance = Chunk.embedding.cosine_distance(vector)
    statement = (
        select(Chunk, Document, distance.label("distance"))
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    if document_id is not None:
        statement = statement.where(Chunk.document_id == document_id)
    hits: list[Hit] = []
    for rank, row in enumerate(session.execute(statement).all(), start=1):
        chunk = row[0]
        document = row[1]
        raw_distance = row[2]
        hits.append(
            Hit(
                chunk_id=chunk.id,
                document_id=document.id,
                filename=document.filename,
                page=chunk.page,
                section=chunk.section,
                text=chunk.text,
                score=float(raw_distance),
                distance=float(raw_distance),
                vector_rank=rank,
            )
        )
    return hits
