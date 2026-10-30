"""English full-text search. The query is a bound parameter. No embedding."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from atlas.database.models import Chunk, Document
from atlas.retrieval.hits import Hit


def search_keywords(
    session: Session,
    query: str,
    *,
    top_k: int,
    document_id: uuid.UUID | None = None,
) -> list[Hit]:
    if query.strip() == "":
        return []
    tsquery = func.plainto_tsquery("english", query)
    rank = func.ts_rank_cd(Chunk.search_vector, tsquery)
    statement = (
        select(Chunk, Document, rank.label("rank"))
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.search_vector.op("@@")(tsquery))
        .order_by(rank.desc(), Chunk.id)
        .limit(top_k)
    )
    if document_id is not None:
        statement = statement.where(Chunk.document_id == document_id)
    hits: list[Hit] = []
    for position, row in enumerate(session.execute(statement).all(), start=1):
        chunk = row[0]
        document = row[1]
        score = row[2]
        hits.append(
            Hit(
                chunk_id=chunk.id,
                document_id=document.id,
                filename=document.filename,
                page=chunk.page,
                section=chunk.section,
                text=chunk.text,
                score=float(score),
                keyword_rank=position,
            )
        )
    return hits
