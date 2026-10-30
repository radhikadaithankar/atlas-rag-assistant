"""Run one retriever mode. Fusion stays here, not in the HTTP handler."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from atlas.embeddings.base import Embedder
from atlas.retrieval.fusion import reciprocal_rank_fusion
from atlas.retrieval.hits import Hit
from atlas.retrieval.keyword_search import search_keywords
from atlas.retrieval.vector_search import search_vectors


def retrieve(
    session: Session,
    *,
    query: str,
    mode: str,
    top_k: int,
    document_id: uuid.UUID | None,
    embedder: Embedder | None,
    rrf_k: int,
    vector: list[float] | None = None,
) -> list[Hit]:
    if mode == "keyword":
        return search_keywords(session, query, top_k=top_k, document_id=document_id)
    if vector is None:
        if embedder is None:
            raise RuntimeError("vector retrieval requires an embedder")
        vector = embedder.embed([query])[0]
    if mode == "vector":
        return search_vectors(session, vector, top_k=top_k, document_id=document_id)
    vector_hits = search_vectors(session, vector, top_k=top_k, document_id=document_id)
    keyword_hits = search_keywords(session, query, top_k=top_k, document_id=document_id)
    fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=rrf_k)
    return fused[:top_k]
