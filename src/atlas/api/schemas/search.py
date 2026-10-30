"""Search requests. Ranking is not computed in the handler."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: uuid.UUID | None = None
    rerank: bool = False


class SearchHitOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    section: str | None
    text: str
    score: float
    distance: float | None = None
    keyword_rank: int | None = None
    vector_rank: int | None = None


class SearchResponse(BaseModel):
    mode: str
    hits: list[SearchHitOut]
