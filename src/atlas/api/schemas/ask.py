"""Ask requests. The response never includes the system prompt."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: uuid.UUID | None = None
    rerank: bool = False


class CitationOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    section: str | None
    quote: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[CitationOut]
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost: None = None
