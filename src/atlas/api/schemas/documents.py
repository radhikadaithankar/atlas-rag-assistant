"""Document responses. Embeddings are not included."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    id: uuid.UUID
    filename: str
    source_format: str
    created_at: datetime
    chunk_count: int


class DocumentList(BaseModel):
    items: list[DocumentSummary]
    total: int
    offset: int
    limit: int


class ErrorBody(BaseModel):
    error: str
    message: str


class DuplicateBody(BaseModel):
    error: str = "duplicate"
    message: str
    document: DocumentSummary


class EmbedBody(BaseModel):
    document_id: uuid.UUID
    updated: int


class PageQuery(BaseModel):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)
