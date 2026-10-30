"""One retrieved passage. Ranking fields stay with the hit, not the route."""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass
class Hit:
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
