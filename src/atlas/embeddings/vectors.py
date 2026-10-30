"""Shared vector checks. A wrong length is rejected before any write."""

from __future__ import annotations

from atlas.config import EMBEDDING_WIDTH
from atlas.embeddings.errors import EmbeddingShapeError


def require_width(
    vectors: list[list[float]], *, expected: int = EMBEDDING_WIDTH
) -> list[list[float]]:
    checked: list[list[float]] = []
    for vector in vectors:
        if len(vector) != expected:
            raise EmbeddingShapeError(
                "the embedding provider returned a vector of the wrong length"
            )
        checked.append([float(value) for value in vector])
    return checked
