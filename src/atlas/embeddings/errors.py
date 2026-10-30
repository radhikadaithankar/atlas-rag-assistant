"""Provider failures that the API maps to HTTP 502."""

from __future__ import annotations


class EmbeddingProviderError(Exception):
    def __init__(self, message: str = "the embedding provider failed") -> None:
        self.message = message
        super().__init__(message)


class EmbeddingShapeError(Exception):
    def __init__(
        self, message: str = "the embedding provider returned the wrong length"
    ) -> None:
        self.message = message
        super().__init__(message)
