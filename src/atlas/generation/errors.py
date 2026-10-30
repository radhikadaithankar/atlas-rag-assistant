"""Generation failures that the API maps to HTTP 502."""

from __future__ import annotations


class GenerationProviderError(Exception):
    def __init__(self, message: str = "the generation provider failed") -> None:
        self.message = message
        super().__init__(message)


class UnavailableError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
