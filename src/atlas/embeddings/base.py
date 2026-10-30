"""A batch of strings in, one vector per string out, same order."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input string, in the same order."""
