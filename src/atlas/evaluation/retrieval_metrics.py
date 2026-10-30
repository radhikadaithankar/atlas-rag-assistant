"""Deterministic retrieval scores. These do not say a model would use the evidence."""

from __future__ import annotations

from atlas.generation.citations import collapse_whitespace


def quote_in_text(quote: str, text: str) -> bool:
    return collapse_whitespace(quote) in collapse_whitespace(text)


def recall_at_k(retrieved: list[str], quotes: list[str], *, k: int = 5) -> float:
    if not quotes:
        raise ValueError("recall requires gold quotes")
    window = retrieved[:k]
    found = sum(
        1 for quote in quotes if any(quote_in_text(quote, text) for text in window)
    )
    return found / len(quotes)


def reciprocal_rank(retrieved: list[str], quotes: list[str]) -> float:
    for index, text in enumerate(retrieved, start=1):
        if any(quote_in_text(quote, text) for quote in quotes):
            return 1.0 / index
    return 0.0


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return sum(values) / len(values)


def containment(chunks: list[str], quotes: list[str]) -> tuple[float, int, int]:
    """Fraction of unique quotes that fit entirely inside some chunk."""

    if not quotes:
        raise ValueError("containment requires quotes")
    found = sum(
        1 for quote in quotes if any(quote_in_text(quote, chunk) for chunk in chunks)
    )
    return found / len(quotes), found, len(quotes)
