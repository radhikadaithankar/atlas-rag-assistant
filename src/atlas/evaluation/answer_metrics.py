"""Citation checks and a judge score that is not ground truth.

The published Gemini price is applied only by the harness. The API field
``estimated_cost`` stays null.
"""

from __future__ import annotations

import math

from atlas.generation.citations import collapse_whitespace

# Gemini Developer API pricing for gemini-3.5-flash-lite, paid standard tier.
# Page last updated 2026-10-22 UTC, checked 23 Oct 2026:
# https://ai.google.dev/gemini-api/docs/pricing
# The free tier on that page is free of charge. This product is a published
# price times measured tokens, not an invoice.
GEMINI_INPUT_USD_PER_MILLION = 0.30
GEMINI_OUTPUT_USD_PER_MILLION = 2.50

FAITHFULNESS_NOT_GROUND_TRUTH = "This score is not ground truth. The judge can agree with a fluent unsupported answer."


def citation_precision(
    *,
    abstained: bool,
    citation_quotes: list[str],
    gold_quotes: list[str],
) -> float:
    """Precision of citations against gold quotes. Abstaining on a gold question scores 0."""

    if abstained or not citation_quotes:
        return 0.0
    hits = 0
    for quote in citation_quotes:
        if any(
            collapse_whitespace(quote) in collapse_whitespace(gold)
            or collapse_whitespace(gold) in collapse_whitespace(quote)
            for gold in gold_quotes
        ):
            hits += 1
    return hits / len(citation_quotes)


def citation_form_ok(
    *,
    abstained: bool,
    citations: list[tuple[str, str]],
    retrieved: dict[str, str],
) -> bool:
    """A citation keeps its chunk id and a quote that occurs in that passage."""

    if abstained:
        return not citations
    if not citations:
        return False
    for chunk_id, quote in citations:
        passage = retrieved.get(chunk_id)
        if passage is None:
            return False
        if collapse_whitespace(quote) not in collapse_whitespace(passage):
            return False
    return True


def linear_percentile(values: list[float], percentile: float) -> float:
    """Linear interpolation on the sorted values. ``percentile`` is between 0 and 1."""

    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def estimate_gemini_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * GEMINI_INPUT_USD_PER_MILLION
        + completion_tokens / 1_000_000 * GEMINI_OUTPUT_USD_PER_MILLION
    )


class PendingScoreError(ValueError):
    """The runner refuses to store a numeric value on a pending row."""


def assign_metric(row: dict[str, object], value: float) -> None:
    if row.get("status") == "pending":
        raise PendingScoreError("refusing to store a numeric value on a pending row")
    row["value"] = value
