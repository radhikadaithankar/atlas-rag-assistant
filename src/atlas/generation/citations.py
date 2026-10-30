"""Keep a citation only when the chunk was retrieved and the quote is in it."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from atlas.retrieval.hits import Hit


@dataclass(frozen=True)
class RawCitation:
    chunk_id: uuid.UUID
    quote: str


@dataclass(frozen=True)
class ParsedModel:
    answer: str
    abstained: bool
    citations: list[RawCitation]


@dataclass(frozen=True)
class KeptCitation:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    section: str | None
    quote: str


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def parse_model_json(text: str) -> ParsedModel | None:
    """Return None when the model text is not the expected JSON object."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    answer = payload.get("answer")
    abstained = payload.get("abstained")
    if not isinstance(answer, str) or not isinstance(abstained, bool):
        return None
    raw_citations = payload.get("citations", [])
    if not isinstance(raw_citations, list):
        return None
    citations: list[RawCitation] = []
    for item in raw_citations:
        if not isinstance(item, dict):
            return None
        quote = item.get("quote")
        if not isinstance(quote, str):
            return None
        chunk_raw = item.get("chunk_id")
        try:
            chunk_id = uuid.UUID(str(chunk_raw))
        except (TypeError, ValueError):
            continue
        citations.append(RawCitation(chunk_id, quote))
    return ParsedModel(answer, abstained, citations)


def validate_citations(
    raw: list[RawCitation], passages: list[Hit]
) -> list[KeptCitation]:
    by_id = {passage.chunk_id: passage for passage in passages}
    kept: list[KeptCitation] = []
    for item in raw:
        passage = by_id.get(item.chunk_id)
        if passage is None:
            continue
        if collapse_whitespace(item.quote) not in collapse_whitespace(passage.text):
            continue
        kept.append(
            KeptCitation(
                chunk_id=passage.chunk_id,
                document_id=passage.document_id,
                filename=passage.filename,
                page=passage.page,
                section=passage.section,
                quote=item.quote,
            )
        )
    return kept
