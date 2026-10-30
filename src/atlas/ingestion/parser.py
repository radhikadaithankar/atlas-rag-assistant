"""Sniff bytes and extract text.

A leading ``%PDF-`` is a PDF. Anything else must be valid UTF-8. UTF-8 with an
ATX heading is markdown. The filename and the client MIME type are not used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from atlas.ingestion.errors import IngestionError

ATX_HEADING = re.compile(r"(?m)^#{1,6}[ \t]+\S")


@dataclass(frozen=True)
class ParsedPage:
    page_number: int | None
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    source_format: str
    pages: list[ParsedPage]


def parse_bytes(data: bytes) -> ParsedDocument:
    if data == b"":
        raise IngestionError("empty", "The file is empty.")
    if data.startswith(b"%PDF-"):
        return _parse_pdf(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError(
            "unsupported", "The file is not a PDF or UTF-8 text."
        ) from exc
    if text.strip() == "":
        raise IngestionError("empty", "The file is empty.")
    if ATX_HEADING.search(text):
        return ParsedDocument("markdown", [ParsedPage(None, text)])
    return ParsedDocument("txt", [ParsedPage(None, text)])


def _parse_pdf(data: bytes) -> ParsedDocument:
    document = _open_pdf(data)
    try:
        if _is_encrypted(document):
            raise IngestionError("unreadable", "The PDF is encrypted.")
        count = _page_count(document)
        pages: list[ParsedPage] = []
        for index in range(count):
            try:
                page = document.load_page(index)
                text = _page_text(page)
            except IngestionError:
                raise
            except Exception as exc:
                raise IngestionError(
                    "unreadable", "A PDF page could not be read."
                ) from exc
            pages.append(ParsedPage(index + 1, text))
        if not pages:
            raise IngestionError("unreadable", "The PDF has no pages.")
        return ParsedDocument("pdf", pages)
    finally:
        _close_pdf(document)


def _open_pdf(data: bytes) -> Any:
    import fitz

    try:
        return fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise IngestionError("unreadable", "The PDF could not be opened.") from exc


def _close_pdf(document: Any) -> None:
    document.close()


def _is_encrypted(document: Any) -> bool:
    value = document.is_encrypted
    return bool(value)


def _page_count(document: Any) -> int:
    value = document.page_count
    if not isinstance(value, int):
        raise IngestionError("unreadable", "The PDF page count could not be read.")
    return value


def _page_text(page: Any) -> str:
    value = page.get_text()
    if not isinstance(value, str):
        raise IngestionError("unreadable", "A PDF page did not return text.")
    return value
