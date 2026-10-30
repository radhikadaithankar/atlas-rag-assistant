"""Overlapping character windows.

PDF pages are windowed separately so a window does not cross a page. Markdown
windows carry the nearest ATX heading at or before the window start. The short
tail is kept. Whitespace-only windows are dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from atlas.ingestion.parser import ParsedDocument

ATX_LINE = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)\s*$")


@dataclass(frozen=True)
class TextChunk:
    text: str
    page: int | None
    section: str | None


def chunk_parsed(parsed: ParsedDocument, *, size: int, overlap: int) -> list[TextChunk]:
    if parsed.source_format == "pdf":
        chunks: list[TextChunk] = []
        for page in parsed.pages:
            for window in _windows(page.text, size=size, overlap=overlap):
                chunks.append(TextChunk(window, page.page_number, None))
        return chunks
    if parsed.source_format == "markdown":
        text = parsed.pages[0].text if parsed.pages else ""
        return _markdown_windows(text, size=size, overlap=overlap)
    chunks = []
    for page in parsed.pages:
        for window in _windows(page.text, size=size, overlap=overlap):
            chunks.append(TextChunk(window, None, None))
    return chunks


def _windows(text: str, *, size: int, overlap: int) -> list[str]:
    spans = _spans(text, size=size, overlap=overlap)
    windows: list[str] = []
    for start, end in spans:
        window = text[start:end]
        if window.strip():
            windows.append(window)
    return windows


def _spans(text: str, *, size: int, overlap: int) -> list[tuple[int, int]]:
    if text == "":
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    step = size - overlap
    while start < length:
        end = min(start + size, length)
        spans.append((start, end))
        if end == length:
            break
        start += step
    return spans


def _markdown_windows(text: str, *, size: int, overlap: int) -> list[TextChunk]:
    headings = [
        (match.start(), match.group(2).strip()) for match in ATX_LINE.finditer(text)
    ]
    chunks: list[TextChunk] = []
    for start, end in _spans(text, size=size, overlap=overlap):
        window = text[start:end]
        if not window.strip():
            continue
        section = _section_at(headings, start)
        chunks.append(TextChunk(window, None, section))
    return chunks


def _section_at(headings: list[tuple[int, str]], start: int) -> str | None:
    current: str | None = None
    for position, title in headings:
        if position <= start:
            current = title
        else:
            break
    return current
