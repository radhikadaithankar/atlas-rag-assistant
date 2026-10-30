from __future__ import annotations

from atlas.ingestion.chunker import chunk_parsed
from atlas.ingestion.parser import ParsedDocument, ParsedPage


def test_tail_is_kept() -> None:
    parsed = ParsedDocument("txt", [ParsedPage(None, "abcdefghij")])
    chunks = chunk_parsed(parsed, size=4, overlap=0)
    assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ij"]


def test_whitespace_window_is_dropped() -> None:
    parsed = ParsedDocument("txt", [ParsedPage(None, "abcd    ")])
    chunks = chunk_parsed(parsed, size=4, overlap=0)
    assert [chunk.text for chunk in chunks] == ["abcd"]


def test_pdf_windows_do_not_cross_pages() -> None:
    parsed = ParsedDocument(
        "pdf",
        [ParsedPage(1, "aaaa"), ParsedPage(2, "bbbb")],
    )
    chunks = chunk_parsed(parsed, size=6, overlap=0)
    assert [(chunk.page, chunk.text) for chunk in chunks] == [(1, "aaaa"), (2, "bbbb")]


def test_blank_pdf_page_emits_no_chunk() -> None:
    parsed = ParsedDocument("pdf", [ParsedPage(1, "   "), ParsedPage(2, "kept")])
    chunks = chunk_parsed(parsed, size=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].page == 2


def test_markdown_section_is_the_heading_at_or_before_the_window() -> None:
    text = "before\n# Millet\nThe store keeps millet.\n"
    parsed = ParsedDocument("markdown", [ParsedPage(None, text)])
    chunks = chunk_parsed(parsed, size=8, overlap=0)
    assert chunks[0].section is None
    assert any(chunk.section == "Millet" for chunk in chunks[1:])


def test_setext_heading_is_not_a_section() -> None:
    text = "Millet\n======\nThe store keeps millet.\n"
    parsed = ParsedDocument("txt", [ParsedPage(None, text)])
    chunks = chunk_parsed(parsed, size=100, overlap=0)
    assert chunks[0].section is None
