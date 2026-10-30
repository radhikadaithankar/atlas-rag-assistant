from __future__ import annotations

import fitz
import pytest

from atlas.ingestion.errors import IngestionError
from atlas.ingestion.parser import parse_bytes


def _pdf(pages: list[str], *, password: str | None = None) -> bytes:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    if password is None:
        data = document.tobytes()
    else:
        data = document.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw=password,
            user_pw=password,
        )
    document.close()
    return bytes(data)


def test_pdf_magic_keeps_page_numbers_including_a_blank_page() -> None:
    parsed = parse_bytes(_pdf(["Page one.", ""]))
    assert parsed.source_format == "pdf"
    assert [page.page_number for page in parsed.pages] == [1, 2]
    assert parsed.pages[0].text
    assert parsed.pages[1].text.strip() == ""


def test_filename_is_not_consulted() -> None:
    parsed = parse_bytes(b"# Title\n\nBody.\n")
    assert parsed.source_format == "markdown"


def test_utf8_without_atx_is_text() -> None:
    parsed = parse_bytes(b"Just a sentence.\n")
    assert parsed.source_format == "txt"


def test_empty_file() -> None:
    with pytest.raises(IngestionError) as caught:
        parse_bytes(b"")
    assert caught.value.code == "empty"


def test_whitespace_only_text_is_empty() -> None:
    with pytest.raises(IngestionError) as caught:
        parse_bytes(b"   \n\t")
    assert caught.value.code == "empty"


def test_undecodable_bytes() -> None:
    with pytest.raises(IngestionError) as caught:
        parse_bytes(b"\xff\xfe not text")
    assert caught.value.code == "unsupported"


def test_encrypted_pdf_is_unreadable() -> None:
    with pytest.raises(IngestionError) as caught:
        parse_bytes(_pdf(["secret"], password="owner"))
    assert caught.value.code == "unreadable"
