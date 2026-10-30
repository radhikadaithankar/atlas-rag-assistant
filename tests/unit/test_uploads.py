from __future__ import annotations

import pytest
from tests.asgi import call

from atlas.config import load_settings
from atlas.ingestion.parser import parse_bytes
from atlas.ingestion.pipeline import prepare_upload, safe_filename
from atlas.main import create_app


def test_safe_filename_keeps_the_base_name() -> None:
    assert safe_filename("../../tmp/escape/note.txt") == "note.txt"
    assert safe_filename(None) == "upload"


def test_empty_upload_is_422() -> None:
    response = call(
        create_app(),
        "POST",
        "/documents",
        files={"file": ("note.txt", b"", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "empty"


def test_oversized_upload_does_not_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(_data: bytes) -> None:
        raise AssertionError("parsed")

    monkeypatch.setattr("atlas.ingestion.pipeline.parse_bytes", explode)
    settings = load_settings().model_copy(update={"max_upload_bytes": 4})
    response = call(
        create_app(settings=settings),
        "POST",
        "/documents",
        files={"file": ("note.txt", b"12345", "text/plain")},
    )
    assert response.status_code == 413
    assert response.json()["error"] == "too_large"


def test_prepare_upload_rejects_the_same_limit() -> None:
    settings = load_settings().model_copy(update={"max_upload_bytes": 4})
    with pytest.raises(Exception) as caught:
        prepare_upload(b"12345", "note.txt", settings)
    assert getattr(caught.value, "code", "") == "too_large"


def test_bad_bytes_are_422_and_are_not_parsed_as_markdown() -> None:
    response = call(
        create_app(),
        "POST",
        "/documents",
        files={"file": ("note.md", b"\xff\xfe", "text/markdown")},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "unsupported"
    with pytest.raises(Exception):
        parse_bytes(b"\xff\xfe")
