from __future__ import annotations

import httpx
from tests.asgi import SyncASGITransport

from atlas.main import create_app
from atlas.ui.client import (
    AtlasClient,
    AtlasClientError,
    ask_error_text,
    chunk_label,
    format_place,
)


def _client(app: object) -> AtlasClient:
    http = httpx.Client(transport=SyncASGITransport(app), base_url="http://test")  # type: ignore[arg-type]
    return AtlasClient(client=http)


def test_chunk_label_and_citation_place() -> None:
    assert chunk_label(1) == "1 chunk"
    assert chunk_label(2) == "2 chunks"
    assert format_place("handbook.txt", 4, "Linen") == "handbook.txt, page 4"
    assert format_place("note.md", None, "Millet") == "note.md, section Millet"


def test_plain_language_errors() -> None:
    assert "Ask needs a key" in ask_error_text(503, "OPENAI_API_KEY is not set.")
    assert ask_error_text(409, "duplicate") == "This file is already stored."
    assert (
        ask_error_text(413, "too large") == "This file is larger than the upload limit."
    )
    assert ask_error_text(422, "The file is empty.") == "The file is empty."


def test_client_health_and_missing_key_against_the_app() -> None:
    client = _client(create_app())
    assert client.health() == {"status": "ok"}
    with pytest_raises_client() as caught:
        client.ask("millet", mode="keyword")
    assert caught.status == 503
    assert "OPENAI_API_KEY" in caught.message
    text = ask_error_text(caught.status, caught.message)
    assert text == (
        "The API is missing OPENAI_API_KEY (503). Ask needs a key. Keyword search does not."
    )


def test_empty_upload_message() -> None:
    client = _client(create_app())
    try:
        client.upload("empty.txt", b"")
    except AtlasClientError as exc:
        assert exc.status == 422
        assert ask_error_text(exc.status, exc.message) == "The file is empty."
    else:
        raise AssertionError("expected an empty-file error")


class _Caught:
    def __init__(self) -> None:
        self.status = 0
        self.message = ""

    def __enter__(self) -> _Caught:
        return self

    def __exit__(
        self, exc_type: object, exc: BaseException | None, _tb: object
    ) -> bool:
        if isinstance(exc, AtlasClientError):
            self.status = exc.status
            self.message = exc.message
            return True
        return False


def pytest_raises_client() -> _Caught:
    return _Caught()
