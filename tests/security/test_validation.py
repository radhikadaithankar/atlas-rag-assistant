from __future__ import annotations

import uuid

import pytest

from atlas.config import load_settings
from atlas.main import create_app
from tests.security.http import call


def test_missing_search_and_ask_fields_are_422() -> None:
    app = create_app()
    assert call(app, "POST", "/search", json={}).status_code == 422
    assert call(app, "POST", "/ask", json={}).status_code == 422


def test_wrong_types_bad_mode_and_top_k_are_422() -> None:
    app = create_app()
    assert (
        call(app, "POST", "/search", json={"query": 1, "mode": "keyword"}).status_code
        == 422
    )
    assert (
        call(
            app, "POST", "/search", json={"query": "millet", "mode": "magic"}
        ).status_code
        == 422
    )
    assert (
        call(app, "POST", "/search", json={"query": "millet", "top_k": 0}).status_code
        == 422
    )
    assert (
        call(app, "POST", "/search", json={"query": "millet", "top_k": 51}).status_code
        == 422
    )
    assert (
        call(app, "POST", "/ask", json={"question": "millet", "top_k": 0}).status_code
        == 422
    )
    bad_id = call(
        app,
        "POST",
        "/search",
        json={"query": "millet", "document_id": "not-a-uuid"},
    )
    assert bad_id.status_code == 422
    missing = call(app, "GET", f"/documents/{uuid.uuid4()}")
    # A missing database is 503. A bad path id is 422 before that.
    assert call(app, "GET", "/documents/not-a-uuid").status_code == 422
    assert missing.status_code in {404, 503}


def test_top_k_above_the_rerank_pool_is_invalid_request() -> None:
    app = create_app()
    search = call(
        app,
        "POST",
        "/search",
        json={"query": "millet", "mode": "keyword", "rerank": True, "top_k": 21},
    )
    ask = call(
        app,
        "POST",
        "/ask",
        json={"question": "millet", "mode": "keyword", "rerank": True, "top_k": 21},
    )
    assert search.status_code == 422
    assert search.json()["error"] == "invalid_request"
    assert ask.status_code == 422
    assert ask.json()["error"] == "invalid_request"


def test_empty_upload_is_422_empty() -> None:
    response = call(
        create_app(),
        "POST",
        "/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "empty"


def test_oversized_upload_is_413_and_does_not_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(_data: bytes) -> None:
        raise AssertionError("parsed")

    monkeypatch.setattr("atlas.ingestion.pipeline.parse_bytes", explode)
    settings = load_settings().model_copy(update={"max_upload_bytes": 4})
    response = call(
        create_app(settings=settings),
        "POST",
        "/documents",
        files={"file": ("big.txt", b"12345", "text/plain")},
    )
    assert response.status_code == 413
    assert response.json()["error"] == "too_large"
