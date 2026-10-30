from __future__ import annotations

import uuid

import pytest
from tests.asgi import call

from atlas.config import load_settings, select_provider
from atlas.embeddings.errors import EmbeddingShapeError
from atlas.embeddings.vectors import require_width
from atlas.main import create_app


def test_shape_check_rejects_the_wrong_length() -> None:
    with pytest.raises(EmbeddingShapeError):
        require_width([[0.1, 0.2]])


def test_provider_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    assert select_provider(load_settings()) == "openai"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert select_provider(load_settings()) == "gemini"
    monkeypatch.delenv("GEMINI_API_KEY")
    assert select_provider(load_settings()) == "none"


def test_vector_search_without_a_key_is_503_and_does_not_retrieve() -> None:
    called = False

    def retrieve(**_kwargs: object) -> list[object]:
        nonlocal called
        called = True
        return []

    app = create_app(retriever=retrieve)
    for mode in ("vector", "hybrid"):
        response = call(app, "POST", "/search", json={"query": "millet", "mode": mode})
        assert response.status_code == 503
        assert response.json()["error"] == "unavailable"
        assert "OPENAI_API_KEY" in response.json()["message"]
    assert called is False


def test_keyword_search_without_a_key_can_return_200() -> None:
    def retrieve(**_kwargs: object) -> list[object]:
        return []

    response = call(
        create_app(retriever=retrieve),
        "POST",
        "/search",
        json={"query": "millet", "mode": "keyword"},
    )
    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_embed_without_a_key_is_503_before_the_database() -> None:
    response = call(
        create_app(),
        "POST",
        f"/documents/{uuid.uuid4()}/embed",
    )
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "unavailable"
    assert "OPENAI_API_KEY" in body["message"]


def test_fake_embedder_failure_is_502_without_the_secret() -> None:
    class Boom:
        def embed(self, _texts: list[str]) -> list[list[float]]:
            raise RuntimeError("sk-secret passage-token")

    response = call(
        create_app(embedder=Boom()),
        "POST",
        "/search",
        json={"query": "millet", "mode": "vector"},
    )
    assert response.status_code == 502
    assert response.json() == {
        "error": "provider",
        "message": "the embedding provider failed",
    }
    assert "sk-secret" not in response.text
    assert "passage-token" not in response.text


def test_ask_modes_without_a_key_are_503(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("socket")

    monkeypatch.setattr("socket.create_connection", explode)
    monkeypatch.setattr("openai.OpenAI", explode)
    monkeypatch.setattr("atlas.database.session.create_engine", explode)
    app = create_app()
    for mode in ("vector", "keyword", "hybrid"):
        response = call(app, "POST", "/ask", json={"question": "millet", "mode": mode})
        assert response.status_code == 503
        assert response.json()["error"] == "unavailable"
        assert "OPENAI_API_KEY" in response.json()["message"]
