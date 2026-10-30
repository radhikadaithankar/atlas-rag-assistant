from __future__ import annotations

import json
import uuid
from collections.abc import Callable

import pytest
from tests.asgi import call

from atlas.generation.base import GenerationResult
from atlas.main import create_app
from atlas.retrieval.hits import Hit


def _hit(number: int, text: str) -> Hit:
    identifier = uuid.UUID(int=number)
    return Hit(
        chunk_id=identifier,
        document_id=identifier,
        filename=f"{number}.txt",
        page=None,
        section=None,
        text=text,
        score=float(number),
        keyword_rank=number,
    )


def test_rerank_false_does_not_call_the_reranker() -> None:
    def retrieve(**_kwargs: object) -> list[Hit]:
        return [_hit(1, "alpha")]

    def rerank(_query: str, _passages: list[Hit]) -> list[Hit]:
        raise AssertionError("reranked")

    response = call(
        create_app(retriever=retrieve, reranker=rerank),  # type: ignore[arg-type]
        "POST",
        "/search",
        json={"query": "alpha", "mode": "keyword", "rerank": False},
    )
    assert response.status_code == 200
    assert response.json()["hits"][0]["text"] == "alpha"


class _FnReranker:
    def __init__(self, fn: Callable[[str, list[Hit]], list[Hit]]) -> None:
        self._fn = fn

    def rerank(self, query: str, passages: list[Hit]) -> list[Hit]:
        return self._fn(query, passages)


def test_rerank_reverses_order_and_keeps_keyword_rank() -> None:
    def retrieve(**kwargs: object) -> list[Hit]:
        assert kwargs["top_k"] == 20
        return [_hit(1, "alpha"), _hit(2, "beta")]

    def rerank(_query: str, passages: list[Hit]) -> list[Hit]:
        ordered = list(reversed(passages))
        for index, passage in enumerate(ordered, start=1):
            passage.score = float(index)
        return ordered

    response = call(
        create_app(retriever=retrieve, reranker=_FnReranker(rerank)),
        "POST",
        "/search",
        json={"query": "beta", "mode": "keyword", "rerank": True, "top_k": 5},
    )
    hits = response.json()["hits"]
    assert [hit["text"] for hit in hits] == ["beta", "alpha"]
    assert hits[0]["score"] == 1.0
    assert hits[0]["keyword_rank"] == 2
    assert hits[1]["keyword_rank"] == 1


def test_top_k_above_the_pool_is_422() -> None:
    response = call(
        create_app(),
        "POST",
        "/search",
        json={"query": "millet", "mode": "keyword", "rerank": True, "top_k": 21},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    ask = call(
        create_app(),
        "POST",
        "/ask",
        json={"question": "millet", "mode": "keyword", "rerank": True, "top_k": 21},
    )
    assert ask.status_code == 422
    assert ask.json()["error"] == "invalid_request"


def test_missing_extra_is_503_and_ask_does_not_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("RERANKER", "cross_encoder")

    def missing() -> None:
        raise ImportError("sentence_transformers")

    monkeypatch.setattr("atlas.api.deps.load_cross_encoder", missing)

    def generate(_system: str, _user: str) -> GenerationResult:
        raise AssertionError("generated")

    class Generator:
        def generate(self, system: str, user: str) -> GenerationResult:
            return generate(system, user)

    def retrieve(**_kwargs: object) -> list[Hit]:
        raise AssertionError("retrieved")

    app = create_app(retriever=retrieve, generator=Generator())
    search = call(
        app,
        "POST",
        "/search",
        json={"query": "millet", "mode": "keyword", "rerank": True},
    )
    ask = call(
        app,
        "POST",
        "/ask",
        json={"question": "millet", "mode": "keyword", "rerank": True},
    )
    assert search.status_code == 503
    assert ask.status_code == 503
    assert search.json()["error"] == "unavailable"
    assert ask.json()["error"] == "unavailable"


def test_empty_rerank_abstains_without_the_generator() -> None:
    def retrieve(**_kwargs: object) -> list[Hit]:
        return [_hit(1, "alpha")]

    def rerank(_query: str, _passages: list[Hit]) -> list[Hit]:
        return []

    def generate(_system: str, _user: str) -> GenerationResult:
        raise AssertionError("generated")

    class Generator:
        def generate(self, system: str, user: str) -> GenerationResult:
            return generate(system, user)

    response = call(
        create_app(
            retriever=retrieve, reranker=_FnReranker(rerank), generator=Generator()
        ),
        "POST",
        "/ask",
        json={"question": "alpha", "mode": "keyword", "rerank": True},
    )
    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["citations"] == []


def test_citation_for_a_cut_chunk_is_dropped() -> None:
    kept = _hit(1, "The sack holds millet.")
    cut = _hit(2, "The other sack holds rice.")

    def retrieve(**_kwargs: object) -> list[Hit]:
        return [kept, cut]

    def rerank(_query: str, _passages: list[Hit]) -> list[Hit]:
        return [kept]

    class Generator:
        def generate(self, _system: str, _user: str) -> GenerationResult:
            return GenerationResult(
                json.dumps(
                    {
                        "answer": "Rice.",
                        "abstained": False,
                        "citations": [{"chunk_id": str(cut.chunk_id), "quote": "rice"}],
                    }
                ),
                1,
                1,
                5.0,
            )

    response = call(
        create_app(
            retriever=retrieve, reranker=_FnReranker(rerank), generator=Generator()
        ),
        "POST",
        "/ask",
        json={"question": "rice", "mode": "keyword", "rerank": True},
    )
    body = response.json()
    assert body["abstained"] is True
    assert str(cut.document_id) not in response.text
