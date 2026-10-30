"""Gemini embeddings at output dimensionality 1536."""

from __future__ import annotations

import logging
from typing import Any

from atlas.config import (
    EMBEDDING_WIDTH,
    GEMINI_EMBEDDING_MODEL,
    Settings,
    effective_embedding_model,
)
from atlas.embeddings.errors import EmbeddingProviderError
from atlas.embeddings.vectors import require_width
from atlas.providers.quota import run_with_quota_retries

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise EmbeddingProviderError()
        self._model = effective_embedding_model(settings) or GEMINI_EMBEDDING_MODEL
        self._client: Any = _client(settings.gemini_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = run_with_quota_retries(lambda: self._embed_once(texts))
            vectors = _values(response)
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            logger.error("%s", type(exc).__name__)
            raise EmbeddingProviderError() from None
        return require_width(vectors)

    def _embed_once(self, texts: list[str]) -> object:
        from google.genai import types

        return self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_WIDTH),
        )


def _client(api_key: str) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30_000),
    )


def _values(response: object) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None)
    if not isinstance(embeddings, list):
        raise EmbeddingProviderError()
    vectors: list[list[float]] = []
    for item in embeddings:
        values = getattr(item, "values", None)
        if not isinstance(values, list):
            raise EmbeddingProviderError()
        vectors.append([float(value) for value in values])
    return vectors
