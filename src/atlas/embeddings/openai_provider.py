"""OpenAI embeddings. Timeout 30 seconds, no retries, float vectors."""

from __future__ import annotations

import logging

from atlas.config import Settings, effective_embedding_model, require_openai_api_key
from atlas.embeddings.errors import EmbeddingProviderError
from atlas.embeddings.vectors import require_width

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 30.0


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        key = require_openai_api_key(settings)
        self._model = effective_embedding_model(settings)
        from openai import OpenAI

        self._client = OpenAI(api_key=key, timeout=TIMEOUT_SECONDS, max_retries=0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=texts,
                encoding_format="float",
                timeout=TIMEOUT_SECONDS,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors = [list(item.embedding) for item in ordered]
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            logger.error("%s", type(exc).__name__)
            raise EmbeddingProviderError() from None
        return require_width(vectors)
