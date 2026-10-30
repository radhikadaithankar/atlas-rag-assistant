"""Choose an embedder. A missing key raises before a client exists."""

from __future__ import annotations

from atlas.config import Settings, select_provider
from atlas.embeddings.base import Embedder
from atlas.embeddings.errors import EmbeddingProviderError


def build_embedder(settings: Settings) -> Embedder:
    provider = select_provider(settings)
    if provider == "openai":
        from atlas.embeddings.openai_provider import OpenAIEmbedder

        return OpenAIEmbedder(settings)
    if provider == "gemini":
        from atlas.embeddings.gemini_provider import GeminiEmbedder

        return GeminiEmbedder(settings)
    raise EmbeddingProviderError("the embedding provider failed")
