"""Build a chat client only after the provider key has been checked."""

from __future__ import annotations

from atlas.config import MISSING_PROVIDER_MESSAGE, Settings, select_provider
from atlas.generation.base import Generator
from atlas.generation.errors import UnavailableError


def build_generator(settings: Settings) -> Generator:
    provider = select_provider(settings)
    if provider == "none":
        raise UnavailableError(MISSING_PROVIDER_MESSAGE)
    if provider == "openai":
        from atlas.generation.openai_chat import OpenAIChat

        return OpenAIChat(settings)
    from atlas.generation.gemini_chat import GeminiChat

    return GeminiChat(settings)
