"""Gemini JSON generation.

``thinking_budget`` 0 is rejected by gemini-3.5-flash-lite, so that failure
retries the same call once without ``thinking_config``. Short 429 responses
use the shared quota policy. Completion tokens add ``thoughts_token_count``
only when that field is an int.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from atlas.config import Settings, effective_generation_model
from atlas.generation.base import GenerationResult
from atlas.generation.errors import GenerationProviderError
from atlas.providers.quota import is_thinking_budget_error, run_with_quota_retries

logger = logging.getLogger(__name__)


class GeminiChat:
    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise GenerationProviderError()
        self._model = effective_generation_model(settings)
        self._client: Any = _client(settings.gemini_api_key)

    def generate(self, system: str, user: str) -> GenerationResult:
        started = time.perf_counter()
        try:
            response = run_with_quota_retries(
                lambda: _with_thinking_fallback(self._client, self._model, system, user)
            )
        except GenerationProviderError:
            raise
        except Exception as exc:
            logger.error("%s", type(exc).__name__)
            raise GenerationProviderError() from None
        latency_ms = (time.perf_counter() - started) * 1000
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            text = ""
        prompt, completion = _token_counts(response)
        return GenerationResult(text, prompt, completion, latency_ms)


def _with_thinking_fallback(client: Any, model: str, system: str, user: str) -> Any:
    try:
        return _generate(client, model, system, user, thinking_budget=0)
    except Exception as exc:
        if not is_thinking_budget_error(exc):
            raise
        logger.warning(
            "generation with thinking_budget 0 failed (%s)", type(exc).__name__
        )
        return _generate(client, model, system, user, thinking_budget=None)


def _generate(
    client: Any,
    model: str,
    system: str,
    user: str,
    *,
    thinking_budget: int | None,
) -> Any:
    from google.genai import types

    config_kwargs: dict[str, Any] = {
        "system_instruction": system,
        "response_mime_type": "application/json",
    }
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )
    return client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(**config_kwargs),
    )


def _token_counts(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    prompt = getattr(usage, "prompt_token_count", None)
    candidates = getattr(usage, "candidates_token_count", None)
    thoughts = getattr(usage, "thoughts_token_count", None)
    prompt_tokens = prompt if isinstance(prompt, int) else None
    if not isinstance(candidates, int):
        return prompt_tokens, None
    completion = candidates
    if isinstance(thoughts, int):
        completion += thoughts
    return prompt_tokens, completion


def _client(api_key: str) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))
