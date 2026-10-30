"""OpenAI chat completions. Timeout 30 seconds and no retries."""

from __future__ import annotations

import logging
import time

from atlas.config import Settings, effective_generation_model, require_openai_api_key
from atlas.generation.base import GenerationResult
from atlas.generation.errors import GenerationProviderError

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 30.0


class OpenAIChat:
    def __init__(self, settings: Settings) -> None:
        key = require_openai_api_key(settings)
        self._model = effective_generation_model(settings)
        from openai import OpenAI

        self._client = OpenAI(api_key=key, timeout=TIMEOUT_SECONDS, max_retries=0)

    def generate(self, system: str, user: str) -> GenerationResult:
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                timeout=TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.error("%s", type(exc).__name__)
            raise GenerationProviderError() from None
        latency_ms = (time.perf_counter() - started) * 1000
        content = response.choices[0].message.content or ""
        usage = response.usage
        prompt = usage.prompt_tokens if usage is not None else None
        completion = usage.completion_tokens if usage is not None else None
        return GenerationResult(content, prompt, completion, latency_ms)
