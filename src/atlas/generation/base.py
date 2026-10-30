"""One chat completion. The protocol does not decide citations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float


class Generator(Protocol):
    def generate(self, system: str, user: str) -> GenerationResult:
        """Return the model text and token counts. Do not log the prompt."""
