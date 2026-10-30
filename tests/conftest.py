"""Keep provider keys out of unit tests unless a test sets them."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("GENERATION_MODEL", raising=False)
    monkeypatch.delenv("RERANKER", raising=False)
