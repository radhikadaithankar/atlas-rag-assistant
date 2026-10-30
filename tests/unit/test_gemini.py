from __future__ import annotations

import pytest

from atlas.generation.gemini_chat import _token_counts, _with_thinking_fallback
from atlas.providers.quota import (
    advertised_delay_seconds,
    is_thinking_budget_error,
    run_with_quota_retries,
)


class QuotaError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = 429


class ThinkingError(Exception):
    def __init__(self) -> None:
        super().__init__("thinking_budget is not supported")
        self.code = 400


def test_short_429_retries_and_long_delays_do_not() -> None:
    calls = {"n": 0}

    def succeed_on_third() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise QuotaError("Please retry in 1s")
        return "ok"

    slept: list[float] = []
    assert run_with_quota_retries(succeed_on_third, sleep=slept.append) == "ok"
    assert slept == [1.0, 1.0]

    def hours() -> None:
        raise QuotaError("retry of about 22 hours")

    with pytest.raises(QuotaError):
        run_with_quota_retries(
            hours, sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError)
        )

    def too_long() -> None:
        raise QuotaError("Please retry in 90s")

    with pytest.raises(QuotaError):
        run_with_quota_retries(
            too_long, sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError)
        )
    assert advertised_delay_seconds(QuotaError("Please retry in 90s")) is None
    assert advertised_delay_seconds(QuotaError("Please retry in 12s")) == 12.0


def test_quota_retry_stops_after_eight() -> None:
    calls = {"n": 0}

    def always() -> None:
        calls["n"] += 1
        raise QuotaError("Please retry in 1s")

    with pytest.raises(QuotaError):
        run_with_quota_retries(always, sleep=lambda _seconds: None)
    assert calls["n"] == 9


def test_thinking_budget_retries_once_without_config() -> None:
    class Models:
        def generate_content(self, **kwargs: object) -> str:
            config = kwargs["config"]
            thinking = getattr(config, "thinking_config", None)
            if thinking is not None:
                raise ThinkingError()
            return "ok"

    class Client:
        models = Models()

    assert is_thinking_budget_error(ThinkingError())
    assert (
        _with_thinking_fallback(Client(), "gemini-3.5-flash-lite", "system", "user")
        == "ok"
    )


def test_thought_tokens_are_added_only_when_present() -> None:
    class Usage:
        prompt_token_count = 10
        candidates_token_count = 4

    class Response:
        usage_metadata = Usage()

    assert _token_counts(Response()) == (10, 4)
    Usage.thoughts_token_count = 3  # type: ignore[attr-defined]
    assert _token_counts(Response()) == (10, 7)
