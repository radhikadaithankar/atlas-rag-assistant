from __future__ import annotations

import logging

import pytest

from atlas.serve import main


def test_missing_database_exits_and_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(
        "uvicorn.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("socket")),
    )
    caplog.set_level(logging.INFO)
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 1
    assert "DATABASE_URL is not set. It is required to start the API." in caplog.text
    assert "keyword search boots without it." in caplog.text


def test_invalid_url_omits_the_value(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "mysql://user:hide-me-please@localhost/atlas")
    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 1
    assert "hide-me-please" not in caplog.text
    assert "DATABASE_URL" in caplog.text


def test_boots_without_a_provider_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/atlas"
    )
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)
    caplog.set_level(logging.INFO)
    main()
    assert "OPENAI_API_KEY is not set. Keyword search is available." in caplog.text
    assert "ask return 503." in caplog.text
