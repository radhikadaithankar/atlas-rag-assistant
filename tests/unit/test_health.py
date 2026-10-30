from __future__ import annotations

import pytest
from tests.asgi import call

from atlas.config import (
    AtlasConfigError,
    Settings,
    load_settings,
    require_database_url,
    require_openai_api_key,
    to_psycopg_url,
)
from atlas.main import create_app


def test_health_without_configuration() -> None:
    response = call(create_app(), "GET", "/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_does_not_open_the_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/atlas"
    )

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("engine")

    monkeypatch.setattr("atlas.database.session.create_engine", explode)
    response = call(create_app(), "GET", "/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_whitespace_in_key_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk test")
    with pytest.raises(AtlasConfigError) as caught:
        load_settings()
    assert caught.value.__cause__ is None
    assert "sk test" not in str(caught.value)
    assert "OPENAI_API_KEY" in str(caught.value)


def test_blank_key_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    settings = load_settings()
    assert settings.openai_api_key is None


def test_non_postgres_url_hides_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "mysql://user:secret-password@localhost/atlas")
    with pytest.raises(AtlasConfigError) as caught:
        load_settings()
    assert caught.value.__cause__ is None
    assert "secret-password" not in str(caught.value)
    assert "DATABASE_URL" in str(caught.value)


def test_postgres_url_without_hostname_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret-password@/atlas")
    with pytest.raises(AtlasConfigError) as caught:
        load_settings()
    assert "secret-password" not in str(caught.value)
    assert "hostname" in str(caught.value)


def test_embedding_width_requires_a_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
    with pytest.raises(AtlasConfigError) as caught:
        load_settings()
    assert "migration" in str(caught.value)


def test_known_wide_model_requires_a_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    with pytest.raises(AtlasConfigError) as caught:
        load_settings()
    assert "3072" in str(caught.value)
    assert "migration" in str(caught.value)


def test_overlap_must_be_smaller_than_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "1000")
    with pytest.raises(AtlasConfigError) as caught:
        load_settings()
    assert "CHUNK_OVERLAP" in str(caught.value)


def test_repr_hides_secrets() -> None:
    settings = Settings(
        openai_api_key="sk-secret",
        gemini_api_key=None,
        database_url="postgresql://user:secret@localhost/atlas",
    )
    text = repr(settings)
    assert "set" in text
    assert "unset" in text
    assert "sk-secret" not in text
    assert "secret@" not in text


def test_require_helpers_name_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings = load_settings()
    with pytest.raises(AtlasConfigError) as caught:
        require_database_url(settings)
    assert "DATABASE_URL" in str(caught.value)
    with pytest.raises(AtlasConfigError) as caught:
        require_openai_api_key(settings)
    assert "OPENAI_API_KEY" in str(caught.value)


def test_psycopg_dialect_rewrite() -> None:
    assert (
        to_psycopg_url("postgresql://localhost/atlas")
        == "postgresql+psycopg://localhost/atlas"
    )
    assert (
        to_psycopg_url("postgres://localhost/atlas")
        == "postgresql+psycopg://localhost/atlas"
    )
    assert to_psycopg_url("postgresql+psycopg://localhost/atlas").startswith(
        "postgresql+psycopg://"
    )
