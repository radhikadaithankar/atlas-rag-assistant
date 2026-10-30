from __future__ import annotations

import pytest
from tests.asgi import call

from atlas.config import AtlasConfigError
from atlas.database.session import session_scope
from atlas.main import create_app


def test_session_scope_without_url_does_not_build_an_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("engine")

    monkeypatch.setattr("atlas.database.session.create_engine", explode)
    with pytest.raises(AtlasConfigError) as caught:
        with session_scope():
            pass
    assert "DATABASE_URL" in str(caught.value)
    assert "psycopg" not in str(caught.value).lower()


def test_ready_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    response = call(create_app(), "GET", "/ready")
    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "database is unreachable",
    }


def test_ready_unreachable_hides_the_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://user:super-secret-pw@127.0.0.1:1/atlas",
    )
    response = call(create_app(), "GET", "/ready")
    assert response.status_code == 503
    assert response.json()["reason"] == "database is unreachable"
    assert "super-secret-pw" not in response.text


def test_importing_database_does_not_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("engine")

    monkeypatch.setattr("atlas.database.session.create_engine", explode)
    import atlas.database as database

    assert database.session_scope is session_scope
