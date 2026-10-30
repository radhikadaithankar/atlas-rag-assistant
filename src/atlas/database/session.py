"""Engine and session helpers.

The engine is created only when a caller asks for a session or a readiness
check. The URL is not logged. Driver failures are not chained into HTTP bodies.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from atlas.config import (
    AtlasConfigError,
    Settings,
    load_settings,
    require_database_url,
    to_psycopg_url,
)

CONNECT_TIMEOUT_SECONDS = 3


def make_engine(url: str) -> Engine:
    return create_engine(
        to_psycopg_url(url),
        pool_pre_ping=True,
        connect_args={"connect_timeout": CONNECT_TIMEOUT_SECONDS},
    )


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """Borrow an engine, yield one session, then close and dispose."""

    active = settings if settings is not None else load_settings()
    url = require_database_url(active)
    engine = make_engine(url)
    try:
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
    finally:
        engine.dispose()


def database_accepts_query(settings: Settings) -> bool:
    """Run ``SELECT 1``. Failures stay boolean; the caller logs the type name."""

    try:
        url = require_database_url(settings)
    except AtlasConfigError:
        return False
    engine = make_engine(url)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    finally:
        engine.dispose()
