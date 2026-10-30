"""Schema and readiness. Skips when DATABASE_URL is unset."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from tests.asgi import call

from atlas.database.models import Chunk, Document
from atlas.database.session import session_scope
from atlas.main import create_app

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL is unset",
)


def test_migration_tables_and_extension() -> None:
    with session_scope() as session:
        names = set(
            session.scalars(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            ).all()
        )
        assert {"documents", "chunks", "alembic_version"} <= names
        extension = session.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        assert extension
        column = session.scalar(
            text(
                "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                "JOIN pg_class c ON c.oid = a.attrelid "
                "WHERE c.relname = 'chunks' AND a.attname = 'embedding'"
            )
        )
        assert column == "vector(1536)"


def test_round_trip_and_cascade() -> None:
    document_id = uuid.uuid4()
    now = datetime.now(UTC)
    with session_scope() as session:
        session.add(
            Document(
                id=document_id,
                filename="round.txt",
                content_sha256=uuid.uuid4().hex + uuid.uuid4().hex[:32],
                source_format="txt",
                created_at=now,
            )
        )
        session.add(
            Chunk(
                id=uuid.uuid4(),
                document_id=document_id,
                chunk_index=0,
                text="first",
                page=None,
                section=None,
                embedding=None,
                created_at=now,
            )
        )
        session.add(
            Chunk(
                id=uuid.uuid4(),
                document_id=document_id,
                chunk_index=1,
                text="second",
                page=None,
                section=None,
                embedding=None,
                created_at=now,
            )
        )
    with session_scope() as session:
        session.execute(
            text("DELETE FROM documents WHERE id = :id"), {"id": document_id}
        )
    with session_scope() as session:
        remaining = session.scalar(
            text("SELECT count(*) FROM chunks WHERE document_id = :id"),
            {"id": document_id},
        )
        assert int(remaining or 0) == 0


def test_ready_returns_200() -> None:
    response = call(create_app(), "GET", "/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
