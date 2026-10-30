"""chunk english tsvector

Revision ID: b8e4c1a27f30
Revises: f6b2a1c09e47
Create Date: 2026-10-22
"""

from __future__ import annotations

from alembic import op

revision = "b8e4c1a27f30"
down_revision = "f6b2a1c09e47"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE chunks ADD COLUMN search_vector tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
    )
    op.execute(
        "CREATE INDEX ix_chunks_search_vector ON chunks USING gin (search_vector)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_search_vector")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS search_vector")
