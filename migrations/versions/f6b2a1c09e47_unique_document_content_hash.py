"""unique document content hash

Revision ID: f6b2a1c09e47
Revises: 5c811c0524ef
Create Date: 2026-10-22
"""

from __future__ import annotations

from alembic import op

revision = "f6b2a1c09e47"
down_revision = "5c811c0524ef"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_documents_content_sha256",
        "documents",
        ["content_sha256"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_documents_content_sha256", "documents", type_="unique")
