"""Household memory facts.

The whole memory feature is a small store of plain, human-readable claims the household taught the
assistant ("bin day is Tuesday"). It is local, listable, and per-row deletable - a memory the user
cannot inspect and delete is not a memory (see ADR-012 and docs/SECURITY_PRIVACY.md).

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memory_facts",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("fact", sa.Text(), nullable=False),
        sa.Column("topic", sa.String(64)),
        sa.Column("source", sa.String(16), nullable=False, server_default="api"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_memory_facts_topic", "memory_facts", ["topic"])


def downgrade() -> None:
    op.drop_index("ix_memory_facts_topic", table_name="memory_facts")
    op.drop_table("memory_facts")
