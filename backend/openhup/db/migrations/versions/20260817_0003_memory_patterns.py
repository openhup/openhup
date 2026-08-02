"""Learned memory patterns.

Derived claims about the household's own episode history - "the kitchen counter usually needs
attention about every 3 days" - stored so they can be reviewed, dismissed, and nudged. Same
reviewability bar as `memory_facts` (ADR-013).

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


JSONB = postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "memory_patterns",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("skill_id", sa.String(64), nullable=False),
        sa.Column("anchor_id", sa.String(64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("evidence", JSONB, nullable=False, server_default="{}"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(12), nullable=False, server_default="active"),
        sa.Column("last_nudge_at", sa.DateTime(timezone=True)),
        sa.Column("last_nudge_basis", sa.String(26)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("kind", "skill_id", "anchor_id", name="uq_memory_pattern"),
    )
    op.create_index("ix_memory_patterns_status", "memory_patterns", ["status"])


def downgrade() -> None:
    op.drop_index("ix_memory_patterns_status", table_name="memory_patterns")
    op.drop_table("memory_patterns")
