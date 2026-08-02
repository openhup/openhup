"""The personality gamble (ADR-014).

One row, drawn at first setup when `personality.gamble` is on, holding the mystery voice until
the household reveals or re-draws it in Settings.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "personality_draw",
        sa.Column("id", sa.String(16), primary_key=True),
        sa.Column("personality_id", sa.String(64), nullable=False),
        sa.Column("reroll_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("personality_draw")
