"""Win milestones (ADR-015).

The ledger of progress the assistant has noticed: an anchor staying clear for whole days, once
per band or per 90-day record. The unique constraint is the dedupe - a milestone is celebrated
when it first happens and never again.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "win_milestones",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("anchor_id", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(24), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("days", sa.Float(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("spoken_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("anchor_id", "kind", "value", name="uq_win_milestone"),
    )
    op.create_index("ix_win_milestones_anchor", "win_milestones", ["anchor_id"])


def downgrade() -> None:
    op.drop_index("ix_win_milestones_anchor", table_name="win_milestones")
    op.drop_table("win_milestones")
