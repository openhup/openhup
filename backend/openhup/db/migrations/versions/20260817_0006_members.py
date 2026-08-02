"""Consent-gated household members (ADR-016).

People who said yes to being remembered. The privacy machinery is in the shape: an embedding only
exists for a member (consent granted), an unknown face is never stored, and the 24-hour no-reask
marker records *that* an unknown was asked, never *what* they look like. Presence windows are the
room fact that connects identity to episodes - "kitchen occupied 19:40-20:10, Sam" - without ever
saying who did what.

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

JSONB = postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite")

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "members",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column(
            "embedding",
            JSONB,
            nullable=False,
            comment="Face embedding, only ever written with consent",
        ),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("enrolled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("name", name="uq_member_name"),
    )
    op.create_index("ix_members_active", "members", ["active"])

    op.create_table(
        "consent_asks",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("anchor_id", sa.String(64), nullable=False),
        sa.Column("asked_on", sa.Date(), nullable=False),
        sa.Column("answer", sa.String(12), nullable=False, server_default="no"),
        sa.Column("answered_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("anchor_id", "asked_on", name="uq_consent_ask_per_day"),
    )
    op.create_index("ix_consent_asks_anchor_day", "consent_asks", ["anchor_id", "asked_on"])

    op.create_table(
        "presence_windows",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("member_id", sa.String(26), nullable=False),
        sa.Column("anchor_id", sa.String(64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Index("ix_presence_windows_open", "anchor_id", "ended_at"),
        sa.Index("ix_presence_windows_member", "member_id"),
    )


def downgrade() -> None:
    op.drop_table("presence_windows")
    op.drop_table("consent_asks")
    op.drop_table("members")
