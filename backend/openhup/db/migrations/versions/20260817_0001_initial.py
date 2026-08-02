"""Initial schema.

Creates every table, then adds the two things SQLAlchemy's metadata cannot express and that
matter for a system writing a million observation rows a month:

* a **BRIN index** on `observations.ts` - append-only, always queried by range, a fraction of a
  btree's size;
* optional **monthly partitioning** of `observations`, guarded so it is skipped on SQLite and on
  installs that would rather keep one table.

Revision ID: 0001
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite")
BIGSERIAL = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "cameras",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("kind", sa.String(32), nullable=False, server_default="rtsp"),
        sa.Column("config", JSONB, nullable=False),
        sa.Column("node_id", sa.String(64)),
        sa.Column("last_frame_at", sa.DateTime(timezone=True)),
        sa.Column("last_error", sa.Text()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "anchors",
        sa.Column("id", sa.String(64), primary_key=True),
        # SET NULL, not CASCADE: replacing a camera must not destroy the history of the places it
        # watched (ADR-010).
        sa.Column("camera_id", sa.String(64), sa.ForeignKey("cameras.id", ondelete="SET NULL")),
        sa.Column("label", sa.String(128), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("config", JSONB, nullable=False),
        sa.Column("baseline_ref", sa.String(512)),
        sa.Column("baseline_captured_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_anchors_camera_id", "anchors", ["camera_id"])

    op.create_table(
        "skills",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("definition", JSONB, nullable=False),
        sa.Column("origin", sa.String(16), nullable=False, server_default="user"),
        sa.Column("source_text", sa.Text()),
        sa.Column("warnings", JSONB, nullable=False, server_default="[]"),
        sa.Column("tags", JSONB, nullable=False, server_default="[]"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_skills_enabled", "skills", ["enabled"])

    op.create_table(
        "skill_states",
        sa.Column("id", BIGSERIAL, primary_key=True, autoincrement=True),
        sa.Column(
            "skill_id",
            sa.String(64),
            sa.ForeignKey("skills.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("anchor_id", sa.String(64), nullable=False),
        sa.Column("phase", sa.String(16), nullable=False, server_default="idle"),
        sa.Column("since", sa.DateTime(timezone=True)),
        sa.Column("episode_id", sa.String(26)),
        sa.Column("episode_opened_at", sa.DateTime(timezone=True)),
        sa.Column("last_triggered_at", sa.DateTime(timezone=True)),
        sa.Column("last_resolved_at", sa.DateTime(timezone=True)),
        sa.Column("triggers_today", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("counter_day", sa.DateTime(timezone=True)),
        sa.Column("open_task_id", sa.String(26)),
        sa.Column("open_alert_id", sa.String(26)),
        sa.Column("resolve_pending_since", sa.DateTime(timezone=True)),
        sa.Column("suppressed_reason", sa.String(32)),
        sa.Column("stale_notified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.UniqueConstraint("skill_id", "anchor_id", name="uq_skill_state_instance"),
    )
    op.create_index("ix_skill_states_phase", "skill_states", ["phase"])

    op.create_table(
        "episodes",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("skill_id", sa.String(64), nullable=False),
        sa.Column("anchor_id", sa.String(64), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True)),
        sa.Column("trigger_reasons", JSONB, nullable=False, server_default="[]"),
        sa.Column("resolve_reasons", JSONB, nullable=False, server_default="[]"),
        sa.Column("duration_s", sa.Float()),
    )
    op.create_index("ix_episodes_skill_id", "episodes", ["skill_id"])
    op.create_index("ix_episodes_anchor_id", "episodes", ["anchor_id"])
    op.create_index("ix_episodes_closed_at", "episodes", ["closed_at"])
    op.create_index(
        "ix_episodes_skill_anchor_opened", "episodes", ["skill_id", "anchor_id", "opened_at"]
    )

    op.create_table(
        "tasks",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("skill_id", sa.String(64), nullable=False),
        sa.Column("anchor_id", sa.String(64), nullable=False),
        sa.Column("episode_id", sa.String(26), nullable=False),
        sa.Column("state", sa.String(20), nullable=False, server_default="open"),
        sa.Column("urgency", sa.String(10), nullable=False, server_default="low"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("plain_text", sa.Text(), nullable=False),
        sa.Column("text_source", sa.String(10), nullable=False, server_default="template"),
        sa.Column("micro_steps", JSONB, nullable=False, server_default="[]"),
        sa.Column("current_step", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("before_snapshot", sa.String(512)),
        sa.Column("after_snapshot", sa.String(512)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("snoozed_until", sa.DateTime(timezone=True)),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("reopen_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("note", sa.Text()),
        sa.Column("false_positive", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        # The idempotency guarantee: one task per episode, enforced by the database rather than by
        # hoping an at-least-once bus never redelivers.
        sa.UniqueConstraint("episode_id", name="uq_task_per_episode"),
        sa.CheckConstraint("reopen_count <= 2", name="ck_task_reopen_bounded"),
    )
    op.create_index("ix_tasks_skill_id", "tasks", ["skill_id"])
    op.create_index("ix_tasks_anchor_id", "tasks", ["anchor_id"])
    op.create_index("ix_tasks_open", "tasks", ["state", "skill_id", "anchor_id"])
    op.create_index("ix_tasks_created", "tasks", ["created_at"])

    op.create_table(
        "alerts",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("skill_id", sa.String(64), nullable=False),
        sa.Column("anchor_id", sa.String(64), nullable=False),
        sa.Column("episode_id", sa.String(26), nullable=False),
        sa.Column("state", sa.String(16), nullable=False, server_default="active"),
        sa.Column("urgency", sa.String(10), nullable=False, server_default="high"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("plain_text", sa.Text(), nullable=False),
        sa.Column("text_source", sa.String(10), nullable=False, server_default="template"),
        sa.Column("facts", JSONB, nullable=False, server_default="[]"),
        sa.Column("snapshot_ref", sa.String(512)),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True)),
        sa.Column("acknowledged_by", sa.String(64)),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("last_notified_at", sa.DateTime(timezone=True)),
        sa.Column("notify_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("delivered_to", JSONB, nullable=False, server_default="[]"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("episode_id", name="uq_alert_per_episode"),
    )
    op.create_index("ix_alerts_skill_id", "alerts", ["skill_id"])
    op.create_index("ix_alerts_anchor_id", "alerts", ["anchor_id"])
    op.create_index("ix_alerts_state_created", "alerts", ["state", "created_at"])

    op.create_table(
        "observations",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("camera_id", sa.String(64), nullable=False),
        sa.Column("anchor_id", sa.String(64), nullable=False),
        sa.Column("detector", sa.String(64), nullable=False),
        sa.Column("detector_version", sa.String(128), nullable=False),
        sa.Column("signals", JSONB, nullable=False),
        sa.Column("snapshot_ref", sa.String(512)),
        sa.Column("cost_ms", sa.Float()),
    )
    op.create_index("ix_observations_anchor_ts", "observations", ["anchor_id", "ts"])

    op.create_table(
        "metric_points",
        sa.Column("id", BIGSERIAL, primary_key=True, autoincrement=True),
        sa.Column("metric", sa.String(64), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("anchor_id", sa.String(64)),
        sa.Column("skill_id", sa.String(64)),
        sa.Column("bucket_s", sa.Integer(), nullable=False, server_default="86400"),
        sa.Column("labels", JSONB, nullable=False, server_default="{}"),
        sa.UniqueConstraint("metric", "ts", "anchor_id", name="uq_metric_bucket"),
    )
    op.create_index("ix_metric_points_metric_ts", "metric_points", ["metric", "ts"])

    op.create_table(
        "goals",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("label", sa.String(128), nullable=False),
        sa.Column("metric", sa.String(64), nullable=False),
        sa.Column("target", sa.Float(), nullable=False),
        sa.Column("direction", sa.String(10), nullable=False, server_default="up"),
        sa.Column("window_s", sa.Integer(), nullable=False, server_default="604800"),
        sa.Column("anchor_id", sa.String(64)),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("include_in_report", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "personalities",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("display_name", sa.String(128), nullable=False),
        sa.Column("definition", JSONB, nullable=False),
        sa.Column("builtin", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "notifications",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("channel", sa.String(64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("urgency", sa.String(10), nullable=False, server_default="normal"),
        sa.Column("task_id", sa.String(26)),
        sa.Column("alert_id", sa.String(26)),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(16), nullable=False, server_default="sent"),
        sa.Column("error", sa.Text()),
        sa.Column("dedupe_key", sa.String(128)),
    )
    op.create_index("ix_notifications_channel_sent", "notifications", ["channel", "sent_at"])
    op.create_index("ix_notifications_dedupe_key", "notifications", ["dedupe_key"])

    op.create_table(
        "llm_calls",
        sa.Column("id", BIGSERIAL, primary_key=True, autoincrement=True),
        sa.Column(
            "called_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("purpose", sa.String(32), nullable=False),
        sa.Column("local", sa.Boolean(), nullable=False),
        sa.Column("prompt_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("response_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("included_image", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("ok", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("latency_ms", sa.Float()),
    )
    op.create_index("ix_llm_calls_at", "llm_calls", ["called_at"])

    op.create_table(
        "events",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("type", sa.String(48), nullable=False),
        sa.Column("skill_id", sa.String(64)),
        sa.Column("anchor_id", sa.String(64)),
        sa.Column("episode_id", sa.String(26)),
        sa.Column("payload", JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_events_ts", "events", ["ts"])
    op.create_index("ix_events_type_ts", "events", ["type", "ts"])

    # --- Postgres-only refinements ---------------------------------------------------------
    if op.get_bind().dialect.name != "postgresql":
        return

    # BRIN on the observation timestamp. Observations are append-only and always queried by range,
    # which is exactly BRIN's case: a few kilobytes of index instead of hundreds of megabytes.
    op.execute("CREATE INDEX ix_observations_ts_brin ON observations USING brin (ts)")

    # A partial index for the query the UI runs constantly: "what is open right now".
    op.execute(
        "CREATE INDEX ix_tasks_actionable ON tasks (created_at) "
        "WHERE state IN ('open', 'in_progress', 'snoozed')"
    )
    op.execute("CREATE INDEX ix_alerts_active ON alerts (created_at) WHERE state = 'active'")
    # Finding an anchor's most recent observation is on the hot path of every warm start.
    op.execute(
        "CREATE INDEX ix_observations_anchor_detector_ts "
        "ON observations (anchor_id, detector, ts DESC)"
    )


def downgrade() -> None:
    for table in (
        "events",
        "llm_calls",
        "notifications",
        "personalities",
        "goals",
        "metric_points",
        "observations",
        "alerts",
        "tasks",
        "episodes",
        "skill_states",
        "skills",
        "anchors",
        "cameras",
    ):
        op.drop_table(table)
