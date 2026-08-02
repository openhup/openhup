"""SQLAlchemy 2.0 models.

Shape notes worth knowing before reading:

* **Episodes are the spine.** A task or alert belongs to exactly one episode, and the unique index
  on `episode_id` is what makes an at-least-once bus safe: a redelivered observation cannot produce
  a second task for one mess.
* **Observations are partitioned by month** and indexed with BRIN on `ts`. They are append-only,
  queried by time range, and expire - which is exactly what BRIN is good at, at a fraction of the
  index size of a btree.
* **Skills are stored as JSONB, not as a normalised condition tree.** The tree is a recursive
  structure that is always read whole, validated by Pydantic, and never queried by its internals.
  Normalising it would buy nothing and cost every read a five-way join.
* **Anchors own history, cameras do not.** Deleting a camera nulls its anchors' `camera_id` rather
  than cascading, so replacing hardware never destroys streaks or metrics (ADR-010).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from openhup_schemas import new_ulid
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

#: JSONB on Postgres, plain JSON on SQLite, so the single-camera SQLite profile keeps working.
JSONColumn = JSONB().with_variant(JSON(), "sqlite")

#: SQLite only autoincrements an INTEGER PRIMARY KEY, never a BIGINT one, so a bare BigInteger
#: surrogate key fails with a NOT NULL violation there. The variant keeps 64-bit ids on Postgres
#: (these tables grow fast) while staying insertable on the SQLite single-camera profile.
AutoBigInt = BigInteger().with_variant(Integer(), "sqlite")


class Base(DeclarativeBase):
    pass


def _ulid() -> str:
    return new_ulid()


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# --------------------------------------------------------------------------------------
# Cameras and anchors
# --------------------------------------------------------------------------------------


class CameraRow(Base, TimestampMixin):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), default="rtsp", nullable=False)
    #: The full Camera model. Never contains a password - only the name of an env var (see
    #: openhup_schemas.Camera.password_env), so a database dump is not a credential leak.
    config: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    #: Which vision node owns this camera. Two nodes claiming one camera decode it twice, which
    #: /system/health warns about.
    node_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_frame_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    anchors: Mapped[list[AnchorRow]] = relationship(back_populates="camera")


class AnchorRow(Base, TimestampMixin):
    __tablename__ = "anchors"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    #: SET NULL, not CASCADE. Replacing a camera must not delete the history of the places it
    #: watched.
    camera_id: Mapped[str | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True
    )
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    baseline_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    baseline_captured_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    camera: Mapped[CameraRow | None] = relationship(back_populates="anchors")


# --------------------------------------------------------------------------------------
# Skills
# --------------------------------------------------------------------------------------


class SkillRow(Base, TimestampMixin):
    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    #: The validated Skill model. See the module docstring for why this is not normalised.
    definition: Mapped[dict[str, Any]] = mapped_column(JSONColumn, nullable=False)
    origin: Mapped[str] = mapped_column(String(16), default="user", nullable=False)
    source_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Compile warnings from the last save, shown in the UI without recompiling.
    warnings: Mapped[list[Any]] = mapped_column(JSONColumn, default=list, nullable=False)
    tags: Mapped[list[Any]] = mapped_column(JSONColumn, default=list, nullable=False)


class SkillStateRow(Base):
    """One FSM instance per (skill, anchor). Mirrors `openhup.skills.fsm.InstanceState`."""

    __tablename__ = "skill_states"
    __table_args__ = (
        UniqueConstraint("skill_id", "anchor_id", name="uq_skill_state_instance"),
        Index("ix_skill_states_phase", "phase"),
    )

    id: Mapped[int] = mapped_column(AutoBigInt, primary_key=True, autoincrement=True)
    skill_id: Mapped[str] = mapped_column(
        ForeignKey("skills.id", ondelete="CASCADE"), nullable=False
    )
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False)

    phase: Mapped[str] = mapped_column(String(16), default="idle", nullable=False)
    since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    episode_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    episode_opened_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_triggered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    triggers_today: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    counter_day: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    open_task_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    open_alert_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    resolve_pending_since: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    suppressed_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    stale_notified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


# --------------------------------------------------------------------------------------
# Episodes, tasks, alerts
# --------------------------------------------------------------------------------------


class EpisodeRow(Base):
    """One trigger→resolve cycle. The idempotency anchor for every effect."""

    __tablename__ = "episodes"
    __table_args__ = (
        Index("ix_episodes_skill_anchor_opened", "skill_id", "anchor_id", "opened_at"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=_ulid)
    skill_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    trigger_reasons: Mapped[list[Any]] = mapped_column(JSONColumn, default=list, nullable=False)
    resolve_reasons: Mapped[list[Any]] = mapped_column(JSONColumn, default=list, nullable=False)
    #: Denormalised for metric rollups, which would otherwise recompute it on every query.
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)


class TaskRow(Base, TimestampMixin):
    __tablename__ = "tasks"
    __table_args__ = (
        # The idempotency guarantee: one task per episode, enforced by the database rather than by
        # hoping the consumer never sees a message twice.
        UniqueConstraint("episode_id", name="uq_task_per_episode"),
        Index("ix_tasks_open", "state", "skill_id", "anchor_id"),
        Index("ix_tasks_created", "created_at"),
        CheckConstraint("reopen_count <= 2", name="ck_task_reopen_bounded"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=_ulid)
    skill_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    episode_id: Mapped[str] = mapped_column(String(26), nullable=False)

    state: Mapped[str] = mapped_column(String(20), default="open", nullable=False)
    urgency: Mapped[str] = mapped_column(String(10), default="low", nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    #: Tone-free version. Always present, used by screen readers and search whatever the
    #: personality.
    plain_text: Mapped[str] = mapped_column(Text, nullable=False)
    text_source: Mapped[str] = mapped_column(String(10), default="template", nullable=False)

    micro_steps: Mapped[list[Any]] = mapped_column(JSONColumn, default=list, nullable=False)
    current_step: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    before_snapshot: Mapped[str | None] = mapped_column(String(512), nullable=True)
    after_snapshot: Mapped[str | None] = mapped_column(String(512), nullable=True)

    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    snoozed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    reopen_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: User-reported false positive. Feeds the threshold-tuning suggestions, and is the most
    #: valuable feedback signal in the system.
    false_positive: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class AlertRow(Base, TimestampMixin):
    __tablename__ = "alerts"
    __table_args__ = (
        UniqueConstraint("episode_id", name="uq_alert_per_episode"),
        Index("ix_alerts_state_created", "state", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=_ulid)
    skill_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    episode_id: Mapped[str] = mapped_column(String(26), nullable=False)

    state: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    urgency: Mapped[str] = mapped_column(String(10), default="high", nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    plain_text: Mapped[str] = mapped_column(Text, nullable=False)
    text_source: Mapped[str] = mapped_column(String(10), default="template", nullable=False)
    facts: Mapped[list[Any]] = mapped_column(JSONColumn, default=list, nullable=False)

    snapshot_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_notified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    notify_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    delivered_to: Mapped[list[Any]] = mapped_column(JSONColumn, default=list, nullable=False)


# --------------------------------------------------------------------------------------
# Observations and metrics
# --------------------------------------------------------------------------------------


class ObservationRow(Base):
    """Append-only observation log.

    Kept because it is what makes `POST /skills/{id}/simulate` possible - replaying a draft skill
    against real history is the single most useful feature for tuning a threshold, and it needs data
    that predates the skill.

    Partition by month in production (see the Alembic migration) and set a retention window; a
    four-camera install writes on the order of a million rows a month.
    """

    __tablename__ = "observations"
    __table_args__ = (
        # BRIN on ts: append-only, always queried by range, a fraction of a btree's size.
        Index("ix_observations_ts_brin", "ts", postgresql_using="brin"),
        Index("ix_observations_anchor_ts", "anchor_id", "ts"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    detector: Mapped[str] = mapped_column(String(64), nullable=False)
    detector_version: Mapped[str] = mapped_column(String(128), nullable=False)
    #: The full signal list. Read whole by the replayer; never queried by internals.
    signals: Mapped[list[Any]] = mapped_column(JSONColumn, nullable=False)
    snapshot_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    cost_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


class MetricPointRow(Base):
    __tablename__ = "metric_points"
    __table_args__ = (
        UniqueConstraint("metric", "ts", "anchor_id", name="uq_metric_bucket"),
        Index("ix_metric_points_metric_ts", "metric", "ts"),
    )

    id: Mapped[int] = mapped_column(AutoBigInt, primary_key=True, autoincrement=True)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    anchor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    skill_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bucket_s: Mapped[int] = mapped_column(Integer, default=86400, nullable=False)
    labels: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)


class GoalRow(Base, TimestampMixin):
    __tablename__ = "goals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    target: Mapped[float] = mapped_column(Float, nullable=False)
    direction: Mapped[str] = mapped_column(String(10), default="up", nullable=False)
    window_s: Mapped[int] = mapped_column(Integer, default=604800, nullable=False)
    anchor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    include_in_report: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


# --------------------------------------------------------------------------------------
# Personalities, notifications, audit
# --------------------------------------------------------------------------------------


class PersonalityRow(Base, TimestampMixin):
    __tablename__ = "personalities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    definition: Mapped[dict[str, Any]] = mapped_column(JSONColumn, nullable=False)
    #: Presets ship with the project and are replaced on upgrade; user copies are never touched.
    builtin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class NotificationRow(Base):
    """Delivery log. Also the dedupe and rate-limit ledger."""

    __tablename__ = "notifications"
    __table_args__ = (Index("ix_notifications_channel_sent", "channel", "sent_at"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=_ulid)
    channel: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    urgency: Mapped[str] = mapped_column(String(10), default="normal", nullable=False)
    task_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    alert_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: sent | held | failed | suppressed. `held` means quiet hours: visible in the UI now,
    #: delivered when the window ends.
    status: Mapped[str] = mapped_column(String(16), default="sent", nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    dedupe_key: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)


class LLMCallRow(Base):
    """Audit log for every model call.

    Exists so the privacy claim is inspectable rather than aspirational: what was sent, where, how
    big, and whether an image went with it. Surfaced at /api/v1/system/llm-usage.
    """

    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_at", "called_at"),)

    id: Mapped[int] = mapped_column(AutoBigInt, primary_key=True, autoincrement=True)
    called_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    #: False means the payload left the operator's network.
    local: Mapped[bool] = mapped_column(Boolean, nullable=False)
    prompt_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    response_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    included_image: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ok: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


class EventRow(Base):
    """Recent bus events, for the UI timeline and for debugging.

    Trimmed aggressively - this is a convenience view, not an event store. Redis Streams is the bus.
    """

    __tablename__ = "events"
    __table_args__ = (Index("ix_events_ts", "ts"), Index("ix_events_type_ts", "type", "ts"))

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    type: Mapped[str] = mapped_column(String(48), nullable=False)
    skill_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    anchor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    episode_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)


# --------------------------------------------------------------------------------------
# Personality gamble and wins
# --------------------------------------------------------------------------------------


class PersonalityDrawRow(Base, TimestampMixin):
    """The personality gamble: the voice drawn at first setup (ADR-014).

    Exactly one row, id `default`. The draw is made at seed time when `personality.gamble` is on,
    and it becomes the effective default personality without being announced - a mystery to live
    with, revealed in Settings and documented in the ADR. Deleting the row returns to the
    configured `default_personality`. `reroll_count` records how many times the household has
    redrawn, so the review screen can show that the voice was earned, not chosen.
    """

    __tablename__ = "personality_draw"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default="default")
    personality_id: Mapped[str] = mapped_column(String(64), nullable=False)
    reroll_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class WinMilestoneRow(Base, TimestampMixin):
    """One celebrated win: an anchor stayed clear for whole days.

    The ledger is the dedupe. A win is claimed once per (anchor, kind, value): a 7-day stretch is
    celebrated when it first happens and never again, and each new 90-day record earns its own
    row. Everything here is reviewable in Settings, matching the bar set for facts and patterns -
    progress the assistant noticed is as inspectable as anything else it thinks it knows.

    `kind` is `clear_days` (a band milestone) or `record_clear_days` (longest stretch in 90 days).
    `value` is the dedupe key: the band floor, or the rounded record. `days` is the actual stretch.
    """

    __tablename__ = "win_milestones"
    __table_args__ = (
        UniqueConstraint("anchor_id", "kind", "value", name="uq_win_milestone"),
        Index("ix_win_milestones_anchor", "anchor_id"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=_ulid)
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    #: Dedupe key: the band floor (1, 3, 7, 14, 30) or the rounded record days.
    value: Mapped[float] = mapped_column(Float, nullable=False)
    days: Mapped[float] = mapped_column(Float, nullable=False)
    #: Tone-free summary for the review screen; the spoken line was already said at the time.
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    #: When the win was spoken aloud (None if it fell inside quiet hours; the milestone still
    #: stands, it is just not announced twice).
    spoken_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# --------------------------------------------------------------------------------------
# Consent-gated household members (ADR-016)
# --------------------------------------------------------------------------------------


class MemberRow(Base, TimestampMixin):
    """A person who said yes to being remembered.

    The embedding here is the *entire* face store - there is no cache of unknown faces anywhere.
    Consent is the only way in: an embedding row is created only when the person answers "yes" to
    the consent question, and deleting the member deletes the embedding. An unknown face is asked
    once per anchor per day (see `ConsentAskRow`), and the marker stores that a question was asked,
    never what the person looked like.

    The embedding is local Postgres and never leaves the house. Members are listable and deletable
    in Settings, the same reviewability bar as facts, patterns, and wins.
    """

    __tablename__ = "members"
    __table_args__ = (
        UniqueConstraint("name", name="uq_member_name"),
        Index("ix_members_active", "active"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=_ulid)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Face embedding, stored as a JSON list of floats. Only ever written with consent.
    embedding: Mapped[list[Any]] = mapped_column(JSONColumn, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    enrolled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ConsentAskRow(Base, TimestampMixin):
    """The 24-hour no-reask marker: "an unknown person was asked in this anchor today".

    Deliberately stores no biometric data - only that a question was asked, where, and what the
    answer was. The person can be asked again tomorrow, and saying "no" now does not prevent them
    from enrolling later from the Settings screen. Unique on (anchor, day), so an anchor full of
    guests costs one row a day, not one per face.
    """

    __tablename__ = "consent_asks"
    __table_args__ = (
        UniqueConstraint("anchor_id", "asked_on", name="uq_consent_ask_per_day"),
        Index("ix_consent_asks_anchor_day", "anchor_id", "asked_on"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=_ulid)
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    #: A calendar day, not a timestamp: the marker exists to stop re-asking, and a day boundary
    #: must reset it even at 11:59 p.m.
    asked_on: Mapped[date] = mapped_column(Date, nullable=False)
    #: no | yes | skipped. `skipped` is the same as no but recorded when the question was
    #: dismissed in the UI rather than answered aloud.
    answer: Mapped[str] = mapped_column(String(12), default="no", nullable=False)
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PresenceWindowRow(Base):
    """A room fact: member X was in anchor Y from A to B (or is there now).

    Identity is presence, never attribution: this row says Sam was *in the kitchen*, it never says
    Sam did anything. The engine opens a window when an observation reports a known member and
    closes it when the member drops out. Wins and nudges may reference who was present, because
    presence is what the member consented to share - but the words stay "was here", never "did it".
    """

    __tablename__ = "presence_windows"
    __table_args__ = (
        Index("ix_presence_windows_open", "anchor_id", "ended_at"),
        Index("ix_presence_windows_member", "member_id"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=_ulid)
    member_id: Mapped[str] = mapped_column(String(26), nullable=False)
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    #: None while the member is still present.
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# --------------------------------------------------------------------------------------
# Household memory
# --------------------------------------------------------------------------------------


class MemoryPatternRow(Base, TimestampMixin):
    """A pattern learned from the household's own episode history.

    Derived, never observed directly: the discovery pass (`openhup.memory.patterns`) turns the
    episodes the skill engine already records into claims like "the kitchen counter usually needs
    attention about every 3 days". Everything here is reviewable and dismissable in Settings, the
    same bar as `MemoryFactRow` - a learned pattern the user cannot inspect and delete is a memory
    that has started watching them back (ADR-013).

    One row per (kind, skill, anchor), upserted on refresh. `status == "dismissed"` means the user
    said it was not useful: the row is kept so it is not learned again, but it is never surfaced or
    nudged. Patterns are never per-person; the subject is always a place and a skill.
    """

    __tablename__ = "memory_patterns"
    __table_args__ = (UniqueConstraint("kind", "skill_id", "anchor_id", name="uq_memory_pattern"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=_ulid)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    skill_id: Mapped[str] = mapped_column(String(64), nullable=False)
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Forward-facing claim, ready to speak: "The kitchen counter usually needs attention about
    #: every 3 days." Never backwards-facing, never a backlog count.
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    #: The numbers behind the claim: n_episodes, median/mean interval, span, last episode. Shown on
    #: the review screen so a claim can be judged by its evidence, not trusted.
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    #: active | dismissed. Dismissed rows are kept so the pattern is not learned again.
    status: Mapped[str] = mapped_column(String(12), default="active", nullable=False, index=True)
    #: When this pattern last earned a spoken nudge, and which episode cycle that nudge was for.
    last_nudge_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_nudge_basis: Mapped[str | None] = mapped_column(String(26), nullable=True)


class MemoryFactRow(Base, TimestampMixin):
    """One thing the household told the assistant, in its own words.

    This is the whole "memory" feature and it is deliberately small: a store of plain, human-
    readable claims - "bin day is Tuesday", "call the spare room the junk room" - that the voice
    interface can teach and recall and the review screen in Settings can list and delete.

    Privacy model, matching the rest of the system: facts live in local Postgres and never leave the
    house by themselves. The only way one reaches a model is as a fragment inside a phrasing prompt,
    which is already gated by `llm.allow_remote_llm` and recorded in the usage audit. A memory the
    user cannot inspect and delete is not a memory, it is a surveillance system, so everything here
    is listable, per-row deletable, and never edited silently.
    """

    __tablename__ = "memory_facts"
    __table_args__ = (Index("ix_memory_facts_topic", "topic"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=_ulid)
    #: The claim, roughly as the person said it. Kept in their words, not normalised, so the recall
    #: reply and the review screen both show exactly what was taught.
    fact: Mapped[str] = mapped_column(Text, nullable=False)
    #: Optional label so a group of related facts can be reviewed and forgotten together.
    topic: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: Where it came from: voice | settings | api.
    source: Mapped[str] = mapped_column(String(16), default="api", nullable=False)


#: Retention defaults for the tables that grow without bound. Applied by the reaper task; every one
#: is overridable, and none of them is "forever".
RETENTION: dict[str, timedelta] = {
    "observations": timedelta(days=14),
    "events": timedelta(days=7),
    "notifications": timedelta(days=30),
    "llm_calls": timedelta(days=30),
    "episodes": timedelta(days=400),  # kept long: metrics and streaks are computed from these
}


__all__ = [
    "RETENTION",
    "AlertRow",
    "AnchorRow",
    "Base",
    "CameraRow",
    "ConsentAskRow",
    "EpisodeRow",
    "EventRow",
    "GoalRow",
    "LLMCallRow",
    "MemberRow",
    "MemoryFactRow",
    "MemoryPatternRow",
    "MetricPointRow",
    "NotificationRow",
    "ObservationRow",
    "PersonalityDrawRow",
    "PersonalityRow",
    "PresenceWindowRow",
    "SkillRow",
    "SkillStateRow",
    "TaskRow",
    "WinMilestoneRow",
]
