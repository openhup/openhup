"""Tasks, alerts, episodes, and the micro-step ladder.

An **Episode** is one trigger→resolve cycle of one skill instance. It is the idempotency key for
every effect: the bus is at-least-once, so a redelivered observation must not be able to create a
second task. Effects are keyed on `(skill_id, anchor_id, episode_id)` and the observable behaviour
is effectively-once.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import (
    ULID,
    Duration,
    Slug,
    StrEnum,
    TaskState,
    TextSource,
    Urgency,
    new_ulid,
    utcnow,
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Episode(_Base):
    """One trigger→resolve cycle."""

    id: ULID = Field(default_factory=new_ulid)
    skill_id: Slug
    anchor_id: Slug
    opened_at: datetime = Field(default_factory=utcnow)
    closed_at: datetime | None = None
    #: Snapshot of the facts that opened the episode, for explanation and audit.
    trigger_reasons: list[str] = Field(default_factory=list)
    resolve_reasons: list[str] = Field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    def duration(self, now: datetime | None = None) -> timedelta:
        return (self.closed_at or now or utcnow()) - self.opened_at


class MicroStep(_Base):
    """One rung of the ladder.

    `subregion_id` is what makes progress observable rather than self-reported: when that slice
    of the anchor clears, the step advances on its own.
    """

    index: int = Field(ge=0)
    text: str
    subregion_id: str | None = None
    done: bool = False
    done_at: datetime | None = None
    #: Clutter score in the step's region when the step was created, for measuring the delta.
    baseline_score: float | None = None


class SnapshotPair(_Base):
    """Before/after. The "after" is the reward, and it is worth persisting for that reason alone."""

    before_ref: str | None = None
    after_ref: str | None = None
    before_at: datetime | None = None
    after_at: datetime | None = None


class Task(_Base):
    """Something to do, created by a skill and usually completed by the camera noticing."""

    id: ULID = Field(default_factory=new_ulid)
    skill_id: Slug
    anchor_id: Slug
    episode_id: ULID

    state: TaskState = TaskState.OPEN
    urgency: Urgency = Urgency.LOW

    #: Final wording shown to the user, already through the personality layer.
    text: str
    text_source: TextSource = TextSource.TEMPLATE
    #: Plain description of the facts, independent of tone. Always available, always safe -
    #: used for accessibility, for search, and whenever the personality output is filtered out.
    plain_text: str

    micro_steps: list[MicroStep] = Field(default_factory=list)
    current_step: int = 0

    snapshots: SnapshotPair = Field(default_factory=SnapshotPair)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    snoozed_until: datetime | None = None
    expires_at: datetime | None = None

    #: How many times this task has been reopened after a failed verification. Capped at 1 by
    #: the engine: OpenHup gets one chance to disagree, then it trusts the human.
    reopen_count: int = 0
    note: str | None = None
    #: Set when the user tells us this was a false positive. Feeds threshold-tuning suggestions.
    false_positive: bool = False

    @model_validator(mode="after")
    def _check_step_index(self) -> Self:
        if self.micro_steps and not 0 <= self.current_step < len(self.micro_steps):
            raise ValueError(
                f"current_step {self.current_step} out of range for "
                f"{len(self.micro_steps)} micro steps"
            )
        return self

    @property
    def is_actionable(self) -> bool:
        return self.state in {TaskState.OPEN, TaskState.IN_PROGRESS}

    @property
    def visible_step(self) -> MicroStep | None:
        """In single-task focus, this is the only thing the user is shown."""
        if not self.micro_steps:
            return None
        return self.micro_steps[self.current_step]

    @property
    def progress(self) -> float:
        if not self.micro_steps:
            return 1.0 if self.state.is_resolved else 0.0
        return sum(1 for s in self.micro_steps if s.done) / len(self.micro_steps)


class AlertState(StrEnum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class Alert(_Base):
    """Something that needs attention now. High-urgency alerts are always phrased plainly."""

    id: ULID = Field(default_factory=new_ulid)
    skill_id: Slug
    anchor_id: Slug
    episode_id: ULID

    state: AlertState = AlertState.ACTIVE
    urgency: Urgency = Urgency.HIGH
    text: str
    text_source: TextSource = TextSource.TEMPLATE
    plain_text: str
    #: The conditions that fired, in human-readable form. Shown in the notification body,
    #: because "burner on for 12m, nobody present for 9m" is more useful than any phrasing.
    facts: list[str] = Field(default_factory=list)

    snapshot_ref: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    resolved_at: datetime | None = None
    last_notified_at: datetime | None = None
    notify_count: int = 0
    #: Channel ids this alert was delivered to, with per-channel success recorded by the notifier.
    delivered_to: list[str] = Field(default_factory=list)

    @property
    def needs_notification(self) -> bool:
        return self.state is AlertState.ACTIVE and self.notify_count == 0


class NotificationRequest(_Base):
    """What the notifier consumes. Deliberately decoupled from Task/Alert so channels don't grow
    knowledge of the task FSM."""

    id: ULID = Field(default_factory=new_ulid)
    channels: list[Slug]
    title: str
    body: str
    urgency: Urgency = Urgency.NORMAL
    snapshot_ref: str | None = None
    #: Deep link back into the UI, e.g. "/tasks/01K3...".
    link: str | None = None
    #: Dedupe key; channels that support it (ntfy, matrix) will replace rather than repeat.
    dedupe_key: str | None = None
    ttl: Duration | None = None


__all__ = [
    "Alert",
    "AlertState",
    "Episode",
    "MicroStep",
    "NotificationRequest",
    "SnapshotPair",
    "Task",
]
