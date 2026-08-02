"""Bus envelope and topic names.

Redis Streams, one stream per topic, consumer groups per subscriber (ADR-002). Topic names are
constants rather than f-strings at call sites so that a rename is a one-line change and a typo is
an ImportError instead of a silently dead subscription.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import ULID, Slug, StrEnum, new_ulid, utcnow

STREAM_PREFIX = "openhup"


class Topic(StrEnum):
    """Redis stream keys."""

    #: Every observation from every vision instance and sensor bridge.
    OBSERVATIONS = "openhup:obs"
    #: Skill FSM transitions. Consumed by the UI hub and the audit log.
    SKILL_EVENTS = "openhup:evt.skill"
    TASK_EVENTS = "openhup:evt.task"
    ALERT_EVENTS = "openhup:evt.alert"
    METRIC_EVENTS = "openhup:evt.metric"
    SYSTEM_EVENTS = "openhup:evt.system"
    #: Outbound notification queue, so a slow Matrix homeserver cannot stall the engine.
    NOTIFICATIONS = "openhup:cmd.notify"
    #: Control channel to the vision service: reload plan, capture snapshot, recapture baseline.
    VISION_COMMANDS = "openhup:cmd.vision"


class ConsumerGroup(StrEnum):
    SKILL_ENGINE = "skill-engine"
    METRICS_ROLLUP = "metrics-rollup"
    WS_FANOUT = "ws-fanout"
    NOTIFIER = "notifier"
    ARCHIVER = "archiver"


class EventType(StrEnum):
    OBSERVATION = "observation"

    SKILL_ARMED = "skill.armed"
    SKILL_TRIGGERED = "skill.triggered"
    SKILL_RESOLVED = "skill.resolved"
    SKILL_STALE = "skill.stale"
    SKILL_SUPPRESSED = "skill.suppressed"

    TASK_CREATED = "task.created"
    TASK_UPDATED = "task.updated"
    TASK_STEP_ADVANCED = "task.step_advanced"
    TASK_COMPLETED = "task.completed"
    TASK_REOPENED = "task.reopened"

    ALERT_RAISED = "alert.raised"
    ALERT_ACKED = "alert.acked"
    ALERT_RESOLVED = "alert.resolved"

    METRIC_POINT = "metric.point"
    GOAL_PROGRESS = "goal.progress"

    CAMERA_OFFLINE = "system.camera_offline"
    CAMERA_ONLINE = "system.camera_online"
    MODEL_LOADED = "system.model_loaded"
    LLM_UNAVAILABLE = "system.llm_unavailable"
    #: Emitted when a notification is dropped by quiet hours or a rate limit, so the UI can show
    #: "held until 07:00" rather than losing the message silently.
    NOTIFICATION_HELD = "system.notification_held"
    ENGINE_LEADER_CHANGED = "system.engine_leader_changed"
    #: A learned pattern says something is about due. Spoken by the client; see ADR-013.
    PATTERN_NUDGE = "system.pattern_nudge"
    #: An anchor stayed clear for whole days - a milestone or a 90-day record. Spoken by the
    #: client; see ADR-015.
    WIN_NOTE = "system.win_note"
    #: An unknown face earned the consent question (ADR-016). The client speaks it and carries the
    #: answer back to /voice/command or the settings screen. No biometric data rides the event.
    CONSENT_ASK = "system.consent_ask"


class Envelope(BaseModel):
    """Uniform wrapper for everything on the bus and everything on the WebSocket.

    `idempotency_key` is what makes at-least-once delivery safe: consumers that produce side
    effects check it before acting. For task creation it is `skill_id:anchor_id:episode_id`.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: ULID = Field(default_factory=new_ulid)
    type: EventType
    ts: datetime = Field(default_factory=utcnow)
    #: Payload shape is determined by `type`; consumers validate into the concrete model.
    payload: dict[str, Any] = Field(default_factory=dict)

    skill_id: Slug | None = None
    anchor_id: Slug | None = None
    episode_id: ULID | None = None
    idempotency_key: str | None = None
    #: Correlates every event caused by one observation, for tracing a decision end to end.
    trace_id: str | None = None
    source: str = Field(default="backend", description="backend | vision | agent | api")

    def redis_fields(self) -> dict[str, str]:
        """Flatten for XADD. The payload is JSON; the rest stay queryable as plain fields."""
        return {
            "id": self.id,
            "type": self.type.value,
            "ts": self.ts.isoformat(),
            "skill_id": self.skill_id or "",
            "anchor_id": self.anchor_id or "",
            "episode_id": self.episode_id or "",
            "idempotency_key": self.idempotency_key or "",
            "trace_id": self.trace_id or "",
            "source": self.source,
            "payload": self.model_dump_json(include={"payload"}),
        }


class VisionCommand(BaseModel):
    """Backend → vision service control messages."""

    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "reload_plan", "capture_snapshot", "capture_baseline", "probe_camera", "shutdown"
    ]
    camera_id: Slug | None = None
    anchor_id: Slug | None = None
    reply_to: str | None = Field(
        default=None, description="Redis key the service writes the result to, for RPC-ish calls."
    )
    args: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "STREAM_PREFIX",
    "ConsumerGroup",
    "Envelope",
    "EventType",
    "Topic",
    "VisionCommand",
]
