"""The skill-instance state machine. Pure transition logic.

One instance per `(skill, anchor)`. `advance()` takes the current state plus two verdicts and
returns the next state and a list of *actions* for the engine to perform. It touches no database and
reads no clock, which is what makes the awkward cases - cooldowns, daily caps, quiet hours, grace
periods, verification reopens - testable without a running deployment.

Rules that are load-bearing rather than incidental:

* **Absence of data never resolves anything.** A task is closed because the camera saw the space
  become clean, not because the camera stopped answering. An unplugged camera moves the instance to
  STALE and raises a notice.
* **One episode at a time per instance**, and every effect is keyed by `episode_id`, so an
  at-least-once bus cannot produce two tasks for one mess.
* **Suppression is visible.** A trigger blocked by quiet hours or a daily cap emits an event so the
  UI can say "held until 07:00" instead of losing it silently.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta

from openhup_schemas import (
    AlertEffect,
    EventType,
    MetricEffect,
    SkillPhase,
    TaskEffect,
    TaskMode,
    new_ulid,
)

from .compile import CompiledSkill
from .evaluate import Verdict

# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InstanceState:
    """Durable state of one (skill, anchor) pair. Persisted; rebuilt on engine restart."""

    skill_id: str
    anchor_id: str
    phase: SkillPhase = SkillPhase.IDLE
    since: datetime | None = None

    episode_id: str | None = None
    episode_opened_at: datetime | None = None
    last_triggered_at: datetime | None = None
    last_resolved_at: datetime | None = None

    #: Daily cap bookkeeping. `counter_day` is a UTC date; for households whose "day" boundary
    #: matters, quiet_hours is the right tool rather than this counter.
    triggers_today: int = 0
    counter_day: date | None = None

    open_task_id: str | None = None
    open_alert_id: str | None = None

    #: When the resolve condition first held, for grace-period timing.
    resolve_pending_since: datetime | None = None
    #: Last suppression reason emitted, so the event fires on change rather than every tick.
    suppressed_reason: str | None = None
    stale_notified: bool = False

    def with_phase(self, phase: SkillPhase, now: datetime) -> InstanceState:
        return replace(self, phase=phase, since=now)

    def attach_task(self, task_id: str) -> InstanceState:
        """Record the task the engine created for the current episode.

        `advance()` emits a `CreateTask` action but cannot know the id, because the id is minted
        when the row is written. The caller must feed it back, or the FSM will never find a task to
        resolve. `simulate.py` and `engine.py` are the two callers that do this.
        """
        return replace(self, open_task_id=task_id)

    def attach_alert(self, alert_id: str) -> InstanceState:
        return replace(self, open_alert_id=alert_id)

    @property
    def has_open_effect(self) -> bool:
        return self.open_task_id is not None or self.open_alert_id is not None


# --------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpenEpisode:
    skill_id: str
    anchor_id: str
    episode_id: str
    opened_at: datetime
    trigger_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CloseEpisode:
    skill_id: str
    anchor_id: str
    episode_id: str
    closed_at: datetime
    resolve_reasons: tuple[str, ...]
    #: How long the condition held. Feeds duration metrics directly.
    duration: timedelta


@dataclass(frozen=True, slots=True)
class CreateTask:
    skill_id: str
    anchor_id: str
    episode_id: str
    idempotency_key: str
    facts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CreateAlert:
    skill_id: str
    anchor_id: str
    episode_id: str
    idempotency_key: str
    facts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolveTask:
    task_id: str
    auto: bool
    facts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpireTask:
    task_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class ResolveAlert:
    alert_id: str
    facts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecordMetricEpisode:
    """A metric skill's episode ended; the rollup worker turns this into points."""

    skill_id: str
    anchor_id: str
    metric: str
    started_at: datetime
    ended_at: datetime
    duration: timedelta


@dataclass(frozen=True, slots=True)
class Notice:
    """A system-level message for the UI and log: stale camera, suppressed trigger."""

    code: str
    message: str
    severity: str = "info"  # info | warning


Action = (
    OpenEpisode
    | CloseEpisode
    | CreateTask
    | CreateAlert
    | ResolveTask
    | ExpireTask
    | ResolveAlert
    | RecordMetricEpisode
    | Notice
)


@dataclass(frozen=True, slots=True)
class Decision:
    """Result of one `advance()` call."""

    state: InstanceState
    actions: tuple[Action, ...] = ()
    events: tuple[EventType, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.actions or self.events)


@dataclass(frozen=True, slots=True)
class EngineContext:
    """Facts `advance()` cannot derive from its own state.

    Kept explicit rather than looked up inside, so the function stays pure.
    """

    now: datetime
    #: Open tasks for this skill across *all* anchors. Enforces single-task focus house-wide.
    open_tasks_for_skill: int = 0
    #: Set when the operator has paused everything (holiday mode, guests).
    globally_paused: bool = False


# --------------------------------------------------------------------------------------
# Transition
# --------------------------------------------------------------------------------------


def advance(
    state: InstanceState,
    compiled: CompiledSkill,
    trigger: Verdict,
    resolve: Verdict,
    context: EngineContext,
) -> Decision:
    """Compute the next state and the actions to take.

    Args:
        state: current instance state.
        compiled: the skill, with anchors and bindings resolved.
        trigger: verdict for `skill.conditions` at `context.now`.
        resolve: verdict for `skill.resolve.conditions` at the same instant.
        context: clock and cross-instance facts.
    """
    skill = compiled.skill
    now = context.now
    actions: list[Action] = []
    events: list[EventType] = []

    state = _roll_day_counter(state, now)

    # -- disabled ---------------------------------------------------------------------
    if not skill.enabled or context.globally_paused:
        if state.phase is SkillPhase.DISABLED:
            return Decision(state)
        if state.open_task_id:
            actions.append(
                ExpireTask(
                    state.open_task_id,
                    "skill disabled" if not skill.enabled else "monitoring paused",
                )
            )
        return Decision(
            state=replace(
                state.with_phase(SkillPhase.DISABLED, now),
                open_task_id=None,
                open_alert_id=None,
                episode_id=None,
                resolve_pending_since=None,
            ),
            actions=tuple(actions),
            events=(EventType.SKILL_SUPPRESSED,),
        )

    if state.phase is SkillPhase.DISABLED:
        state = state.with_phase(SkillPhase.IDLE, now)

    # -- data health -------------------------------------------------------------------
    # Staleness is only allowed to change the phase while nothing is in flight. Mid-episode, a
    # camera dropping out must not be able to close a task, so ACTING and RESOLVING are handled
    # below with an explicit `resolve.is_evaluable` guard instead.
    if state.phase in {SkillPhase.IDLE, SkillPhase.ARMED} and not trigger.is_evaluable:
        stale_state = state.with_phase(SkillPhase.STALE, now)
        if not state.stale_notified:
            missing = ", ".join(trigger.missing or trigger.stale)
            actions.append(
                Notice(
                    "signals_stale",
                    f"{skill.id} on {state.anchor_id}: no fresh data for {missing}. "
                    f"Check the camera or the vision service - this skill is not watching "
                    f"anything right now.",
                    severity="warning",
                )
            )
            events.append(EventType.SKILL_STALE)
            stale_state = replace(stale_state, stale_notified=True)
        return Decision(stale_state, tuple(actions), tuple(events))

    if state.phase is SkillPhase.STALE:
        if not trigger.is_evaluable:
            return Decision(state)
        state = replace(state.with_phase(SkillPhase.ARMED, now), stale_notified=False)
        events.append(EventType.SKILL_ARMED)

    if state.phase is SkillPhase.IDLE:
        state = state.with_phase(SkillPhase.ARMED, now)
        events.append(EventType.SKILL_ARMED)

    # -- armed: should we fire? --------------------------------------------------------
    if state.phase is SkillPhase.ARMED:
        if not trigger.matched:
            return Decision(replace(state, suppressed_reason=None), tuple(actions), tuple(events))

        suppression = _suppression_reason(state, compiled, context)
        if suppression is not None:
            key, message = suppression
            # Compare on the stable key, not the message: "cooling down, 12m remaining" changes
            # every tick and would emit a notice every second.
            if state.suppressed_reason != key:
                actions.append(Notice("trigger_suppressed", f"{skill.id}: {message}"))
                events.append(EventType.SKILL_SUPPRESSED)
            return Decision(replace(state, suppressed_reason=key), tuple(actions), tuple(events))

        episode_id = new_ulid()
        facts = tuple(trigger.facts())
        actions.append(OpenEpisode(skill.id, state.anchor_id, episode_id, now, facts))
        events.append(EventType.SKILL_TRIGGERED)

        key = f"{skill.id}:{state.anchor_id}:{episode_id}"
        if isinstance(skill.effect, TaskEffect):
            actions.append(CreateTask(skill.id, state.anchor_id, episode_id, key, facts))
            events.append(EventType.TASK_CREATED)
        elif isinstance(skill.effect, AlertEffect):
            actions.append(CreateAlert(skill.id, state.anchor_id, episode_id, key, facts))
            events.append(EventType.ALERT_RAISED)
        # Metric effects create nothing now; the episode itself is the measurement.

        return Decision(
            state=replace(
                state.with_phase(SkillPhase.ACTING, now),
                episode_id=episode_id,
                episode_opened_at=now,
                last_triggered_at=now,
                triggers_today=state.triggers_today + 1,
                resolve_pending_since=None,
                suppressed_reason=None,
            ),
            actions=tuple(actions),
            events=tuple(events),
        )

    # -- acting: wait for resolution ---------------------------------------------------
    if state.phase is SkillPhase.ACTING:
        spec = skill.resolve
        opened = state.episode_opened_at or now

        expire_after = spec.auto_expire_after if spec is not None else None
        if expire_after is not None and now - opened >= expire_after:
            if state.open_task_id:
                actions.append(
                    ExpireTask(state.open_task_id, f"unresolved after {_short(expire_after)}")
                )
            actions.append(_close_episode(state, now, ("expired without visible resolution",)))
            events.append(EventType.SKILL_RESOLVED)
            return Decision(_to_cooldown(state, now), tuple(actions), tuple(events))

        if spec is not None and spec.manual_only:
            return Decision(state, tuple(actions), tuple(events))

        # The guard that keeps a dead camera from closing tasks.
        if not resolve.is_evaluable:
            return Decision(state, tuple(actions), tuple(events))

        if resolve.matched:
            grace = spec.grace if spec is not None else timedelta(0)
            if grace <= timedelta(0):
                return _finish(state, compiled, resolve, now, actions, events)
            return Decision(
                replace(
                    state.with_phase(SkillPhase.RESOLVING, now),
                    resolve_pending_since=now,
                ),
                tuple(actions),
                tuple(events),
            )
        return Decision(state, tuple(actions), tuple(events))

    # -- resolving: hold for the grace period ------------------------------------------
    if state.phase is SkillPhase.RESOLVING:
        if not resolve.matched or not resolve.is_evaluable:
            # It got messy again during the grace window. Back to work, no new task.
            return Decision(
                replace(state.with_phase(SkillPhase.ACTING, now), resolve_pending_since=None),
                tuple(actions),
                tuple(events),
            )

        grace = skill.resolve.grace if skill.resolve is not None else timedelta(0)
        pending_since = state.resolve_pending_since or now
        if now - pending_since >= grace:
            return _finish(state, compiled, resolve, now, actions, events)
        return Decision(state, tuple(actions), tuple(events))

    # -- cooldown ----------------------------------------------------------------------
    if state.phase is SkillPhase.COOLDOWN:
        anchor_time = state.last_resolved_at or state.since or now
        if now - anchor_time >= skill.limits.cooldown:
            return Decision(
                state.with_phase(SkillPhase.ARMED, now), tuple(actions), (EventType.SKILL_ARMED,)
            )
        if trigger.matched:
            # The condition is true again but we are deliberately holding off. Say so once, so the
            # UI can show "held, cooling down" instead of appearing to have missed it.
            remaining = skill.limits.cooldown - (now - anchor_time)
            if state.suppressed_reason != "cooldown":
                actions.append(
                    Notice(
                        "trigger_suppressed",
                        f"{skill.id}: cooling down, {_short(remaining)} remaining",
                    )
                )
                events.append(EventType.SKILL_SUPPRESSED)
            state = replace(state, suppressed_reason="cooldown")
        return Decision(state, tuple(actions), tuple(events))

    return Decision(state, tuple(actions), tuple(events))


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _finish(
    state: InstanceState,
    compiled: CompiledSkill,
    resolve: Verdict,
    now: datetime,
    actions: list[Action],
    events: list[EventType],
) -> Decision:
    """Close out an episode: resolve the effect, record the episode, enter cooldown."""
    skill = compiled.skill
    facts = tuple(resolve.facts())

    if state.open_task_id:
        actions.append(ResolveTask(state.open_task_id, auto=True, facts=facts))
        events.append(EventType.TASK_COMPLETED)
    if state.open_alert_id:
        actions.append(ResolveAlert(state.open_alert_id, facts=facts))
        events.append(EventType.ALERT_RESOLVED)

    actions.append(_close_episode(state, now, facts))
    events.append(EventType.SKILL_RESOLVED)

    if isinstance(skill.effect, MetricEffect):
        opened = state.episode_opened_at or now
        actions.append(
            RecordMetricEpisode(
                skill_id=skill.id,
                anchor_id=state.anchor_id,
                metric=skill.effect.metric,
                started_at=opened,
                ended_at=now,
                duration=now - opened,
            )
        )
        events.append(EventType.METRIC_POINT)

    return Decision(_to_cooldown(state, now), tuple(actions), tuple(events))


def _close_episode(state: InstanceState, now: datetime, facts: tuple[str, ...]) -> CloseEpisode:
    opened = state.episode_opened_at or now
    return CloseEpisode(
        skill_id=state.skill_id,
        anchor_id=state.anchor_id,
        episode_id=state.episode_id or "",
        closed_at=now,
        resolve_reasons=facts,
        duration=now - opened,
    )


def _to_cooldown(state: InstanceState, now: datetime) -> InstanceState:
    return replace(
        state.with_phase(SkillPhase.COOLDOWN, now),
        episode_id=None,
        episode_opened_at=None,
        last_resolved_at=now,
        open_task_id=None,
        open_alert_id=None,
        resolve_pending_since=None,
    )


def _roll_day_counter(state: InstanceState, now: datetime) -> InstanceState:
    today = now.date()
    if state.counter_day == today:
        return state
    return replace(state, counter_day=today, triggers_today=0)


def _suppression_reason(
    state: InstanceState,
    compiled: CompiledSkill,
    context: EngineContext,
) -> tuple[str, str] | None:
    """Why this trigger should not fire right now, as (stable key, human message), or None.

    Safety comes first: an alert at high urgency ignores quiet hours and daily caps. Everything
    else is negotiable, and being negotiable is the point.
    """
    skill = compiled.skill
    now = context.now
    limits = skill.limits
    urgent = skill.urgency.bypasses_personality

    if state.last_resolved_at is not None and not urgent:
        elapsed = now - state.last_resolved_at
        if elapsed < limits.cooldown:
            remaining = limits.cooldown - elapsed
            return "cooldown", f"cooling down, {_short(remaining)} remaining"

    if limits.max_per_day is not None and not urgent and state.triggers_today >= limits.max_per_day:
        return "daily_cap", f"daily cap of {limits.max_per_day} reached"

    if limits.quiet_hours is not None and not urgent and limits.quiet_hours.contains(now):
        return "quiet_hours", f"quiet hours ({limits.quiet_hours})"

    if isinstance(skill.effect, TaskEffect):
        if skill.effect.mode is TaskMode.SINGLE_TASK_FOCUS and context.open_tasks_for_skill > 0:
            return (
                "single_task_focus",
                "single-task focus: an earlier task is still open",
            )
        if (
            limits.max_open_tasks is not None
            and context.open_tasks_for_skill >= limits.max_open_tasks
        ):
            return "max_open_tasks", f"already at max_open_tasks ({limits.max_open_tasks})"

    return None


def _short(delta: timedelta) -> str:
    total = int(delta.total_seconds())
    if total >= 3600:
        return f"{total // 3600}h{(total % 3600) // 60:02d}m"
    if total >= 60:
        return f"{total // 60}m"
    return f"{total}s"


__all__ = [
    "Action",
    "CloseEpisode",
    "CreateAlert",
    "CreateTask",
    "Decision",
    "EngineContext",
    "ExpireTask",
    "InstanceState",
    "Notice",
    "OpenEpisode",
    "RecordMetricEpisode",
    "ResolveAlert",
    "ResolveTask",
    "advance",
]
