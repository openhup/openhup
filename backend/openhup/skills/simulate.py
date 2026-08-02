"""Replay stored observations against a skill, without touching anything.

This is the answer to "will this skill annoy me?". Before arming a skill, replay it against the last
week of observations and see that it would have fired fourteen times - which is the fastest possible
cure for a badly chosen threshold, and much kinder than discovering it overnight.

It is also the reference implementation of the engine's inner loop, and the driver the integration
tests use. Keeping the loop here, in a pure module with an injected clock, means the tested loop and
the shipped loop are the same shape:

    observations + ticks → windows → evaluate → advance → actions

The only thing simulated is the *effects*: tasks get fake ids instead of database rows, and no
notification is ever sent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from openhup_schemas import Observation, SkillPhase, TaskEffect, TaskMode

from .compile import CompiledSkill
from .evaluate import evaluate_both
from .fsm import (
    Action,
    CreateAlert,
    CreateTask,
    EngineContext,
    ExpireTask,
    InstanceState,
    Notice,
    ResolveAlert,
    ResolveTask,
    advance,
)
from .window import WindowStore

#: How often the simulated clock ticks between observations. Matches the engine's tick so that
#: `for:` and `absent_for:` resolve at the same granularity in simulation as in production.
DEFAULT_TICK = timedelta(seconds=30)


@dataclass(frozen=True, slots=True)
class SimulationStep:
    """One evaluation instant worth recording. Boring instants are dropped."""

    ts: datetime
    anchor_id: str
    phase: SkillPhase
    trigger_matched: bool
    resolve_matched: bool
    evaluable: bool
    actions: tuple[Action, ...]
    reasons: tuple[str, ...] = ()


@dataclass
class SimulationResult:
    """What the skill would have done."""

    skill_id: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    steps: list[SimulationStep] = field(default_factory=list)

    tasks_created: int = 0
    alerts_raised: int = 0
    tasks_auto_resolved: int = 0
    tasks_expired: int = 0
    suppressions: int = 0
    stale_periods: int = 0
    observations_seen: int = 0
    ticks: int = 0

    #: How long each episode stayed open. The distribution is more informative than the count:
    #: twenty episodes of four seconds means a threshold problem, not a messy house.
    episode_durations: list[timedelta] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)

    @property
    def episodes(self) -> int:
        return len(self.episode_durations)

    @property
    def mean_episode_duration(self) -> timedelta:
        if not self.episode_durations:
            return timedelta(0)
        return sum(self.episode_durations, timedelta(0)) / len(self.episode_durations)

    @property
    def shortest_episode(self) -> timedelta:
        return min(self.episode_durations, default=timedelta(0))

    def per_day(self) -> float:
        if not (self.started_at and self.ended_at):
            return 0.0
        days = max((self.ended_at - self.started_at).total_seconds() / 86400, 1e-9)
        return round((self.tasks_created + self.alerts_raised) / days, 2)

    def verdict_line(self) -> str:
        """One-line summary for the UI's "before you enable this" panel."""
        fired = self.tasks_created + self.alerts_raised
        if not fired:
            return (
                "Would not have fired at all over this period. Either the space stayed fine, or "
                "the threshold is too strict."
            )
        parts = [f"Would have fired {fired}x ({self.per_day()}/day)"]
        if self.episode_durations:
            parts.append(f"typical episode {_human(self.mean_episode_duration)}")
            if self.shortest_episode < timedelta(minutes=2):
                parts.append(
                    f"shortest {_human(self.shortest_episode)} - suspiciously brief, check "
                    f"your resolve threshold"
                )
        if self.suppressions:
            parts.append(f"{self.suppressions} trigger(s) suppressed by limits")
        if self.stale_periods:
            parts.append(f"{self.stale_periods} gap(s) with no usable data")
        return "; ".join(parts)


def simulate(
    compiled: CompiledSkill,
    observations: Iterable[Observation],
    *,
    anchor_id: str | None = None,
    tick: timedelta = DEFAULT_TICK,
    record_all_steps: bool = False,
    initial_state: InstanceState | None = None,
) -> SimulationResult:
    """Replay observations through the full evaluate → advance loop.

    Args:
        compiled: the skill to test.
        observations: stored observations, any order (they are sorted here).
        anchor_id: which anchor's instance to simulate. Defaults to the skill's first anchor.
        tick: simulated clock granularity between observations.
        record_all_steps: keep every instant, not just the interesting ones. Useful for charts,
            expensive for a week of history.
        initial_state: start from a known phase, for testing recovery behaviour.

    Returns:
        A `SimulationResult`. Nothing is written anywhere.
    """
    target_anchor = anchor_id or (compiled.anchor_ids[0] if compiled.anchor_ids else "unknown")
    relevant = sorted(
        (o for o in observations if o.source.anchor_id == target_anchor), key=lambda o: o.ts
    )
    result = SimulationResult(skill_id=compiled.skill.id)
    if not relevant:
        return result

    result.started_at = relevant[0].ts
    result.ended_at = relevant[-1].ts
    result.observations_seen = len(relevant)

    store = WindowStore()
    bindings = compiled.signal_keys(target_anchor)
    for key in bindings.values():
        store.ensure(key, compiled.horizon + tick)

    state = initial_state or InstanceState(
        skill_id=compiled.skill.id, anchor_id=target_anchor, phase=SkillPhase.IDLE
    )
    fake_id = 0

    for now in _instants(relevant, tick):
        for observation in _due(relevant, now, tick):
            store.ingest(observation)

        view = store.view(bindings)
        trigger, resolve = evaluate_both(
            compiled.skill.conditions,
            compiled.skill.resolve.conditions if compiled.skill.resolve else None,
            view,
            now,
            staleness_timeout=compiled.staleness_timeout,
        )

        # Single-task focus is a house-wide rule, so the engine passes in a cross-anchor count.
        # In simulation there is one instance, which is a faithful approximation of the common case.
        open_tasks = 1 if state.open_task_id else 0
        decision = advance(
            state,
            compiled,
            trigger,
            resolve,
            EngineContext(now=now, open_tasks_for_skill=open_tasks),
        )
        state = decision.state

        for action in decision.actions:
            state, changed = _apply(action, state, result, now)
            if changed:
                fake_id += 1
                state = (
                    state.attach_task(f"sim-task-{fake_id}")
                    if isinstance(action, CreateTask)
                    else state.attach_alert(f"sim-alert-{fake_id}")
                )

        if record_all_steps or decision.actions:
            result.steps.append(
                SimulationStep(
                    ts=now,
                    anchor_id=target_anchor,
                    phase=state.phase,
                    trigger_matched=trigger.matched,
                    resolve_matched=resolve.matched,
                    evaluable=trigger.is_evaluable,
                    actions=decision.actions,
                    reasons=tuple(trigger.facts() or trigger.failures()),
                )
            )
        store.evict(now)
        result.ticks += 1

    return result


def _apply(
    action: Action,
    state: InstanceState,
    result: SimulationResult,
    now: datetime,
) -> tuple[InstanceState, bool]:
    """Update counters. Returns (state, needs_effect_id)."""
    if isinstance(action, CreateTask):
        result.tasks_created += 1
        return state, True
    if isinstance(action, CreateAlert):
        result.alerts_raised += 1
        return state, True
    if isinstance(action, ResolveTask):
        result.tasks_auto_resolved += 1
        return state, False
    if isinstance(action, ExpireTask):
        result.tasks_expired += 1
        return state, False
    if isinstance(action, ResolveAlert):
        return state, False
    if isinstance(action, Notice):
        result.notices.append(action.message)
        if action.code == "trigger_suppressed":
            result.suppressions += 1
        elif action.code == "signals_stale":
            result.stale_periods += 1
        return state, False
    if hasattr(action, "duration"):  # CloseEpisode / RecordMetricEpisode
        result.episode_durations.append(action.duration)
    return state, False


def _instants(observations: Sequence[Observation], tick: timedelta) -> list[datetime]:
    """Every moment the loop should evaluate: each observation, plus regular ticks between them.

    The ticks are the important half. `for: 15m` and `absent_for: 4h` become true through the
    passage of time, not through the arrival of data, so a replay that only stepped on observations
    would miss exactly the conditions this system is built around.
    """
    start, end = observations[0].ts, observations[-1].ts
    instants = {o.ts for o in observations}
    cursor = start
    guard = 0
    while cursor <= end and guard < 200_000:
        instants.add(cursor)
        cursor += tick
        guard += 1
    return sorted(instants)


def _due(observations: Sequence[Observation], now: datetime, tick: timedelta) -> list[Observation]:
    """Observations landing in (now - tick, now]."""
    return [o for o in observations if now - tick < o.ts <= now]


def suggest_thresholds(result: SimulationResult, compiled: CompiledSkill) -> list[str]:
    """Plain-language tuning advice from a simulation. Shown next to the verdict line."""
    advice: list[str] = []
    skill = compiled.skill

    if result.tasks_created + result.alerts_raised == 0:
        advice.append(
            "Nothing fired. Try lowering the trigger threshold, or shortening the `for:` "
            "duration, then simulate again."
        )
    if result.per_day() > 6 and isinstance(skill.effect, TaskEffect):
        advice.append(
            f"{result.per_day()} triggers per day is a lot. Raise the threshold, lengthen `for:`, "
            f"or set limits.max_per_day to keep the list calm."
        )
    if result.shortest_episode and result.shortest_episode < timedelta(minutes=2):
        advice.append(
            "Some episodes lasted under two minutes, which usually means the resolve threshold is "
            "too close to the trigger threshold. Widen the gap."
        )
    if result.stale_periods:
        advice.append(
            f"{result.stale_periods} period(s) had no usable data. Check the camera's uptime "
            f"before trusting these numbers."
        )
    if (
        isinstance(skill.effect, TaskEffect)
        and skill.effect.mode is TaskMode.BACKLOG
        and result.tasks_created > 10
    ):
        advice.append(
            "This would have added more than ten items to a visible backlog. "
            "`mode: single_task_focus` shows one at a time instead."
        )
    return advice


def _human(delta: timedelta) -> str:
    total = int(delta.total_seconds())
    if total >= 3600:
        return f"{total // 3600}h{(total % 3600) // 60:02d}m"
    if total >= 60:
        return f"{total // 60}m"
    return f"{total}s"


__all__ = [
    "DEFAULT_TICK",
    "SimulationResult",
    "SimulationStep",
    "simulate",
    "suggest_thresholds",
]
