"""The skill engine worker.

    observations (bus) ─┐
                        ├─▶ windows ─▶ evaluate ─▶ advance ─▶ actions ─▶ tasks/alerts/metrics
    1 Hz timer tick ────┘

The tick is not an optimisation, it is a correctness requirement. `for: 15m` and `absent_for: 4h`
become true through the passage of time rather than the arrival of data, so an engine that only woke
on new observations would never fire exactly the conditions this system is built around.

One engine per deployment, enforced by a Redis lock: two would each create a task for every mess. A
second instance runs as a warm standby and takes over when the lock lapses.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from openhup_schemas import (
    ConsumerGroup,
    EventType,
    Observation,
    SkillPhase,
    TaskState,
    Topic,
)
from sqlalchemy import func, select

from .bus import Bus
from .core.config import Settings, load_settings
from .db import (
    AlertRow,
    AnchorRow,
    CameraRow,
    ConsentAskRow,
    MemberRow,
    MemoryPatternRow,
    ObservationRow,
    PresenceWindowRow,
    SkillRow,
    SkillStateRow,
    TaskRow,
    init_engine,
    session_scope,
)
from .identity import consent_question, should_ask
from .llm import PersonalityRenderer, UsageLog, build_provider
from .memory import nudge_text, pattern_due, refresh_patterns
from .memory.patterns import NUDGE_MIN_CONFIDENCE
from .metrics import Rollup
from .notify import Dispatcher, build_channels
from .personality import effective_default_id, load_draw
from .skills.compile import CompiledSkill, compile_all
from .skills.evaluate import evaluate_both
from .skills.fsm import CreateAlert, CreateTask, EngineContext, InstanceState, advance
from .skills.window import Sample, WindowStore
from .tasks import Executor

log = logging.getLogger("openhup.engine")
UTC = UTC


@dataclass
class Engine:
    settings: Settings
    bus: Bus
    renderer: PersonalityRenderer
    dispatcher: Dispatcher

    windows: WindowStore = field(default_factory=WindowStore)
    compiled: dict[str, CompiledSkill] = field(default_factory=dict)
    anchors: dict[str, Any] = field(default_factory=dict)
    states: dict[tuple[str, str], InstanceState] = field(default_factory=dict)
    #: Latest snapshot reference and object inventory per anchor, so a task created on a tick can
    #: still attach the picture and the object list from the last observation.
    context_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    is_leader: bool = False
    _last_rollup_at: datetime | None = None
    _last_pattern_refresh_at: datetime | None = None
    _stopping: asyncio.Event = field(default_factory=asyncio.Event)
    observations_seen: int = 0
    ticks: int = 0

    # -- lifecycle ---------------------------------------------------------------------

    async def load(self) -> None:
        """Load skills and anchors, compile, and size the signal windows."""
        async with session_scope() as session:
            anchor_rows = (await session.execute(select(AnchorRow))).scalars().all()
            skill_rows = (
                (await session.execute(select(SkillRow).where(SkillRow.enabled))).scalars().all()
            )
            state_rows = (await session.execute(select(SkillStateRow))).scalars().all()

        from openhup_schemas import Anchor, Skill

        self.anchors = {row.id: Anchor.model_validate(row.config) for row in anchor_rows}
        skills = []
        for row in skill_rows:
            with contextlib.suppress(Exception):
                skills.append(Skill.model_validate(row.definition))

        compiled, failures = compile_all(skills, anchors=self.anchors)
        self.compiled = {c.skill.id: c for c in compiled}
        for skill_id, error in failures.items():
            log.error("skill %s will not run: %s", skill_id, error.findings[0].message)

        # Window retention is the deepest horizon of any skill reading each signal, so enabling a
        # skill with `absent_for: 4h` automatically deepens the buffers it needs.
        for compiled_skill in self.compiled.values():
            for anchor_id in compiled_skill.anchor_ids:
                for key in compiled_skill.signal_keys(anchor_id).values():
                    self.windows.ensure(key, compiled_skill.horizon)

        self.states = {(row.skill_id, row.anchor_id): _state_from_row(row) for row in state_rows}
        for compiled_skill in self.compiled.values():
            for skill_id, anchor_id in compiled_skill.instances:
                self.states.setdefault(
                    (skill_id, anchor_id),
                    InstanceState(skill_id=skill_id, anchor_id=anchor_id),
                )

        log.info(
            "loaded %d skill(s) over %d anchor(s), %d window(s), %d instance(s)",
            len(self.compiled),
            len(self.anchors),
            len(self.windows.tracked_keys()),
            len(self.states),
        )

    async def warm_start(self) -> None:
        """Refill windows from stored observations.

        Without this, a restart resets every `for: 4h` condition to zero and the engine spends hours
        blind to conditions that were already true. Cheap: one indexed range scan per anchor.
        """
        if not self.settings.engine.warm_start:
            return
        since = datetime.now(tz=UTC) - self.settings.engine.warm_start_window
        anchors = {key.anchor_id for key in self.windows.tracked_keys()}
        if not anchors:
            return

        async with session_scope() as session:
            rows = (
                (
                    await session.execute(
                        select(ObservationRow)
                        .where(ObservationRow.anchor_id.in_(anchors), ObservationRow.ts >= since)
                        .order_by(ObservationRow.ts)
                        .limit(500_000)
                    )
                )
                .scalars()
                .all()
            )

        for row in rows:
            for reading in row.signals:
                key = _signal_key(row.anchor_id, row.detector, reading["key"])
                window = self.windows.get(key)
                if window is not None:
                    window.append(
                        Sample(
                            ts=row.ts,
                            value=reading["value"],
                            confidence=reading.get("confidence"),
                        )
                    )
        log.info("warm start: replayed %d stored observation(s)", len(rows))

    # -- main loop ---------------------------------------------------------------------

    async def run(self) -> int:
        await self.load()
        await self.warm_start()

        tasks = [
            asyncio.create_task(self._leadership_loop(), name="leadership"),
            asyncio.create_task(self._observation_loop(), name="observations"),
            asyncio.create_task(self._tick_loop(), name="tick"),
            asyncio.create_task(self._maintenance_loop(), name="maintenance"),
        ]
        await self._stopping.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.bus.release_leadership(self.settings.engine.leader_lock_key)
        log.info("engine stopped (%d observations, %d ticks)", self.observations_seen, self.ticks)
        return 0

    async def _leadership_loop(self) -> None:
        """Acquire and renew the singleton lock. A standby polls until the leader goes away."""
        key = self.settings.engine.leader_lock_key
        ttl = self.settings.engine.leader_ttl
        while not self._stopping.is_set():
            if self.is_leader:
                if not await self.bus.renew_leadership(key, ttl):
                    log.warning("lost engine leadership; standing by")
                    self.is_leader = False
            else:
                self.is_leader = await self.bus.acquire_leadership(key, ttl)
                if self.is_leader:
                    log.info("acquired engine leadership")
            await asyncio.sleep(max(ttl.total_seconds() / 3, 1))

    async def _observation_loop(self) -> None:
        async for batch in self.bus.consume(Topic.OBSERVATIONS, ConsumerGroup.SKILL_ENGINE):
            if self._stopping.is_set():
                return
            observations = [obs for message in batch if (obs := message.observation())]
            if observations and self.is_leader:
                await self._handle_observations(observations)
            elif observations:
                # Standby: keep windows warm so a takeover is instant, but produce no effects.
                for observation in observations:
                    self.windows.ingest(observation)
            await self.bus.ack(
                Topic.OBSERVATIONS, ConsumerGroup.SKILL_ENGINE, *(m.id for m in batch)
            )

    async def _handle_observations(self, observations: list[Observation]) -> None:
        touched: set[str] = set()
        for observation in observations:
            self.observations_seen += 1
            self.windows.ingest(observation)
            touched.add(observation.source.anchor_id)
            self._remember_context(observation)

        async with session_scope() as session:
            for observation in observations:
                session.add(
                    ObservationRow(
                        id=observation.id,
                        ts=observation.ts,
                        camera_id=observation.source.camera_id,
                        anchor_id=observation.source.anchor_id,
                        detector=observation.detector.name,
                        detector_version=observation.detector.version,
                        signals=[s.model_dump(mode="json") for s in observation.signals],
                        snapshot_ref=observation.media.snapshot_ref if observation.media else None,
                        cost_ms=observation.cost_ms,
                    )
                )
                camera = await session.get(CameraRow, observation.source.camera_id)
                if camera is not None:
                    camera.last_frame_at = observation.ts
                    camera.last_error = None
            if self.settings.identity.enabled:
                await self._track_identity(observations, session, now=datetime.now(tz=UTC))

        await self._evaluate(anchors=touched)

    async def _track_identity(
        self, observations: list[Observation], session: Any, *, now: datetime
    ) -> None:
        """Turn face_id observations into presence windows and consent asks (ADR-016).

        Two rules, both enforced here so no other code path can bend them:

        * Identity is presence, never attribution. A known member opens a presence window; the
          window says they were *in* the room, and nothing anywhere says they *did* anything.
        * An unknown face earns exactly one question per anchor per day. The marker records that
          the question was asked, never what the person looked like.
        """
        # key: (member_id, anchor_id) -> open window row, so a member appearing in two anchors
        # at once keeps two windows.
        open_windows: dict[tuple[str, str], PresenceWindowRow] = {}
        rows = (
            (
                await session.execute(
                    select(PresenceWindowRow).where(PresenceWindowRow.ended_at.is_(None))
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            open_windows[(row.member_id, row.anchor_id)] = row

        anchors_asked_today: set[str] = set()
        ask_rows = (
            (
                await session.execute(
                    select(ConsentAskRow).where(ConsentAskRow.asked_on == now.date())
                )
            )
            .scalars()
            .all()
        )
        for row in ask_rows:
            anchors_asked_today.add(row.anchor_id)

        for observation in observations:
            if observation.detector.name != "face_id":
                continue
            anchor_id = observation.source.anchor_id
            signals = {s.key: s.value for s in observation.signals}
            known = {str(m) for m in signals.get("known_members", []) or []}
            unknown = bool(signals.get("unknown_face", False))

            # Close windows for members no longer reported present in this anchor.
            for (member_id, win_anchor), row in list(open_windows.items()):
                if win_anchor == anchor_id and member_id not in known:
                    row.ended_at = now
                    open_windows.pop((member_id, win_anchor), None)

            for member_id in known:
                row = open_windows.get((member_id, anchor_id))
                if row is None:
                    row = PresenceWindowRow(
                        member_id=member_id,
                        anchor_id=anchor_id,
                        started_at=now,
                        ended_at=None,
                    )
                    session.add(row)
                    open_windows[(member_id, anchor_id)] = row
                member = await session.get(MemberRow, member_id)
                if member is not None:
                    member.last_seen_at = now

            if should_ask(unknown_face=unknown, asked_here_today=anchor_id in anchors_asked_today):
                # One question per anchor per day, however many guests pass through.
                session.add(
                    ConsentAskRow(
                        anchor_id=anchor_id,
                        asked_on=now.date(),
                        answer="no",
                    )
                )
                anchors_asked_today.add(anchor_id)
                await session.flush()
                await self.bus.emit(
                    Topic.SYSTEM_EVENTS,
                    EventType.CONSENT_ASK,
                    {
                        "text": consent_question(),
                        "anchor_id": anchor_id,
                        "anchor_label": (
                            await session.execute(
                                select(AnchorRow.label).where(AnchorRow.id == anchor_id)
                            )
                        ).scalar_one_or_none()
                        or anchor_id,
                    },
                    anchor_id=anchor_id,
                )
        await session.flush()

    def _remember_context(self, observation: Observation) -> None:
        """Cache the latest snapshot, object list, and subregion scores per anchor.

        A task fired by a timer tick still deserves a picture and a sensible micro-step ladder, and
        the tick has no observation of its own to draw them from.
        """
        entry = self.context_cache.setdefault(observation.source.anchor_id, {})
        if observation.media:
            entry["snapshot_ref"] = observation.media.snapshot_ref
        for reading in observation.signals:
            if reading.key == "objects" and isinstance(reading.value, list):
                entry["objects"] = tuple(str(v) for v in reading.value)[:8]
        entry["ts"] = observation.ts

    async def _tick_loop(self) -> None:
        interval = self.settings.engine.tick.total_seconds()
        while not self._stopping.is_set():
            await asyncio.sleep(interval)
            self.ticks += 1
            if self.is_leader:
                with contextlib.suppress(Exception):
                    await self._evaluate()

    async def _evaluate(
        self, *, anchors: set[str] | None = None, now: datetime | None = None
    ) -> None:
        """Evaluate every affected skill instance and apply the resulting actions.

        `now` is injectable for the same reason it is throughout the engine: every temporal decision
        should be reproducible, and tests should not have to sleep.
        """
        now = now or datetime.now(tz=UTC)
        self.windows.evict(now)

        async with session_scope() as session:
            executor = Executor(
                session=session,
                renderer=self.renderer,
                bus=self.bus,
                dispatcher=self.dispatcher,
            )
            open_counts: dict[str, int] = {}

            for (skill_id, anchor_id), state in list(self.states.items()):
                compiled = self.compiled.get(skill_id)
                if compiled is None:
                    continue
                if anchors is not None and anchor_id not in anchors and not _needs_tick(state):
                    continue

                view = self.windows.view(compiled.signal_keys(anchor_id))
                trigger, resolve = evaluate_both(
                    compiled.skill.conditions,
                    compiled.skill.resolve.conditions if compiled.skill.resolve else None,
                    view,
                    now,
                    staleness_timeout=compiled.staleness_timeout,
                )

                if skill_id not in open_counts:
                    open_counts[skill_id] = await executor.open_task_count(skill_id)

                decision = advance(
                    state,
                    compiled,
                    trigger,
                    resolve,
                    EngineContext(
                        now=now,
                        open_tasks_for_skill=open_counts[skill_id],
                        globally_paused=self.settings.engine.paused,
                    ),
                )

                if not decision.actions and decision.state == state:
                    continue

                new_state = decision.state
                cached = self.context_cache.get(anchor_id, {})
                for action in decision.actions:
                    effect_id = await executor.apply(
                        action,
                        skill=compiled.skill,
                        anchor=self.anchors.get(anchor_id),
                        now=now,
                        snapshot_ref=cached.get("snapshot_ref"),
                        objects=cached.get("objects", ()),
                    )
                    if effect_id and isinstance(action, CreateTask):
                        new_state = new_state.attach_task(effect_id)
                        open_counts[skill_id] += 1
                    elif effect_id and isinstance(action, CreateAlert):
                        new_state = new_state.attach_alert(effect_id)

                self.states[(skill_id, anchor_id)] = new_state
                await _persist_state(session, new_state)

                if decision.events:
                    log.info(
                        "%s/%s %s -> %s",
                        skill_id,
                        anchor_id,
                        state.phase.value,
                        new_state.phase.value,
                    )

    async def _maintenance_loop(self) -> None:
        """Expiry, alert repetition, and metric rollups. Every minute is plenty."""
        while not self._stopping.is_set():
            await asyncio.sleep(60)
            if not self.is_leader:
                continue
            now = datetime.now(tz=UTC)
            try:
                await self._refresh_personality()
                rollup_due = self._last_rollup_at is None or (
                    now - self._last_rollup_at >= timedelta(minutes=15)
                )
                rollup_written = 0
                async with session_scope() as session:
                    executor = Executor(
                        session=session,
                        renderer=self.renderer,
                        bus=self.bus,
                        dispatcher=self.dispatcher,
                    )
                    expired = await executor.expire_overdue(now=now)
                    repeated = await executor.repeat_unacknowledged(now=now)
                    if rollup_due:
                        rollup_written = await Rollup(session).run_daily(now=now)
                if rollup_due:
                    self._last_rollup_at = now
                pattern_nudges = await self._pattern_nudge_pass(now=now)
                if expired or repeated or rollup_written or pattern_nudges:
                    log.info(
                        "maintenance: %d expired, %d re-notified, %d metric points, "
                        "%d pattern nudges",
                        expired,
                        repeated,
                        rollup_written,
                        pattern_nudges,
                    )
                await self.dispatcher.release_held(now=now)
            except Exception as exc:
                log.exception("maintenance failed: %s", exc)

    async def _pattern_nudge_pass(self, *, now: datetime) -> int:
        """Refresh learned patterns and speak any that are due. Returns nudges sent.

        A pattern nudge is a nudge *about a skill*, so it obeys that skill's own anti-nag limits -
        there is no hidden global cap for a household to discover and fight. The guards, in order:
        only the leader nudges; a pattern must be active, confident enough, and inside its
        predicted window; the skill that produced it must still exist and be enabled; the skill's
        `quiet_hours`, `cooldown`, and `max_per_day` apply exactly as they do to the skill's own
        triggers; there must be no open task or active alert for that spot (the system is already
        on it); and a cycle that has been nudged is not nudged again until a new episode arrives.
        """
        if not self.is_leader:
            return 0

        refresh_due = self._last_pattern_refresh_at is None or (
            now - self._last_pattern_refresh_at >= timedelta(minutes=15)
        )
        labels = {anchor_id: anchor.label for anchor_id, anchor in self.anchors.items()}
        sent = 0
        async with session_scope() as session:
            if refresh_due:
                patterns = await refresh_patterns(session, now=now, labels=labels)
                self._last_pattern_refresh_at = now
            else:
                patterns = (
                    (
                        await session.execute(
                            select(MemoryPatternRow).where(MemoryPatternRow.status == "active")
                        )
                    )
                    .scalars()
                    .all()
                )

            for pattern in patterns:
                if pattern.confidence < NUDGE_MIN_CONFIDENCE:
                    continue
                due, basis = pattern_due(pattern, now=now)
                if not due:
                    continue
                compiled = self.compiled.get(pattern.skill_id)
                if compiled is None or not compiled.skill.enabled:
                    continue  # the rule that produced this pattern is gone; do not nudge about it
                limits = compiled.skill.limits
                quiet = limits.quiet_hours
                if quiet is not None and quiet.contains(now):
                    continue  # held back entirely; a pattern nudge is never worth a buzz at 2am
                last_nudge_at = pattern.last_nudge_at
                if last_nudge_at is not None:
                    if last_nudge_at.tzinfo is None:
                        last_nudge_at = last_nudge_at.replace(tzinfo=UTC)
                    if now - last_nudge_at < limits.cooldown:
                        continue  # the skill's own cooldown applies to pattern nudges about it
                if (
                    limits.max_per_day is not None
                    and await self._pattern_nudges_today_for(session, pattern.skill_id, now)
                    >= limits.max_per_day
                ):
                    continue
                if await self._spot_is_handled(session, pattern):
                    continue

                text = nudge_text(pattern)
                await self.bus.emit(
                    Topic.SYSTEM_EVENTS,
                    EventType.PATTERN_NUDGE,
                    {"text": text},
                    skill_id=pattern.skill_id,
                    anchor_id=pattern.anchor_id,
                )
                pattern.last_nudge_at = now
                pattern.last_nudge_basis = basis
                sent += 1
            if sent:
                await session.flush()
        return sent

    @staticmethod
    async def _pattern_nudges_today_for(session: Any, skill_id: str, now: datetime) -> int:
        """How many pattern nudges this skill has earned since the start of today (UTC).

        Mirrors the FSM's own `triggers_today` counter: the skill's `max_per_day` means the same
        thing here it means for the skill's triggers.
        """
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        count = await session.execute(
            select(func.count())
            .select_from(MemoryPatternRow)
            .where(
                MemoryPatternRow.skill_id == skill_id,
                MemoryPatternRow.last_nudge_at >= day_start,
            )
        )
        return int(count.scalar() or 0)

    @staticmethod
    async def _spot_is_handled(session: Any, pattern: MemoryPatternRow) -> bool:
        """Is something already watching this spot? A nudge about an open task would be noise."""
        task = (
            await session.execute(
                select(TaskRow.id)
                .where(
                    TaskRow.skill_id == pattern.skill_id,
                    TaskRow.anchor_id == pattern.anchor_id,
                    TaskRow.state.in_(
                        [
                            TaskState.OPEN.value,
                            TaskState.IN_PROGRESS.value,
                            TaskState.SNOOZED.value,
                        ]
                    ),
                )
                .limit(1)
            )
        ).scalar_one_or_none()
        if task is not None:
            return True
        alert = (
            await session.execute(
                select(AlertRow.id)
                .where(
                    AlertRow.skill_id == pattern.skill_id,
                    AlertRow.anchor_id == pattern.anchor_id,
                    AlertRow.state == "active",
                )
                .limit(1)
            )
        ).scalar_one_or_none()
        return alert is not None

    async def _refresh_personality(self) -> None:
        """Pick up a re-drawn or deleted gamble voice without a restart.

        Cheap: one indexed read per minute. When the effective default changed (a re-draw in
        Settings, or the draw deleted), the renderer's settings are swapped so new phrasing uses
        the new voice.
        """
        async with session_scope() as session:
            draw_row = await load_draw(session)
        effective = effective_default_id(self.settings.personality.default_personality, draw_row)
        if self.renderer.settings.default_personality != effective:
            self.renderer.settings = self.renderer.settings.model_copy(
                update={"default_personality": effective}
            )
            log.info("effective default personality is now %s", effective)

    async def reload(self) -> None:
        """Re-read skills and anchors after a change. Windows and FSM state survive."""
        await self.load()

    def stop(self) -> None:
        self._stopping.set()

    def status(self) -> dict[str, Any]:
        phases: dict[str, int] = {}
        for state in self.states.values():
            phases[state.phase.value] = phases.get(state.phase.value, 0) + 1
        return {
            "leader": self.is_leader,
            "observations_seen": self.observations_seen,
            "ticks": self.ticks,
            "skills": len(self.compiled),
            "instances": len(self.states),
            "phases": phases,
            "windows": self.windows.stats,
            "bus": self.bus.stats(),
        }


def _needs_tick(state: InstanceState) -> bool:
    """Does this instance need evaluating even with no new data for its anchor?

    Anything mid-episode or cooling down does: grace periods, auto-expiry, and cooldown release are
    all driven by the clock, not by observations.
    """
    return state.phase in {
        SkillPhase.ACTING,
        SkillPhase.RESOLVING,
        SkillPhase.COOLDOWN,
        SkillPhase.ARMED,
        SkillPhase.STALE,
    }


def _signal_key(anchor_id: str, detector: str, key: str) -> Any:
    from openhup_schemas import SignalKey

    return SignalKey(anchor_id, detector, key)


def _state_from_row(row: SkillStateRow) -> InstanceState:
    return InstanceState(
        skill_id=row.skill_id,
        anchor_id=row.anchor_id,
        phase=SkillPhase(row.phase),
        since=row.since,
        episode_id=row.episode_id,
        episode_opened_at=row.episode_opened_at,
        last_triggered_at=row.last_triggered_at,
        last_resolved_at=row.last_resolved_at,
        triggers_today=row.triggers_today,
        counter_day=row.counter_day.date() if row.counter_day else None,
        open_task_id=row.open_task_id,
        open_alert_id=row.open_alert_id,
        resolve_pending_since=row.resolve_pending_since,
        suppressed_reason=row.suppressed_reason,
        stale_notified=row.stale_notified,
    )


async def _persist_state(session: Any, state: InstanceState) -> None:
    row = (
        await session.execute(
            select(SkillStateRow).where(
                SkillStateRow.skill_id == state.skill_id,
                SkillStateRow.anchor_id == state.anchor_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        row = SkillStateRow(skill_id=state.skill_id, anchor_id=state.anchor_id)
        session.add(row)

    row.phase = state.phase.value
    row.since = state.since
    row.episode_id = state.episode_id
    row.episode_opened_at = state.episode_opened_at
    row.last_triggered_at = state.last_triggered_at
    row.last_resolved_at = state.last_resolved_at
    row.triggers_today = state.triggers_today
    row.counter_day = (
        datetime.combine(state.counter_day, datetime.min.time(), tzinfo=UTC)
        if state.counter_day
        else None
    )
    row.open_task_id = state.open_task_id
    row.open_alert_id = state.open_alert_id
    row.resolve_pending_since = state.resolve_pending_since
    row.suppressed_reason = state.suppressed_reason
    row.stale_notified = state.stale_notified


async def build_engine(settings: Settings) -> Engine:
    init_engine(settings.database)
    bus = Bus(
        url=settings.bus.url,
        observation_maxlen=settings.bus.observation_maxlen,
        block_ms=settings.bus.consumer_block_ms,
        claim_after=settings.bus.claim_after,
        consumer_name="openhup-engine",
    )
    await bus.connect()

    usage = UsageLog()
    provider = None
    try:
        provider = build_provider(settings.llm)
    except Exception as exc:
        log.error("LLM unavailable (%s); using templates", exc)

    personalities: dict[str, Any] = {}
    async with session_scope() as session:
        from openhup_schemas import Personality

        from .db import PersonalityRow

        for row in (await session.execute(select(PersonalityRow))).scalars().all():
            with contextlib.suppress(Exception):
                personalities[row.id] = Personality.model_validate(row.definition)

    engine = Engine(
        settings=settings,
        bus=bus,
        renderer=PersonalityRenderer(
            provider,
            settings=settings.personality,
            personalities=personalities,
            usage=usage,
            timeout_s=settings.llm.timeout.total_seconds(),
        ),
        dispatcher=Dispatcher(
            channels=build_channels(settings.notify.channels),
            max_per_hour=settings.notify.max_per_hour,
        ),
    )
    await engine._refresh_personality()
    return engine


def run(argv: list[str] | None = None) -> int:
    """Entrypoint for `openhup-engine`."""
    parser = argparse.ArgumentParser(description="OpenHup skill engine")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--log-level", default=None)
    parser.add_argument(
        "--once",
        action="store_true",
        help="evaluate once and exit (for cron-style or debugging use)",
    )
    args = parser.parse_args(argv)

    settings = load_settings(*args.config)
    logging.basicConfig(
        level=(args.log_level or settings.log_level).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    async def main() -> int:
        engine = await build_engine(settings)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, engine.stop)

        if args.once:
            await engine.load()
            await engine.warm_start()
            engine.is_leader = True
            await engine._evaluate()
            log.info("single evaluation complete: %s", engine.status())
            return 0
        return await engine.run()

    return asyncio.run(main())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())


__all__ = ["Engine", "build_engine", "run"]
