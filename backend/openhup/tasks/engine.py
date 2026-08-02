"""Task and alert execution: turning FSM actions into rows, text, and notifications.

`skills.fsm.advance()` decides *what* should happen and stays pure. This module does it: writes the
task, asks the personality layer for wording, attaches the snapshot, builds the micro-step ladder,
and hands notifications to the dispatcher.

The rules that live here rather than in the FSM, because they need the database:

* **One task per episode**, enforced by a unique index, so an at-least-once bus cannot double up.
* **Manual completion is verified once.** If the camera disagrees the task reopens exactly once, and
  after that the human wins. Arguing twice is a bug.
* **Micro-step advance is observed, not reported.** A step whose subregion clears is ticked off by
  the next observation, so progress does not depend on remembering to press anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from openhup_schemas import (
    Anchor,
    EventType,
    MicroStepStrategy,
    NotificationRequest,
    Skill,
    TaskEffect,
    TaskMode,
    TaskState,
    TextSource,
    Topic,
    Urgency,
)
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..bus import Bus
from ..db import AlertRow, AnchorRow, EpisodeRow, TaskRow, WinMilestoneRow
from ..llm import PersonalityRenderer
from ..memory import relevant_facts
from ..notify import Dispatcher
from ..skills.fsm import (
    Action,
    CloseEpisode,
    CreateAlert,
    CreateTask,
    ExpireTask,
    Notice,
    OpenEpisode,
    RecordMetricEpisode,
    ResolveAlert,
    ResolveTask,
)
from ..wins import win_candidates

log = logging.getLogger(__name__)
UTC = UTC

#: A task may be reopened at most once after a manual completion the camera disagrees with.
MAX_REOPENS = 1


@dataclass
class Executor:
    """Applies FSM actions. One instance per engine loop."""

    session: AsyncSession
    renderer: PersonalityRenderer
    bus: Bus
    dispatcher: Dispatcher
    ui_base: str = ""

    async def apply(
        self,
        action: Action,
        *,
        skill: Skill,
        anchor: Anchor | None,
        now: datetime,
        snapshot_ref: str | None = None,
        objects: tuple[str, ...] = (),
        subregion_scores: tuple[tuple[str, float], ...] = (),
    ) -> str | None:
        """Perform one action. Returns a new task or alert id when one was created."""
        if isinstance(action, OpenEpisode):
            await self._open_episode(action, now)
            return None
        if isinstance(action, CloseEpisode):
            await self._close_episode(action)
            return None
        if isinstance(action, CreateTask):
            return await self.create_task(
                action,
                skill=skill,
                anchor=anchor,
                now=now,
                snapshot_ref=snapshot_ref,
                objects=objects,
                subregion_scores=subregion_scores,
            )
        if isinstance(action, CreateAlert):
            return await self.create_alert(
                action, skill=skill, anchor=anchor, now=now, snapshot_ref=snapshot_ref
            )
        if isinstance(action, ResolveTask):
            await self.resolve_task(action, now=now, snapshot_ref=snapshot_ref, skill=skill)
            return None
        if isinstance(action, ExpireTask):
            await self.expire_task(action, now=now)
            return None
        if isinstance(action, ResolveAlert):
            await self.resolve_alert(action, now=now)
            return None
        if isinstance(action, RecordMetricEpisode):
            # The rollup worker turns episodes into metric points; nothing to do synchronously.
            await self.bus.emit(
                Topic.METRIC_EVENTS,
                EventType.METRIC_POINT,
                {"metric": action.metric, "duration_s": action.duration.total_seconds()},
                skill_id=action.skill_id,
                anchor_id=action.anchor_id,
            )
            return None
        if isinstance(action, Notice):
            await self.bus.emit(
                Topic.SYSTEM_EVENTS,
                EventType.SKILL_SUPPRESSED
                if action.code == "trigger_suppressed"
                else EventType.SKILL_STALE,
                {"code": action.code, "message": action.message, "severity": action.severity},
            )
            log.info("%s: %s", action.code, action.message)
            return None
        return None

    # -- episodes ----------------------------------------------------------------------

    async def _open_episode(self, action: OpenEpisode, now: datetime) -> None:
        self.session.add(
            EpisodeRow(
                id=action.episode_id,
                skill_id=action.skill_id,
                anchor_id=action.anchor_id,
                opened_at=action.opened_at,
                trigger_reasons=list(action.trigger_reasons),
            )
        )
        await self.session.flush()

    async def _close_episode(self, action: CloseEpisode) -> None:
        await self.session.execute(
            update(EpisodeRow)
            .where(EpisodeRow.id == action.episode_id)
            .values(
                closed_at=action.closed_at,
                resolve_reasons=list(action.resolve_reasons),
                duration_s=action.duration.total_seconds(),
            )
        )

    # -- tasks -------------------------------------------------------------------------

    async def create_task(
        self,
        action: CreateTask,
        *,
        skill: Skill,
        anchor: Anchor | None,
        now: datetime,
        snapshot_ref: str | None = None,
        objects: tuple[str, ...] = (),
        subregion_scores: tuple[tuple[str, float], ...] = (),
    ) -> str | None:
        effect = skill.effect
        if not isinstance(effect, TaskEffect):
            return None

        label = anchor.label if anchor else action.anchor_id
        # Facts the household taught the assistant that touch this task - "call the spare room the
        # junk room" - phrased as context for the personality layer. Local retrieval, never a model.
        memory = await relevant_facts(self.session, query=f"{effect.title_hint} {label}")
        rendered = await self.renderer.task(
            title_hint=effect.title_hint,
            anchor_label=label,
            urgency=effect.urgency,
            personality_id=skill.effective_personality,
            objects=objects,
            memory=memory,
        )
        steps = await self._build_ladder(effect, skill, anchor, label, objects, subregion_scores)

        row = TaskRow(
            skill_id=action.skill_id,
            anchor_id=action.anchor_id,
            episode_id=action.episode_id,
            state=(
                TaskState.PROPOSED.value if effect.require_confirmation else TaskState.OPEN.value
            ),
            urgency=effect.urgency.value,
            text=rendered.text,
            plain_text=rendered.plain,
            text_source=rendered.source.value,
            micro_steps=steps,
            before_snapshot=snapshot_ref,
            expires_at=now + effect.expires_after if effect.expires_after else None,
        )
        self.session.add(row)
        try:
            await self.session.flush()
        except IntegrityError:
            # The unique index on episode_id did its job: this observation was redelivered.
            await self.session.rollback()
            log.debug("task for episode %s already exists; ignoring", action.episode_id)
            return None

        await self.bus.emit(
            Topic.TASK_EVENTS,
            EventType.TASK_CREATED,
            {"task_id": row.id, "text": row.text, "urgency": row.urgency},
            skill_id=action.skill_id,
            anchor_id=action.anchor_id,
            episode_id=action.episode_id,
            idempotency_key=action.idempotency_key,
        )
        await self._notify_task(row, effect, label, snapshot_ref)
        return row.id

    async def _build_ladder(
        self,
        effect: TaskEffect,
        skill: Skill,
        anchor: Anchor | None,
        label: str,
        objects: tuple[str, ...],
        subregion_scores: tuple[tuple[str, float], ...],
    ) -> list[dict[str, Any]]:
        """Build the micro-step ladder.

        Spatial wins whenever the anchor has subregions: "just clear the left third" is verifiable
        by the camera, needs no model, and the worst region goes first so step one is the most
        visibly satisfying. The LLM is the fallback, not the first choice.
        """
        if not effect.micro_steps.enabled:
            return []

        count = effect.micro_steps.count
        spatial_ok = (
            anchor is not None
            and anchor.subregions
            and effect.micro_steps.strategy in {MicroStepStrategy.AUTO, MicroStepStrategy.SPATIAL}
        )

        if effect.micro_steps.strategy is MicroStepStrategy.EXPLICIT and effect.micro_steps.steps:
            texts = effect.micro_steps.steps[:count]
            return [
                {"index": i, "text": text, "subregion_id": None, "done": False}
                for i, text in enumerate(texts)
            ]

        if spatial_ok:
            assert anchor is not None
            by_id = {s.id: s for s in anchor.subregions}
            ordered = (
                [by_id[sid] for sid, _ in subregion_scores if sid in by_id]
                if subregion_scores
                else anchor.ordered_subregions()
            )
            chosen = ordered[:count] or anchor.ordered_subregions()[:count]
            labels = [s.label for s in chosen]
            texts = await self.renderer.micro_steps(
                title_hint=effect.title_hint,
                anchor_label=label,
                count=len(labels),
                personality_id=skill.effective_personality,
                subregion_labels=labels,
            )
            return [
                {
                    "index": i,
                    "text": text,
                    "subregion_id": region.id,
                    "done": False,
                    "baseline_score": next(
                        (score for sid, score in subregion_scores if sid == region.id), None
                    ),
                }
                for i, (text, region) in enumerate(zip(texts, chosen, strict=False))
            ]

        texts = await self.renderer.micro_steps(
            title_hint=effect.title_hint,
            anchor_label=label,
            count=count,
            personality_id=skill.effective_personality,
            objects=objects,
        )
        return [
            {"index": i, "text": text, "subregion_id": None, "done": False}
            for i, text in enumerate(texts)
        ]

    async def _notify_task(
        self, row: TaskRow, effect: TaskEffect, label: str, snapshot_ref: str | None
    ) -> None:
        if row.state == TaskState.PROPOSED.value:
            return  # not on the list yet; nothing to announce
        step = row.micro_steps[row.current_step]["text"] if row.micro_steps else None
        await self.dispatcher.dispatch(
            NotificationRequest(
                channels=list(effect.channels),
                title=label,
                body=step or row.text,
                urgency=effect.urgency,
                snapshot_ref=snapshot_ref,
                link=f"/tasks/{row.id}",
                # One buzz per episode, however many times the engine re-evaluates it.
                dedupe_key=f"{row.skill_id}:{row.anchor_id}:{row.episode_id}",
            )
        )

    async def resolve_task(
        self,
        action: ResolveTask,
        *,
        now: datetime,
        snapshot_ref: str | None = None,
        skill: Skill | None = None,
    ) -> None:
        row = await self.session.get(TaskRow, action.task_id)
        if row is None or TaskState(row.state).is_terminal:
            return
        row.state = (
            TaskState.RESOLVED_AUTO.value if action.auto else TaskState.RESOLVED_MANUAL.value
        )
        row.completed_at = now
        row.after_snapshot = snapshot_ref or row.after_snapshot
        for step in row.micro_steps:
            step["done"] = True
        await self.session.flush()

        await self.bus.emit(
            Topic.TASK_EVENTS,
            EventType.TASK_COMPLETED,
            {
                "task_id": row.id,
                "auto": action.auto,
                "before": row.before_snapshot,
                "after": row.after_snapshot,
            },
            skill_id=row.skill_id,
            anchor_id=row.anchor_id,
            episode_id=row.episode_id,
        )
        if skill is not None:
            await self._celebrate_win(row, skill=skill, now=now)

    async def expire_task(self, action: ExpireTask, *, now: datetime) -> None:
        """Quietly retire a task.

        Deliberately silent: no notification, no summary, no mention of it later. A task that aged
        out is not a failure to be reported back, it is a task that stopped being relevant.
        """
        row = await self.session.get(TaskRow, action.task_id)
        if row is None or TaskState(row.state).is_terminal:
            return
        row.state = TaskState.EXPIRED.value
        row.completed_at = now
        row.note = action.reason
        await self.session.flush()
        await self.bus.emit(
            Topic.TASK_EVENTS,
            EventType.TASK_UPDATED,
            {"task_id": row.id, "state": row.state, "reason": action.reason},
            skill_id=row.skill_id,
            anchor_id=row.anchor_id,
        )

    async def complete_manually(
        self,
        task_id: str,
        *,
        skill: Skill,
        now: datetime,
        still_matching: bool | None = None,
    ) -> TaskRow | None:
        """Handle a user pressing done.

        `still_matching` is the fresh verification, or None when none was taken. If the camera
        disagrees and this task has not been reopened before, it reopens once with gentler wording.
        Otherwise the human is right - and after one disagreement, the human is right regardless.
        """
        row = await self.session.get(TaskRow, task_id)
        if row is None or TaskState(row.state).is_terminal:
            return row

        spec = skill.resolve
        verify = bool(spec and spec.verify_on_manual_complete)
        if verify and still_matching and row.reopen_count < MAX_REOPENS:
            row.reopen_count += 1
            row.state = TaskState.OPEN.value
            row.note = "Looks like there is still something there - reopened once, just in case."
            await self.session.flush()
            await self.bus.emit(
                Topic.TASK_EVENTS,
                EventType.TASK_REOPENED,
                {"task_id": row.id, "reopen_count": row.reopen_count},
                skill_id=row.skill_id,
                anchor_id=row.anchor_id,
            )
            return row

        row.state = TaskState.RESOLVED_MANUAL.value
        row.completed_at = now
        for step in row.micro_steps:
            step["done"] = True
        await self.session.flush()
        await self.bus.emit(
            Topic.TASK_EVENTS,
            EventType.TASK_COMPLETED,
            {"task_id": row.id, "auto": False},
            skill_id=row.skill_id,
            anchor_id=row.anchor_id,
        )
        await self._celebrate_win(row, skill=skill, now=now)
        return row

    async def advance_steps(
        self,
        task_id: str,
        subregion_scores: dict[str, float],
        *,
        clear_below: float = 0.3,
        now: datetime | None = None,
    ) -> bool:
        """Tick off completed micro-steps from observed subregion scores.

        This is what makes the ladder work for someone whose difficulty is starting, not finishing:
        clearing the left third is noticed, credited, and the next step appears - with no
        interaction at all. Partial progress is permanent even if the session stops there.
        """
        row = await self.session.get(TaskRow, task_id)
        if row is None or not row.micro_steps or TaskState(row.state).is_terminal:
            return False

        changed = False
        for step in row.micro_steps:
            region = step.get("subregion_id")
            if step.get("done") or not region or region not in subregion_scores:
                continue
            if subregion_scores[region] <= clear_below:
                step["done"] = True
                step["done_at"] = (now or datetime.now(tz=UTC)).isoformat()
                changed = True

        if changed:
            pending = [s for s in row.micro_steps if not s.get("done")]
            row.current_step = pending[0]["index"] if pending else len(row.micro_steps) - 1
            row.state = TaskState.IN_PROGRESS.value if pending else row.state
            await self.session.flush()
            await self.bus.emit(
                Topic.TASK_EVENTS,
                EventType.TASK_STEP_ADVANCED,
                {
                    "task_id": row.id,
                    "current_step": row.current_step,
                    "done": sum(1 for s in row.micro_steps if s.get("done")),
                    "total": len(row.micro_steps),
                },
                skill_id=row.skill_id,
                anchor_id=row.anchor_id,
            )
        return changed

    # -- alerts ------------------------------------------------------------------------

    async def create_alert(
        self,
        action: CreateAlert,
        *,
        skill: Skill,
        anchor: Anchor | None,
        now: datetime,
        snapshot_ref: str | None = None,
    ) -> str | None:
        effect = skill.effect
        label = anchor.label if anchor else action.anchor_id
        urgency = getattr(effect, "urgency", Urgency.HIGH)

        rendered = self.renderer.alert(
            facts=list(action.facts),
            anchor_label=label,
            urgency=urgency,
            personality_id=skill.effective_personality,
            summary=getattr(effect, "title_hint", None),
        )
        row = AlertRow(
            skill_id=action.skill_id,
            anchor_id=action.anchor_id,
            episode_id=action.episode_id,
            urgency=urgency.value,
            text=rendered.text,
            plain_text=rendered.plain,
            text_source=TextSource.TEMPLATE.value,
            facts=list(action.facts),
            snapshot_ref=snapshot_ref,
        )
        self.session.add(row)
        try:
            await self.session.flush()
        except IntegrityError:
            await self.session.rollback()
            return None

        results = await self.dispatcher.dispatch(
            NotificationRequest(
                channels=list(getattr(effect, "channels", [])),
                title=f"{label}: {rendered.plain}",
                body="\n".join(action.facts) or rendered.plain,
                urgency=urgency,
                snapshot_ref=snapshot_ref,
                link=f"/alerts/{row.id}",
            )
        )
        row.notify_count = sum(1 for r in results if r.ok and r.status == "sent")
        row.last_notified_at = now
        row.delivered_to = [r.channel for r in results if r.ok]
        await self.session.flush()

        await self.bus.emit(
            Topic.ALERT_EVENTS,
            EventType.ALERT_RAISED,
            {"alert_id": row.id, "text": row.plain_text, "urgency": row.urgency},
            skill_id=action.skill_id,
            anchor_id=action.anchor_id,
            episode_id=action.episode_id,
            idempotency_key=action.idempotency_key,
        )
        return row.id

    async def resolve_alert(self, action: ResolveAlert, *, now: datetime) -> None:
        row = await self.session.get(AlertRow, action.alert_id)
        if row is None or row.state in {"resolved", "expired"}:
            return
        row.state = "resolved"
        row.resolved_at = now
        await self.session.flush()
        await self.bus.emit(
            Topic.ALERT_EVENTS,
            EventType.ALERT_RESOLVED,
            {"alert_id": row.id},
            skill_id=row.skill_id,
            anchor_id=row.anchor_id,
        )

    # -- wins -------------------------------------------------------------------------

    async def _celebrate_win(self, row: TaskRow, *, skill: Skill, now: datetime) -> None:
        """Notice and announce a win: this anchor stayed clear for whole days (ADR-015).

        At most one win per resolution, computed purely from the episode history. The
        `win_milestones` ledger dedupes on (anchor, kind, value), so a band or a 90-day record is
        celebrated when it first happens and never again. Quiet hours suppress the *spoken* note,
        never the milestone itself - a 2am clear is still a clear, it is just not announced twice.
        """
        episodes = (
            (
                await self.session.execute(
                    select(EpisodeRow)
                    .where(
                        EpisodeRow.skill_id == row.skill_id,
                        EpisodeRow.anchor_id == row.anchor_id,
                    )
                    .order_by(EpisodeRow.opened_at)
                )
            )
            .scalars()
            .all()
        )
        if not episodes:
            return
        label = await self._anchor_label(row.anchor_id)
        wins = win_candidates(episodes, label=label, now=now)
        if not wins:
            return
        win = wins[0]

        already = (
            await self.session.execute(
                select(WinMilestoneRow.id).where(
                    WinMilestoneRow.anchor_id == row.anchor_id,
                    WinMilestoneRow.kind == win.kind,
                    WinMilestoneRow.value == win.value,
                )
            )
        ).scalar_one_or_none()
        if already is not None:
            return

        milestone = WinMilestoneRow(
            anchor_id=row.anchor_id,
            kind=win.kind,
            value=win.value,
            days=win.days,
            summary=win.summary,
        )
        self.session.add(milestone)

        quiet = skill.limits.quiet_hours
        if quiet is None or not quiet.contains(now):
            rendered = await self.renderer.win(
                anchor_label=label,
                days=win.days,
                record=win.record,
                personality_id=skill.effective_personality,
            )
            await self.bus.emit(
                Topic.SYSTEM_EVENTS,
                EventType.WIN_NOTE,
                {
                    "text": rendered.text,
                    "plain": rendered.plain,
                    "days": round(win.days, 1),
                    "record": win.record,
                },
                skill_id=row.skill_id,
                anchor_id=row.anchor_id,
                episode_id=row.episode_id,
            )
            milestone.spoken_at = now

    async def _anchor_label(self, anchor_id: str) -> str:
        """The anchor's human label, falling back to its id when the anchor is gone."""
        label = (
            await self.session.execute(select(AnchorRow.label).where(AnchorRow.id == anchor_id))
        ).scalar_one_or_none()
        return label or anchor_id

    async def repeat_unacknowledged(self, *, now: datetime) -> int:
        """Re-notify active, unacknowledged alerts whose skill asks for repetition."""
        rows = (
            (
                await self.session.execute(
                    select(AlertRow).where(AlertRow.state == "active").limit(50)
                )
            )
            .scalars()
            .all()
        )
        repeated = 0
        for row in rows:
            if row.last_notified_at and now - row.last_notified_at < timedelta(minutes=2):
                continue
            await self.dispatcher.dispatch(
                NotificationRequest(
                    channels=[],
                    title=f"Still unresolved: {row.plain_text}",
                    body="\n".join(row.facts),
                    urgency=Urgency(row.urgency),
                    link=f"/alerts/{row.id}",
                )
            )
            row.notify_count += 1
            row.last_notified_at = now
            repeated += 1
        return repeated

    # -- queries used by the FSM -------------------------------------------------------

    async def open_task_count(self, skill_id: str) -> int:
        """Open tasks for a skill across every anchor. Enforces single-task focus house-wide."""
        result = await self.session.execute(
            select(func.count())
            .select_from(TaskRow)
            .where(
                TaskRow.skill_id == skill_id,
                TaskRow.state.in_([TaskState.OPEN.value, TaskState.IN_PROGRESS.value]),
            )
        )
        return int(result.scalar() or 0)

    async def next_task(self, *, single_focus: bool = True) -> TaskRow | None:
        """The one thing to do now, for GET /tasks/next.

        Ordered by urgency then age. In single-focus mode the UI shows only this, which is the
        whole point of the mode: a list you cannot see cannot overwhelm you.
        """
        urgency_rank = {
            "critical": 0,
            "high": 1,
            "normal": 2,
            "low": 3,
            "info": 4,
        }
        rows = (
            (
                await self.session.execute(
                    select(TaskRow)
                    .where(TaskRow.state.in_([TaskState.OPEN.value, TaskState.IN_PROGRESS.value]))
                    .order_by(TaskRow.created_at)
                    .limit(50)
                )
            )
            .scalars()
            .all()
        )
        visible = [
            row
            for row in rows
            if not row.snoozed_until or row.snoozed_until <= datetime.now(tz=UTC)
        ]
        if not visible:
            return None
        return min(visible, key=lambda r: (urgency_rank.get(r.urgency, 5), r.created_at))

    async def expire_overdue(self, *, now: datetime) -> int:
        """Retire tasks past their expiry. Quietly, and without comment."""
        rows = (
            (
                await self.session.execute(
                    select(TaskRow).where(
                        TaskRow.expires_at.is_not(None),
                        TaskRow.expires_at <= now,
                        TaskRow.state.in_(
                            [
                                TaskState.OPEN.value,
                                TaskState.IN_PROGRESS.value,
                                TaskState.SNOOZED.value,
                                TaskState.PROPOSED.value,
                            ]
                        ),
                    )
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            row.state = TaskState.EXPIRED.value
            row.completed_at = now
        return len(rows)


def is_single_focus(skill: Skill) -> bool:
    return isinstance(skill.effect, TaskEffect) and skill.effect.mode is TaskMode.SINGLE_TASK_FOCUS


__all__ = ["MAX_REOPENS", "Executor", "is_single_focus"]
