"""Tasks and alerts.

Note what is missing from the list response: there is no total count. `ux.hide_task_counts` defaults
to true and the API simply does not compute it, because a number next to "things you have not done"
is the single most reliable way to make someone close the tab (see docs/UX_NEURODIVERGENT.md).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from openhup_schemas import AlertState, TaskState, Urgency
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db import AlertRow, EpisodeRow, TaskRow, get_session
from ...notify import Dispatcher
from ...tasks import Executor
from ..state import AppState

router = APIRouter(tags=["tasks"])
UTC = UTC

Session = Annotated[AsyncSession, Depends(get_session)]


def state_of(request: Request) -> AppState:
    return request.app.state.openhup


class MicroStepOut(BaseModel):
    index: int
    text: str
    done: bool = False
    subregion_id: str | None = None


class TaskOut(BaseModel):
    id: str
    skill_id: str
    anchor_id: str
    anchor_label: str
    state: str
    urgency: str
    #: Personality-rendered wording.
    text: str
    #: Tone-free wording. Always present: screen readers and search use it whatever the
    #: personality.
    plain_text: str
    text_source: str
    #: In single-task focus this is the only line the UI shows.
    current_text: str
    micro_steps: list[MicroStepOut] = Field(default_factory=list)
    current_step: int = 0
    progress: float = 0.0
    before_snapshot: str | None = None
    after_snapshot: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    snoozed_until: datetime | None = None
    note: str | None = None
    reopened: bool = False


class TaskUpdate(BaseModel):
    action: Literal["complete", "start", "dismiss", "snooze", "reopen", "false_positive"]
    #: For snooze. Defaults to an hour; "until tomorrow morning" is `minutes` from the client.
    minutes: int | None = Field(default=None, ge=1, le=60 * 24 * 14)
    note: str | None = None


def _task_out(row: TaskRow, anchor_label: str) -> TaskOut:
    steps = [MicroStepOut.model_validate(s) for s in row.micro_steps]
    current = (
        steps[row.current_step].text if steps and 0 <= row.current_step < len(steps) else row.text
    )
    done = sum(1 for s in steps if s.done)
    return TaskOut(
        id=row.id,
        skill_id=row.skill_id,
        anchor_id=row.anchor_id,
        anchor_label=anchor_label,
        state=row.state,
        urgency=row.urgency,
        text=row.text,
        plain_text=row.plain_text,
        text_source=row.text_source,
        current_text=current,
        micro_steps=steps,
        current_step=row.current_step,
        progress=(done / len(steps)) if steps else (1.0 if row.completed_at else 0.0),
        before_snapshot=row.before_snapshot,
        after_snapshot=row.after_snapshot,
        created_at=row.created_at,
        completed_at=row.completed_at,
        snoozed_until=row.snoozed_until,
        note=row.note,
        reopened=row.reopen_count > 0,
    )


def _executor(request: Request, session: AsyncSession) -> Executor:
    state = state_of(request)
    return Executor(
        session=session,
        renderer=state.renderer,
        bus=state.bus,
        dispatcher=state.dispatcher or Dispatcher(channels={}),
    )


@router.get("/tasks", response_model=list[TaskOut])
async def list_tasks(
    request: Request,
    session: Session,
    state: str | None = Query(default="open", description="open | done | all | any TaskState"),
    anchor: str | None = None,
    skill: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[TaskOut]:
    app_state = state_of(request)
    query = select(TaskRow).order_by(TaskRow.created_at.desc()).limit(limit)

    if state == "open":
        query = query.where(
            TaskRow.state.in_(
                [TaskState.OPEN.value, TaskState.IN_PROGRESS.value, TaskState.SNOOZED.value]
            )
        )
    elif state == "done":
        query = query.where(
            TaskRow.state.in_([TaskState.RESOLVED_AUTO.value, TaskState.RESOLVED_MANUAL.value])
        )
    elif state not in {None, "all"}:
        query = query.where(TaskRow.state == state)

    if anchor:
        query = query.where(TaskRow.anchor_id == anchor)
    if skill:
        query = query.where(TaskRow.skill_id == skill)

    rows = (await session.execute(query)).scalars().all()
    return [_task_out(row, _label(app_state, row.anchor_id)) for row in rows]


@router.get("/tasks/next", response_model=TaskOut | None)
async def next_task(request: Request, session: Session) -> TaskOut | None:
    """The one thing to do now.

    This endpoint is the whole single-task-focus mode. The UI in that mode calls only this, so a
    backlog the user has not asked to see cannot appear on screen at all.
    """
    app_state = state_of(request)
    row = await _executor(request, session).next_task()
    return _task_out(row, _label(app_state, row.anchor_id)) if row else None


@router.get("/tasks/{task_id}", response_model=TaskOut)
async def get_task(task_id: str, request: Request, session: Session) -> TaskOut:
    row = await session.get(TaskRow, task_id)
    if row is None:
        raise HTTPException(404, "no such task")
    return _task_out(row, _label(state_of(request), row.anchor_id))


@router.patch("/tasks/{task_id}", response_model=TaskOut)
async def update_task(
    task_id: str,
    request: Request,
    session: Session,
    update: TaskUpdate = Body(...),
) -> TaskOut:
    app_state = state_of(request)
    row = await session.get(TaskRow, task_id)
    if row is None:
        raise HTTPException(404, "no such task")

    now = datetime.now(tz=UTC)
    executor = _executor(request, session)

    if update.action == "complete":
        skill = app_state.skills.get(row.skill_id)
        if skill is None:
            row.state = TaskState.RESOLVED_MANUAL.value
            row.completed_at = now
        else:
            # `still_matching=None`: no fresh observation was taken on this request path. The engine
            # verifies asynchronously if the skill asks for it, so pressing done is never slow.
            row = (
                await executor.complete_manually(task_id, skill=skill, now=now, still_matching=None)
                or row
            )
    elif update.action == "start":
        row.state = TaskState.IN_PROGRESS.value
    elif update.action == "dismiss":
        row.state = TaskState.DISMISSED.value
        row.completed_at = now
        row.note = update.note or row.note
    elif update.action == "snooze":
        row.state = TaskState.SNOOZED.value
        row.snoozed_until = now + timedelta(minutes=update.minutes or 60)
    elif update.action == "reopen":
        row.state = TaskState.OPEN.value
        row.completed_at = None
    elif update.action == "false_positive":
        # The most valuable feedback the system gets: it drives threshold suggestions and the
        # false_positive_rate metric that tells you which skill needs attention.
        row.false_positive = True
        row.state = TaskState.DISMISSED.value
        row.completed_at = now
        row.note = update.note or "marked as a false positive"

    if update.note and update.action != "dismiss":
        row.note = update.note
    await session.flush()
    return _task_out(row, _label(app_state, row.anchor_id))


class AlertOut(BaseModel):
    id: str
    skill_id: str
    anchor_id: str
    anchor_label: str
    state: str
    urgency: str
    text: str
    plain_text: str
    facts: list[str] = Field(default_factory=list)
    snapshot_ref: str | None = None
    created_at: datetime
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    notify_count: int = 0
    delivered_to: list[str] = Field(default_factory=list)


@router.get("/alerts", response_model=list[AlertOut])
async def list_alerts(
    request: Request,
    session: Session,
    state: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[AlertOut]:
    app_state = state_of(request)
    query = select(AlertRow).order_by(AlertRow.created_at.desc()).limit(limit)
    if state:
        query = query.where(AlertRow.state == state)
    rows = (await session.execute(query)).scalars().all()
    return [
        AlertOut(
            **{
                **{
                    c.name: getattr(row, c.name)
                    for c in AlertRow.__table__.columns
                    if c.name in AlertOut.model_fields
                },
                "anchor_label": _label(app_state, row.anchor_id),
            }
        )
        for row in rows
    ]


@router.post("/alerts/{alert_id}/ack", response_model=AlertOut)
async def acknowledge_alert(
    alert_id: str,
    request: Request,
    session: Session,
    by: str = Query(default="user"),
) -> AlertOut:
    """Acknowledge an alert: stop repeating it, but leave it open until it actually resolves."""
    row = await session.get(AlertRow, alert_id)
    if row is None:
        raise HTTPException(404, "no such alert")
    row.state = AlertState.ACKNOWLEDGED.value
    row.acknowledged_at = datetime.now(tz=UTC)
    row.acknowledged_by = by
    await session.flush()
    return AlertOut(
        **{
            **{
                c.name: getattr(row, c.name)
                for c in AlertRow.__table__.columns
                if c.name in AlertOut.model_fields
            },
            "anchor_label": _label(state_of(request), row.anchor_id),
        }
    )


@router.get("/episodes")
async def list_episodes(
    session: Session,
    skill: str | None = None,
    anchor: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """Trigger→resolve cycles. The raw material for every metric and streak."""
    query = select(EpisodeRow).order_by(EpisodeRow.opened_at.desc()).limit(limit)
    if skill:
        query = query.where(EpisodeRow.skill_id == skill)
    if anchor:
        query = query.where(EpisodeRow.anchor_id == anchor)
    rows = (await session.execute(query)).scalars().all()
    return [
        {
            "id": row.id,
            "skill_id": row.skill_id,
            "anchor_id": row.anchor_id,
            "opened_at": row.opened_at.isoformat(),
            "closed_at": row.closed_at.isoformat() if row.closed_at else None,
            "duration_s": row.duration_s,
            "trigger_reasons": row.trigger_reasons,
            "resolve_reasons": row.resolve_reasons,
        }
        for row in rows
    ]


def _label(state: AppState, anchor_id: str) -> str:
    anchor = state.anchors.get(anchor_id)
    return anchor.label if anchor else anchor_id


#: Urgency ordering, exposed so the frontend sorts identically to `/tasks/next`.
URGENCY_ORDER = [u.value for u in sorted(Urgency, key=lambda u: u.rank)]

__all__ = ["URGENCY_ORDER", "router"]
