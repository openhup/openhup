"""Metrics, goals, personalities, notification channels, and system health."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from openhup_schemas import (
    BUILTIN_METRICS,
    Goal,
    GoalDirection,
    Personality,
    TaskState,
)
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ... import __version__
from ...db import (
    AlertRow,
    CameraRow,
    GoalRow,
    LLMCallRow,
    MetricPointRow,
    NotificationRow,
    PersonalityRow,
    TaskRow,
    WinMilestoneRow,
    get_session,
)
from ...personality import clear, draw, load_draw
from ..state import AppState

router = APIRouter(tags=["insights"])
UTC = UTC

Session = Annotated[AsyncSession, Depends(get_session)]


def state_of(request: Request) -> AppState:
    return request.app.state.openhup


# --------------------------------------------------------------------------------------
# Metrics and goals
# --------------------------------------------------------------------------------------


@router.get("/metrics/catalog")
async def metric_catalog() -> dict[str, str]:
    """Built-in metrics and what they mean. `nag_index` is the one to watch."""
    return BUILTIN_METRICS


@router.get("/metrics/series")
async def metric_series(
    session: Session,
    metric: str = Query(...),
    days: int = Query(default=30, ge=1, le=365),
    anchor: str | None = None,
) -> dict[str, Any]:
    since = datetime.now(tz=UTC) - timedelta(days=days)
    query = (
        select(MetricPointRow)
        .where(MetricPointRow.metric == metric, MetricPointRow.ts >= since)
        .order_by(MetricPointRow.ts)
    )
    if anchor:
        query = query.where(MetricPointRow.anchor_id == anchor)
    rows = (await session.execute(query)).scalars().all()
    values = [row.value for row in rows]
    return {
        "metric": metric,
        "days": days,
        "points": [
            {"ts": row.ts.isoformat(), "value": row.value, "anchor_id": row.anchor_id}
            for row in rows
        ],
        "total": sum(values),
        "mean": (sum(values) / len(values)) if values else 0.0,
        "description": BUILTIN_METRICS.get(metric, ""),
    }


class GoalIn(BaseModel):
    id: str
    label: str
    metric: str
    target: float
    direction: GoalDirection = GoalDirection.UP
    window_days: int = Field(default=7, ge=1, le=90)
    anchor_id: str | None = None
    include_in_report: bool = True


@router.get("/metrics/goals")
async def list_goals(session: Session) -> list[dict[str, Any]]:
    """Goals with current progress. Never returns a "failed" state - see `GoalStatus`."""
    rows = (await session.execute(select(GoalRow).where(GoalRow.enabled))).scalars().all()
    out = []
    for row in rows:
        goal = Goal(
            id=row.id,
            label=row.label,
            metric=row.metric,
            target=row.target,
            direction=GoalDirection(row.direction),
            window=timedelta(seconds=row.window_s),
            anchor_id=row.anchor_id,
            include_in_report=row.include_in_report,
        )
        since = datetime.now(tz=UTC) - goal.window
        points = (
            (
                await session.execute(
                    select(MetricPointRow.value).where(
                        MetricPointRow.metric == goal.metric, MetricPointRow.ts >= since
                    )
                )
            )
            .scalars()
            .all()
        )
        actual = (
            sum(points)
            if goal.direction is GoalDirection.UP
            else ((sum(points) / len(points)) if points else 0.0)
        )
        progress = goal.evaluate(actual, samples=len(points))
        out.append(
            {
                "goal": goal.model_dump(mode="json"),
                "actual": round(actual, 2),
                "ratio": progress.ratio,
                "status": progress.status.value,
                "samples": len(points),
            }
        )
    return out


@router.post("/metrics/goals", status_code=201)
async def create_goal(session: Session, payload: GoalIn) -> dict[str, str]:
    if await session.get(GoalRow, payload.id):
        raise HTTPException(409, "goal already exists")
    session.add(
        GoalRow(
            id=payload.id,
            label=payload.label,
            metric=payload.metric,
            target=payload.target,
            direction=payload.direction.value,
            window_s=payload.window_days * 86400,
            anchor_id=payload.anchor_id,
            include_in_report=payload.include_in_report,
        )
    )
    await session.flush()
    return {"created": payload.id}


@router.delete("/metrics/goals/{goal_id}", status_code=204)
async def delete_goal(goal_id: str, session: Session) -> None:
    row = await session.get(GoalRow, goal_id)
    if row is None:
        raise HTTPException(404, "no such goal")
    await session.delete(row)


@router.get("/metrics/report/weekly")
async def weekly_report(request: Request, session: Session) -> dict[str, Any]:
    """The coaching summary.

    Note what is counted and what is not: created, resolved, auto-resolved, notifications. There is
    no "missed" or "overdue" number anywhere, and the narrative is filtered for `backlog_counts`
    before it is shown (see `openhup.llm.safety`).
    """
    state = state_of(request)
    now = datetime.now(tz=UTC)
    start = now - timedelta(days=7)

    created = await _count(session, TaskRow, TaskRow.created_at >= start)
    resolved = await _count(
        session,
        TaskRow,
        TaskRow.completed_at >= start,
        TaskRow.state.in_([TaskState.RESOLVED_AUTO.value, TaskState.RESOLVED_MANUAL.value]),
    )
    auto = await _count(
        session,
        TaskRow,
        TaskRow.completed_at >= start,
        TaskRow.state == TaskState.RESOLVED_AUTO.value,
    )
    alerts = await _count(session, AlertRow, AlertRow.created_at >= start)
    notifications = await _count(session, NotificationRow, NotificationRow.sent_at >= start)

    facts = {
        "tasks_completed": resolved,
        "closed_by_the_camera": auto,
        "alerts": alerts,
    }
    goals = await list_goals(session)
    for goal in goals:
        if goal["goal"]["include_in_report"]:
            facts[goal["goal"]["label"]] = f"{goal['actual']} of {goal['goal']['target']}"

    plain = (
        f"This week: {resolved} task(s) done, {auto} closed by the camera on its own, "
        f"{alerts} alert(s)."
    )
    rendered = await state.renderer.weekly(facts, plain_summary=plain)

    return {
        "period_start": start.isoformat(),
        "period_end": now.isoformat(),
        "tasks_created": created,
        "tasks_resolved": resolved,
        "tasks_auto_resolved": auto,
        "alerts_raised": alerts,
        "notifications_sent": notifications,
        # Notifications per completed task. If this climbs, the thresholds are wrong and OpenHup is
        # becoming the thing it was built to avoid.
        "nag_index": round(notifications / resolved, 2) if resolved else 0.0,
        "completion_rate": round(resolved / created, 2) if created else 0.0,
        "goals": goals,
        "narrative": rendered.text,
        "plain_summary": rendered.plain,
        "narrative_source": rendered.source.value,
    }


# --------------------------------------------------------------------------------------
# Personalities
# --------------------------------------------------------------------------------------


@router.get("/personalities")
async def list_personalities(request: Request) -> list[dict[str, Any]]:
    state = state_of(request)
    return [
        {
            "id": personality.id,
            "display_name": personality.display_name,
            "description": personality.description,
            "intensity": personality.intensity,
            "effective_intensity": state.renderer.resolve(personality.id).intensity,
            "tone": personality.tone,
            "templates_only": personality.templates_only,
            "boundaries": [b.value for b in personality.boundaries.never],
            "max_words": personality.boundaries.max_words,
        }
        for personality in sorted(state.personalities.values(), key=lambda p: p.id)
    ]


@router.post("/personalities/{personality_id}/preview")
async def preview_personality(personality_id: str, request: Request) -> dict[str, str]:
    """Sample output, so tone can be chosen by seeing it rather than by reading a label."""
    state = state_of(request)
    if personality_id not in state.personalities:
        raise HTTPException(404, "no such personality")
    return await state.renderer.preview(personality_id)


@router.put("/personalities/{personality_id}")
async def upsert_personality(
    personality_id: str,
    request: Request,
    session: Session,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """Create or update a personality.

    The boundary audit runs on save and reports anything in the custom templates that a stricter
    household member's settings would reject - better to know now than when it is filtered silently.
    """
    from ...llm import audit_personality

    state = state_of(request)
    personality = Personality.model_validate({**payload, "id": personality_id})
    row = await session.get(PersonalityRow, personality_id)
    if row is None:
        row = PersonalityRow(
            id=personality_id, display_name=personality.display_name, definition={}, builtin=False
        )
        session.add(row)
    if row.builtin:
        raise HTTPException(
            409,
            "this is a shipped preset and is replaced on upgrade. Copy it under a new id instead.",
        )
    row.definition = personality.model_dump(mode="json")
    row.display_name = personality.display_name
    await session.flush()
    state.personalities[personality_id] = personality
    state.renderer.personalities = state.personalities

    concerns = {
        field: audit_personality(getattr(personality.templates, field))
        for field in ("task", "task_step", "alert", "task_done", "nudge", "win")
    }
    return {
        "saved": personality_id,
        "template_concerns": {k: list(v) for k, v in concerns.items() if v},
    }


@router.get("/personality/draw")
async def get_personality_draw(request: Request, session: Session) -> dict[str, Any]:
    """The state of the personality gamble (ADR-014).

    `drawn` names the mystery voice; the UI keeps it hidden behind a Reveal button so the gamble
    is discovered by living with it, not announced at setup. None means the config default speaks.
    """
    state = state_of(request)
    row = await load_draw(session)
    return {
        "drawn": row.personality_id if row else None,
        "reroll_count": row.reroll_count if row else 0,
        "pool": state.settings.personality.gamble_pool,
        "gamble_enabled": state.settings.personality.gamble,
    }


@router.post("/personality/draw", status_code=201)
async def draw_personality(request: Request, session: Session) -> dict[str, Any]:
    """Draw the mystery voice, or re-draw it.

    A re-draw is an explicit household choice (the Settings button), not something the system
    does on its own - the voice is stable until someone asks for a new one.
    """
    state = state_of(request)
    try:
        row = await draw(
            session,
            pool=state.settings.personality.gamble_pool,
            available=list(state.personalities),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    state.personality_draw = row
    state._apply_draw()
    return {
        "drawn": row.personality_id,
        "reroll_count": row.reroll_count,
        "pool": state.settings.personality.gamble_pool,
    }


@router.delete("/personality/draw", status_code=204)
async def clear_personality_draw(request: Request, session: Session) -> None:
    """Stop the gamble: the configured default personality speaks again."""
    state = state_of(request)
    await clear(session)
    state.personality_draw = None
    state._apply_draw()


@router.get("/personality/wins")
async def list_wins(request: Request, session: Session) -> dict[str, Any]:
    """The ledger of wins the assistant has noticed (ADR-015).

    Reviewable like facts and patterns: every celebrated milestone - an anchor staying clear for
    whole days - with its tone-free summary. Only the most recent are kept in the response.
    """
    rows = (
        (
            await session.execute(
                select(WinMilestoneRow).order_by(WinMilestoneRow.created_at.desc()).limit(50)
            )
        )
        .scalars()
        .all()
    )
    return {
        "wins": [
            {
                "id": row.id,
                "anchor_id": row.anchor_id,
                "kind": row.kind,
                "days": row.days,
                "summary": row.summary,
                "spoken": row.spoken_at is not None,
                "achieved_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
    }


@router.delete("/personality/wins/{win_id}", status_code=204)
async def delete_win(win_id: str, session: Session) -> None:
    """Forget a win. A deleted ledger row is gone, and the same milestone can be celebrated
    again if it ever repeats - same semantics as forgetting a fact."""
    row = await session.get(WinMilestoneRow, win_id)
    if row is None:
        raise HTTPException(404, "no such win")
    await session.delete(row)


# --------------------------------------------------------------------------------------
# Notification channels
# --------------------------------------------------------------------------------------


@router.get("/notify/channels")
async def list_channels(request: Request) -> list[dict[str, Any]]:
    state = state_of(request)
    return [
        {
            "id": channel.id,
            "type": type(channel).__name__.replace("Channel", "").lower(),
            "enabled": channel.enabled,
            "min_urgency": channel.min_urgency.value,
            "supports_images": channel.supports_images,
        }
        for channel in state.dispatcher.channels.values()
    ]


@router.post("/notify/channels/{channel_id}/test")
async def test_channel(channel_id: str, request: Request) -> dict[str, Any]:
    state = state_of(request)
    channel = state.dispatcher.channels.get(channel_id)
    if channel is None:
        raise HTTPException(404, "no such channel")
    result = await channel.test()
    return {
        "channel": result.channel,
        "ok": result.ok,
        "status": result.status,
        "detail": result.detail,
    }


@router.get("/notify/held")
async def held_notifications(request: Request) -> list[dict[str, Any]]:
    """Notifications waiting for quiet hours to end. Held, never dropped."""
    state = state_of(request)
    return [
        {"title": r.title, "body": r.body, "urgency": r.urgency.value, "link": r.link}
        for r in state.dispatcher.held
    ]


# --------------------------------------------------------------------------------------
# System
# --------------------------------------------------------------------------------------


@router.get("/system/info")
async def system_info(request: Request) -> dict[str, Any]:
    state = state_of(request)
    settings = state.settings
    return {
        "version": __version__,
        "instance_name": settings.instance_name,
        "skills": {
            "total": len(state.skills),
            "enabled": len(state.enabled_compiled()),
            "failing": len(state.compile_failures),
        },
        "anchors": len(state.anchors),
        "cameras": len(state.cameras),
        "plan_revision": state.plan_revision,
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "local": bool(state.provider and state.provider.caps.local),
            "available": state.provider is not None,
            "remote_allowed": settings.llm.allow_remote_llm,
            "redaction_profile": settings.llm.redaction_profile,
        },
        "personality": {
            "default": state.effective_default_personality(),
            "configured_default": settings.personality.default_personality,
            "humor_ceiling": settings.personality.humor_ceiling,
            "roast_consent": settings.personality.roast_consent,
            "gamble": settings.personality.gamble,
        },
        "ux": settings.ux.model_dump(mode="json"),
        "bus": state.bus.stats(),
        "warnings": settings.warnings(),
    }


@router.get("/system/health")
async def system_health(request: Request, session: Session) -> dict[str, Any]:
    """Liveness plus the things that silently break a camera-driven system.

    A stale camera is reported as a *problem*, not omitted: a dead camera producing no tasks looks
    exactly like a tidy house, and that confusion is the worst failure mode this system has.
    """
    state = state_of(request)
    now = datetime.now(tz=UTC)
    rows = (await session.execute(select(CameraRow))).scalars().all()

    cameras = []
    problems: list[str] = []
    for row in rows:
        stale = not row.last_frame_at or (now - row.last_frame_at) > timedelta(minutes=5)
        cameras.append(
            {
                "id": row.id,
                "enabled": row.enabled,
                "last_frame_at": row.last_frame_at.isoformat() if row.last_frame_at else None,
                "stale": stale,
                "error": row.last_error,
                "node_id": row.node_id,
            }
        )
        if row.enabled and stale:
            problems.append(
                f"camera {row.id} has sent no frames recently - skills watching it are not running"
            )

    if state.provider is None:
        # The AI layer is core, not a bolt-on: a deployment without a provider is degraded, and
        # health should say so rather than letting every surface silently fall back to templates.
        problems.append(
            "no LLM provider available - the assistant is running without its brain and every "
            "surface degrades to templates. Run `openhup setup` and choose a provider "
            "(local Ollama by default, or a trusted cloud provider with the egress gate)."
        )

    for skill_id, errors in state.compile_failures.items():
        problems.append(f"skill {skill_id} will not run: {errors[0] if errors else 'unknown'}")

    needs_baseline = [
        anchor.id
        for anchor in state.anchors.values()
        if not anchor.baseline_ref and state.skills_watching(anchor.id)
    ]
    if needs_baseline:
        problems.append(
            f"no clean baseline captured for: {', '.join(sorted(needs_baseline))} - "
            f"clutter scores on these anchors are less accurate until you capture one"
        )

    return {
        "status": "degraded" if problems else "ok",
        "cameras": cameras,
        "bus_connected": state.bus.connected,
        "llm_available": state.provider is not None,
        "problems": problems,
        "loaded_at": state.loaded_at.isoformat() if state.loaded_at else None,
    }


@router.get("/system/llm-usage")
async def llm_usage(request: Request, session: Session) -> dict[str, Any]:
    """Audit trail for every model call.

    Exists so the privacy claim is inspectable rather than aspirational: what was sent, where, how
    big, and whether an image went with it.
    """
    state = state_of(request)
    rows = (
        (await session.execute(select(LLMCallRow).order_by(LLMCallRow.called_at.desc()).limit(200)))
        .scalars()
        .all()
    )
    return {
        "in_memory": {
            "calls": len(state.usage.entries),
            "remote_calls": state.usage.remote_calls,
            "remote_bytes_sent": state.usage.remote_bytes_sent,
            "recent": state.usage.entries[-25:],
        },
        "persisted": [
            {
                "at": row.called_at.isoformat(),
                "provider": row.provider,
                "model": row.model,
                "purpose": row.purpose,
                "local": row.local,
                "prompt_bytes": row.prompt_bytes,
                "included_image": row.included_image,
                "ok": row.ok,
            }
            for row in rows
        ],
    }


async def _count(session: AsyncSession, model: Any, *where: Any) -> int:
    result = await session.execute(select(func.count()).select_from(model).where(*where))
    return int(result.scalar() or 0)


__all__ = ["router"]
