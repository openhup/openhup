"""Skills: CRUD, natural-language parsing, simulation, and the compiled explanation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from openhup_schemas import (
    BUILTIN_DETECTORS,
    Observation,
    Skill,
    SkillOrigin,
    load_skills_yaml,
)
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db import ObservationRow, SkillRow, SkillStateRow, get_session
from ...skills.parse import ParseResult, describe, parse_skill
from ...skills.simulate import simulate, suggest_thresholds
from ..state import AppState

router = APIRouter(tags=["skills"])
UTC = UTC


def state_of(request: Request) -> AppState:
    return request.app.state.openhup


Session = Annotated[AsyncSession, Depends(get_session)]


class SkillSummary(BaseModel):
    id: str
    enabled: bool
    description: str
    effect_type: str
    urgency: str
    anchors: list[str]
    #: Plain-language rendering of what this skill does. Some users will only ever read this.
    explanation: str
    warnings: list[str] = Field(default_factory=list)
    #: Present when the skill is stored but cannot run.
    errors: list[str] = Field(default_factory=list)
    origin: str = "user"
    source_text: str | None = None


class ParseRequest(BaseModel):
    text: str = Field(min_length=3, max_length=500)


class ParseResponse(BaseModel):
    ok: bool
    skill: dict[str, Any] | None = None
    explanation: str = ""
    confidence: float = 0.0
    unsupported: str | None = None
    problems: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    #: True when the keyword fallback produced this rather than a model. Shown in the UI, because a
    #: guess presented as understanding is worse than an admitted guess.
    heuristic: bool = False
    fallback_reason: str | None = None
    #: Always true. A draft is never armed without the user seeing it.
    needs_confirmation: bool = True


def _summary(state: AppState, skill: Skill) -> SkillSummary:
    compiled = state.compiled.get(skill.id)
    return SkillSummary(
        id=skill.id,
        enabled=skill.enabled,
        description=skill.description,
        effect_type=str(skill.effect.type),
        urgency=skill.urgency.value,
        anchors=list(compiled.anchor_ids) if compiled else [w.anchor or "" for w in skill.watch],
        explanation=describe(skill),
        warnings=[w.message for w in (compiled.warnings if compiled else ())],
        errors=state.compile_failures.get(skill.id, []),
        origin=skill.origin.value,
        source_text=skill.source_text,
    )


@router.get("/skills", response_model=list[SkillSummary])
async def list_skills(request: Request, enabled: bool | None = None) -> list[SkillSummary]:
    state = state_of(request)
    skills = state.skills.values()
    if enabled is not None:
        skills = [s for s in skills if s.enabled == enabled]
    return [_summary(state, skill) for skill in sorted(skills, key=lambda s: s.id)]


@router.get("/skills/{skill_id}")
async def get_skill(skill_id: str, request: Request) -> dict[str, Any]:
    state = state_of(request)
    skill = state.skills.get(skill_id)
    if skill is None:
        raise HTTPException(404, f"no skill {skill_id!r}")
    compiled = state.compiled.get(skill_id)
    return {
        "skill": skill.to_yaml_dict(),
        "summary": _summary(state, skill).model_dump(),
        "compiled": {
            "anchors": list(compiled.anchor_ids) if compiled else [],
            "signal_keys": [str(k) for k in compiled.all_signal_keys()] if compiled else [],
            "horizon_s": compiled.horizon.total_seconds() if compiled else 0,
        },
    }


@router.post("/skills", status_code=201)
async def create_skill(
    request: Request,
    session: Session,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """Create a skill. Compiled before it is stored, so a bad skill never reaches the database."""
    state = state_of(request)
    skill = Skill.model_validate(payload)
    if skill.id in state.skills:
        raise HTTPException(409, f"skill {skill.id!r} already exists")

    compiled = state.compile_one(skill)  # raises SkillCompileError -> 422 with every finding
    session.add(
        SkillRow(
            id=skill.id,
            version=skill.version,
            enabled=skill.enabled,
            description=skill.description,
            definition=skill.to_yaml_dict(),
            origin=skill.origin.value,
            source_text=skill.source_text,
            warnings=[w.message for w in compiled.warnings],
            tags=skill.tags,
        )
    )
    await session.flush()

    state.skills[skill.id] = skill
    state.recompile()
    return {"created": skill.id, "warnings": [w.message for w in compiled.warnings]}


@router.patch("/skills/{skill_id}")
async def update_skill(
    skill_id: str,
    request: Request,
    session: Session,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    state = state_of(request)
    row = await session.get(SkillRow, skill_id)
    if row is None:
        raise HTTPException(404, f"no skill {skill_id!r}")

    merged = {**row.definition, **payload, "id": skill_id}
    skill = Skill.model_validate(merged)
    compiled = state.compile_one(skill)

    row.definition = skill.to_yaml_dict()
    row.enabled = skill.enabled
    row.description = skill.description
    row.version = skill.version
    row.warnings = [w.message for w in compiled.warnings]
    await session.flush()

    state.skills[skill_id] = skill
    state.recompile()
    return {"updated": skill_id, "warnings": row.warnings}


@router.delete("/skills/{skill_id}", status_code=204)
async def delete_skill(skill_id: str, request: Request, session: Session) -> None:
    state = state_of(request)
    row = await session.get(SkillRow, skill_id)
    if row is None:
        raise HTTPException(404, f"no skill {skill_id!r}")
    await session.delete(row)
    # Instance state goes with it; keeping orphaned FSM rows would resurrect the skill's phase if it
    # were later recreated with the same id.
    for instance in (
        await session.execute(select(SkillStateRow).where(SkillStateRow.skill_id == skill_id))
    ).scalars():
        await session.delete(instance)
    state.skills.pop(skill_id, None)
    state.recompile()


@router.post("/skills/parse", response_model=ParseResponse)
async def parse(request: Request, payload: ParseRequest) -> ParseResponse:
    """Turn a sentence into a draft skill.

    The result is never saved and never enabled. The UI shows the compiled meaning and offers a
    simulation before anything starts watching (ADR-008).
    """
    state = state_of(request)
    result: ParseResult = await parse_skill(
        payload.text,
        provider=state.provider,
        anchors=state.anchors,
        registry=BUILTIN_DETECTORS,
        timeout_s=state.settings.llm.timeout.total_seconds(),
    )
    return ParseResponse(
        ok=result.ok,
        skill=result.skill.to_yaml_dict() if result.skill else None,
        explanation=result.explanation or result.summary(),
        confidence=result.confidence,
        unsupported=result.unsupported,
        problems=result.problems,
        warnings=[w.message for w in result.warnings],
        heuristic=result.heuristic,
        fallback_reason=result.fallback_reason,
    )


@router.post("/skills/{skill_id}/simulate")
async def simulate_skill(
    skill_id: str,
    request: Request,
    session: Session,
    days: int = Query(default=7, ge=1, le=60),
    anchor: str | None = None,
) -> dict[str, Any]:
    """Replay stored observations against a skill.

    The most useful endpoint in the API for anyone tuning a threshold: it answers "would this have
    annoyed me last week?" before the skill is ever armed.
    """
    state = state_of(request)
    compiled = state.compiled.get(skill_id)
    if compiled is None:
        raise HTTPException(
            404, f"skill {skill_id!r} is not compiled: {state.compile_failures.get(skill_id)}"
        )

    target = anchor or (compiled.anchor_ids[0] if compiled.anchor_ids else None)
    if target is None:
        raise HTTPException(400, "skill watches no anchors")

    since = datetime.now(tz=UTC) - timedelta(days=days)
    rows = (
        (
            await session.execute(
                select(ObservationRow)
                .where(ObservationRow.anchor_id == target, ObservationRow.ts >= since)
                .order_by(ObservationRow.ts)
                .limit(200_000)
            )
        )
        .scalars()
        .all()
    )

    observations = [_to_observation(row) for row in rows]
    result = simulate(compiled, observations, anchor_id=target)
    return {
        "skill_id": skill_id,
        "anchor_id": target,
        "days": days,
        "observations": result.observations_seen,
        "verdict": result.verdict_line(),
        "advice": suggest_thresholds(result, compiled),
        "tasks_created": result.tasks_created,
        "alerts_raised": result.alerts_raised,
        "auto_resolved": result.tasks_auto_resolved,
        "suppressed": result.suppressions,
        "episodes": result.episodes,
        "per_day": result.per_day(),
        "mean_episode_s": result.mean_episode_duration.total_seconds(),
        "timeline": [
            {
                "ts": step.ts.isoformat(),
                "phase": step.phase.value,
                "actions": [type(a).__name__ for a in step.actions],
                "reasons": list(step.reasons),
            }
            for step in result.steps[:200]
        ],
    }


@router.post("/skills/import")
async def import_skills(
    request: Request,
    session: Session,
    body: str = Body(..., media_type="text/plain"),
    enable: bool = Query(default=False),
) -> dict[str, Any]:
    """Import a multi-document YAML file of skills.

    Imported skills are disabled unless `enable=true` is passed explicitly, because someone else's
    thresholds are almost never right for your house.
    """
    state = state_of(request)
    created, failed = [], {}
    for skill in load_skills_yaml(body):
        try:
            compiled = state.compile_one(skill)
        except Exception as exc:
            failed[skill.id] = str(exc)
            continue
        if skill.id in state.skills:
            failed[skill.id] = "already exists"
            continue
        stored = skill.model_copy(
            update={"enabled": enable and skill.enabled, "origin": SkillOrigin.IMPORT}
        )
        session.add(
            SkillRow(
                id=stored.id,
                version=stored.version,
                enabled=stored.enabled,
                description=stored.description,
                definition=stored.to_yaml_dict(),
                origin=stored.origin.value,
                warnings=[w.message for w in compiled.warnings],
                tags=stored.tags,
            )
        )
        state.skills[stored.id] = stored
        created.append(stored.id)

    await session.flush()
    state.recompile()
    return {"imported": created, "failed": failed, "enabled": enable}


@router.get("/skills/{skill_id}/state")
async def skill_state(skill_id: str, request: Request, session: Session) -> dict[str, Any]:
    """FSM phase per anchor, plus when the skill could next fire."""
    state = state_of(request)
    if skill_id not in state.skills:
        raise HTTPException(404, f"no skill {skill_id!r}")
    rows = (
        (await session.execute(select(SkillStateRow).where(SkillStateRow.skill_id == skill_id)))
        .scalars()
        .all()
    )
    skill = state.skills[skill_id]
    return {
        "skill_id": skill_id,
        "enabled": skill.enabled,
        "instances": [
            {
                "anchor_id": row.anchor_id,
                "phase": row.phase,
                "since": row.since.isoformat() if row.since else None,
                "episode_id": row.episode_id,
                "triggers_today": row.triggers_today,
                "suppressed_reason": row.suppressed_reason,
                "next_eligible_at": (
                    (row.last_resolved_at + skill.limits.cooldown).isoformat()
                    if row.last_resolved_at
                    else None
                ),
            }
            for row in rows
        ],
    }


def _to_observation(row: ObservationRow) -> Observation:
    return Observation.model_validate(
        {
            "id": row.id,
            "ts": row.ts,
            "source": {"camera_id": row.camera_id, "anchor_id": row.anchor_id, "replay": True},
            "detector": {"name": row.detector, "version": row.detector_version},
            "signals": row.signals,
        }
    )


__all__ = ["router"]
