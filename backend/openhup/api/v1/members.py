"""Household members: the consent-gated identity store (ADR-016).

A member is a person who answered "yes" to the consent question - nothing here can invent one.
The embedding is the entire biometric surface of the system: it is only ever written at consent
time, it lives in local Postgres, it is listable and deletable in Settings like facts, patterns,
and wins, and deleting a member deletes their embedding. There is no unknown-face store anywhere.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db import ConsentAskRow, MemberRow, PresenceWindowRow, get_session
from ...identity import consent_accept, consent_decline
from ..state import AppState

router = APIRouter(tags=["members"])
UTC = UTC

Session = Annotated[AsyncSession, Depends(get_session)]


def state_of(request: Request) -> AppState:
    return request.app.state.openhup


def _member_out(row: MemberRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "active": row.active,
        "enrolled_at": row.enrolled_at.isoformat(),
        "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
        "embedding_dim": len(row.embedding) if row.embedding else 0,
    }


class EnrollIn(BaseModel):
    """Enroll a member from a consent answer or the settings screen.

    `embedding` is only ever provided by the consent flow: the person said yes, so the embedding
    that was computed for them (and held nowhere else) becomes the store. The settings screen
    enrolls from a capture the vision service took at the moment of consent.
    """

    name: str = Field(min_length=1, max_length=64)
    embedding: list[float] = Field(min_length=32, max_length=2048)


class ConsentAnswer(BaseModel):
    anchor_id: str
    answer: str  # yes | no
    name: str | None = Field(default=None, max_length=64)


@router.get("/members")
async def list_members(request: Request, session: Session) -> dict[str, Any]:
    """Every enrolled member, with their consent state. Reviewable and deletable, like facts."""
    rows = (
        (await session.execute(select(MemberRow).order_by(MemberRow.enrolled_at))).scalars().all()
    )
    return {
        "members": [_member_out(row) for row in rows],
        "enabled": state_of(request).settings.identity.enabled,
    }


@router.post("/members", status_code=201)
async def enroll(request: Request, session: Session, payload: EnrollIn) -> dict[str, Any]:
    """Enroll a member. The only path that stores an embedding - and it is a consent path.

    The person said yes; the embedding computed for them at that moment (and held nowhere else
    before this) becomes the store. The name is what they gave, not what the system guessed.
    """
    if not state_of(request).settings.identity.enabled:
        raise HTTPException(409, "identity is disabled - enable it in config before enrolling")
    name = payload.name.strip()
    if not name:
        raise HTTPException(422, "a name is required")
    existing = (
        await session.execute(select(MemberRow).where(func.lower(MemberRow.name) == name.lower()))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(409, f"{payload.name} is already enrolled")
    row = MemberRow(
        name=name,
        embedding=payload.embedding,
        active=True,
        enrolled_at=datetime.now(tz=UTC),
    )
    session.add(row)
    await session.flush()
    state = state_of(request)
    state.members_revision += 1
    state.plan_revision = state._revision()
    return {"member": _member_out(row), "reply": consent_accept(name)}


@router.delete("/members/{member_id}", status_code=204)
async def delete_member(request: Request, member_id: str, session: Session) -> None:
    """Forget a member: delete their row, their embedding, and their presence history.

    Same semantics as forgetting a fact - deletion is immediate and complete. The vision service
    drops them from the gallery on the next plan refresh because `members_revision` bumped.
    """
    row = await session.get(MemberRow, member_id)
    if row is None:
        raise HTTPException(404, "no such member")
    await session.delete(row)
    open_windows = (
        (
            await session.execute(
                select(PresenceWindowRow).where(
                    PresenceWindowRow.member_id == member_id,
                    PresenceWindowRow.ended_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    for window in open_windows:
        window.ended_at = datetime.now(tz=UTC)
    state = state_of(request)
    state.members_revision += 1
    state.plan_revision = state._revision()


@router.post("/members/consent", status_code=200)
async def answer_consent(
    request: Request, session: Session, payload: ConsentAnswer
) -> dict[str, Any]:
    """Record the answer to a consent ask (spoken via voice, or tapped in the UI).

    The marker exists to stop the re-asking, not to remember the person: a "no" answer updates
    the day's marker and nothing else. A "yes" answer without an embedding (consent granted at
    the mic but no capture yet) is recorded as asked-yes so the flow can hand off to enrollment.
    """
    today = datetime.now(tz=UTC)
    marker = (
        await session.execute(
            select(ConsentAskRow).where(
                ConsentAskRow.anchor_id == payload.anchor_id,
                ConsentAskRow.asked_on == today.date(),
            )
        )
    ).scalar_one_or_none()
    if marker is None:
        marker = ConsentAskRow(anchor_id=payload.anchor_id, asked_on=today.date(), answer="no")
        session.add(marker)
    marker.answer = payload.answer
    marker.answered_at = today

    if payload.answer == "no":
        return {"reply": consent_decline()}
    if payload.answer == "yes":
        if payload.name:
            return {"reply": "Great. Say the name you want me to use, or add it in Settings."}
        return {"reply": "Great. What should I call you?"}
    raise HTTPException(422, "answer must be 'yes' or 'no'")


__all__ = ["router"]
