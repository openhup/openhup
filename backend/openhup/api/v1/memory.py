"""Household memory: what the assistant has been told, and what it has learned.

Facts (taught, `memory_facts`) and patterns (learned, `memory_patterns`) both live in local
Postgres and never leave the house by themselves - the only way either reaches a model is as a
fragment inside a phrasing prompt, which is already gated by `llm.allow_remote_llm`, subject to the
redaction profile, and recorded in the usage audit.

This router is the review screen's backend: everything the assistant "knows" - taught *and*
learned - can be listed, and everything can be dismissed or deleted. A memory the user cannot
inspect and delete is not a memory, it is a surveillance system (ADR-012, ADR-013).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db import MemoryFactRow, MemoryPatternRow, get_session
from ...memory import list_facts, refresh_patterns
from ...memory.patterns import NUDGE_MIN_CONFIDENCE
from ..state import AppState

router = APIRouter(tags=["memory"])

Session = Annotated[AsyncSession, Depends(get_session)]
UTC = UTC


def state_of(request: Request) -> AppState:
    return request.app.state.openhup


class FactIn(BaseModel):
    fact: str = Field(min_length=1, max_length=500)
    topic: str | None = Field(default=None, max_length=64)


# --------------------------------------------------------------------------------------
# Taught facts
# --------------------------------------------------------------------------------------


@router.get("/memory")
async def list_memory(session: Session) -> list[dict[str, Any]]:
    """Everything the assistant has been told, newest first. The review screen calls only this."""
    rows = await list_facts(session)
    return [_fact_out(row) for row in rows]


@router.post("/memory", status_code=201)
async def add_memory(session: Session, payload: FactIn) -> dict[str, str]:
    fact = payload.fact.strip()
    if not fact:
        raise HTTPException(422, "a fact cannot be blank")
    row = MemoryFactRow(
        fact=fact,
        topic=payload.topic.strip() if payload.topic and payload.topic.strip() else None,
        source="settings",
    )
    session.add(row)
    await session.flush()
    return {"created": row.id}


@router.delete("/memory/{fact_id}", status_code=204)
async def delete_memory(fact_id: str, session: Session) -> None:
    row = await session.get(MemoryFactRow, fact_id)
    if row is None:
        raise HTTPException(404, "no such memory fact")
    await session.delete(row)


# --------------------------------------------------------------------------------------
# Learned patterns
# --------------------------------------------------------------------------------------


@router.get("/memory/patterns")
async def list_patterns(request: Request, session: Session) -> dict[str, Any]:
    """Learned patterns, freshly recomputed from episode history, with their evidence.

    Everything a claim says is backed by the numbers in `evidence`, so the review screen can show
    how confident the claim is and why. Dismissed patterns are not returned here.
    """
    state = state_of(request)
    labels = {anchor_id: anchor.label for anchor_id, anchor in state.anchors.items()}
    rows = await refresh_patterns(session, now=datetime.now(tz=UTC), labels=labels)
    return {
        "patterns": [_pattern_out(row) for row in rows],
        "note": (
            "Derived on this device from the episodes the skill engine already records. "
            "Forward-looking only, dismissable, never per-person."
        ),
    }


@router.delete("/memory/patterns/{pattern_id}", status_code=204)
async def dismiss_pattern(pattern_id: str, session: Session) -> None:
    """Dismiss a pattern: it is kept (so it is not learned again) but never surfaced or nudged."""
    row = await session.get(MemoryPatternRow, pattern_id)
    if row is None:
        raise HTTPException(404, "no such memory pattern")
    row.status = "dismissed"
    await session.flush()


@router.get("/memory/patterns/active")
async def active_patterns(session: Session) -> list[dict[str, Any]]:
    """Active patterns without recomputing. Cheaper than GET /memory/patterns for polls."""
    rows = (
        (
            await session.execute(
                select(MemoryPatternRow)
                .where(MemoryPatternRow.status == "active")
                .order_by(MemoryPatternRow.confidence.desc())
            )
        )
        .scalars()
        .all()
    )
    return [_pattern_out(row) for row in rows]


def _fact_out(row: MemoryFactRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "fact": row.fact,
        "topic": row.topic,
        "source": row.source,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _pattern_out(row: MemoryPatternRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "kind": row.kind,
        "skill_id": row.skill_id,
        "anchor_id": row.anchor_id,
        "summary": row.summary,
        "confidence": row.confidence,
        "evidence": row.evidence,
        "nudge_eligible": row.confidence >= NUDGE_MIN_CONFIDENCE,
        "last_nudge_at": row.last_nudge_at.isoformat() if row.last_nudge_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


__all__ = ["router"]
