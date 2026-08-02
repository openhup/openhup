"""Household memory: things the assistant has been told.

Facts live in local Postgres (`memory_facts`) and never leave the house by themselves. The only way
one reaches a model is as a fragment inside a phrasing prompt, which is already gated by
`llm.allow_remote_llm`, subject to the redaction profile, and recorded in the usage audit - the
"you can see exactly what left the house" promise covers memory like everything else.

Retrieval is deliberately cheap keyword matching rather than embeddings. A household stores hundreds
of facts at most, not millions, so an embedding pipeline would add a model dependency to a problem
that does not need one (ADR-008: nothing here may depend on the LLM). The query is the skill's title
hint plus the anchor label, so "clear the kitchen counter" surfaces "I call the kitchen counter the
junk counter" without ever calling a model.

A memory the user cannot inspect and delete is not a memory, it is a surveillance system. That is
why the API exposes every fact for listing and per-row deletion, and why facts are never edited
silently - a fact that is no longer trusted is deleted and taught again.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select

from ..db import EpisodeRow, MemoryFactRow, MemoryPatternRow
from .patterns import LOOKBACK, NUDGE_WINDOW, PatternCandidate, discover_patterns

UTC = UTC

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession

#: Keep the prompt-context injection small: beyond a handful of facts the personality layer starts
#: talking about the facts instead of the task.
MAX_FACTS_PER_CONTEXT = 3
#: Terms shorter than this are noise ("is", "the") and match half the store.
_MIN_TERM = 3

#: Function words that would match half the store. Kept small and obvious; the cost of a missed
#: stopword is one irrelevant fact in a prompt, which is harmless.
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "you",
        "your",
        "that",
        "this",
        "what",
        "about",
        "have",
        "has",
        "are",
        "was",
        "were",
        "when",
        "where",
        "will",
        "would",
        "can",
        "could",
        "should",
        "from",
        "them",
        "they",
        "there",
        "then",
        "than",
        "its",
        "it's",
    }
)


async def relevant_facts(
    session: AsyncSession,
    *,
    query: str,
    limit: int = MAX_FACTS_PER_CONTEXT,
) -> list[str]:
    """Keyword-match facts against a query, most relevant first.

    Matching is on word sets, not substrings - "bin day trash" finds "bin day is Tuesday" and does
    not find "water the plants on Sundays" - scored by how many query terms hit, with recency as
    the tie-break. Empty or stop-word-only queries return nothing: an empty query must not dump the
    whole store into a prompt.
    """
    terms = _terms(query)
    if not terms:
        return []

    # The store is a few hundred rows at most, so the whole filter can run in Python: word-set
    # matching rather than SQL LIKE, which keeps "day" from matching "sundays" and makes
    # relevance trivially testable.
    rows = (await session.execute(select(MemoryFactRow))).scalars().all()
    scored = sorted(
        (row for row in rows if _matches(row, terms)),
        key=lambda row: (_score(row, terms), row.created_at or _MIN_DT),
        reverse=True,
    )
    return [row.fact for row in scored[:limit]]


async def list_facts(session: AsyncSession, *, limit: int = 200) -> list[MemoryFactRow]:
    """Everything the assistant knows, newest first, for the review screen."""
    rows = (
        (await session.execute(select(MemoryFactRow).order_by(MemoryFactRow.created_at.desc())))
        .scalars()
        .all()
    )
    return list(rows[:limit])


# --------------------------------------------------------------------------------------
# Learned patterns
# --------------------------------------------------------------------------------------


async def refresh_patterns(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    labels: dict[str, str] | None = None,
) -> list[MemoryPatternRow]:
    """Recompute learned patterns from episode history and return the active ones.

    Idempotent upsert keyed by (kind, skill_id, anchor_id): re-running corrects rather than
    duplicates. Dismissed patterns are left exactly as the user left them - dismissal is a
    statement that the pattern is not useful, not a request to see it again with fresh evidence.
    Active patterns whose evidence has aged out are deleted, ready to be learned again from
    scratch if the behaviour returns.
    """
    now = now or datetime.now(tz=UTC)
    since = now - LOOKBACK
    episodes = (
        (
            await session.execute(
                select(EpisodeRow)
                .where(EpisodeRow.opened_at >= since)
                .order_by(EpisodeRow.opened_at)
            )
        )
        .scalars()
        .all()
    )
    candidates = discover_patterns(episodes, now=now, labels=labels)
    by_key = {_pattern_key(c): c for c in candidates}

    rows = (await session.execute(select(MemoryPatternRow))).scalars().all()
    existing = {(row.kind, row.skill_id, row.anchor_id): row for row in rows}

    for key, candidate in by_key.items():
        row = existing.get(key)
        if row is None:
            session.add(
                MemoryPatternRow(
                    kind=candidate.kind,
                    skill_id=candidate.skill_id,
                    anchor_id=candidate.anchor_id,
                    summary=candidate.summary,
                    evidence=candidate.evidence,
                    confidence=candidate.confidence,
                    status="active",
                )
            )
        elif row.status != "dismissed":
            row.summary = candidate.summary
            row.evidence = candidate.evidence
            row.confidence = candidate.confidence

    # Active rows whose claim no longer holds are gone until the behaviour returns.
    for key, row in existing.items():
        if key not in by_key and row.status == "active":
            await session.delete(row)
    await session.flush()

    active = (
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
    return list(active)


def pattern_due(pattern: MemoryPatternRow, *, now: datetime) -> tuple[bool, str | None]:
    """Is a cadence pattern's predicted window open right now?

    Returns ``(due, basis)`` where `basis` is the episode id the prediction is anchored to - the
    nudge fires at most once per episode cycle, because once an episode is nudged its basis is
    recorded and the window cannot open again until a new episode arrives.
    """
    if pattern.kind != "cadence" or pattern.status != "active":
        return False, None
    evidence = pattern.evidence
    median_h = evidence.get("median_interval_h")
    last_at_raw = evidence.get("last_episode_at")
    basis = evidence.get("last_episode_id")
    if not median_h or not last_at_raw or not basis:
        return False, None
    last_at = datetime.fromisoformat(str(last_at_raw))
    if last_at.tzinfo is None:
        last_at = last_at.replace(tzinfo=UTC)
    if pattern.last_nudge_basis == str(basis):
        return False, None

    window_h = float(median_h)
    start = last_at + timedelta(hours=window_h * NUDGE_WINDOW[0])
    end = last_at + timedelta(hours=window_h * NUDGE_WINDOW[1])
    if start <= now < end:
        return True, str(basis)
    return False, None


def nudge_text(pattern: MemoryPatternRow) -> str:
    """The spoken form of a due pattern. Forward, short, never a backlog count."""
    return f"{pattern.summary} It's about that time."


def _pattern_key(candidate: PatternCandidate) -> tuple[str, str, str]:
    return (candidate.kind, candidate.skill_id, candidate.anchor_id)


def _terms(text: str) -> list[str]:
    return [
        term
        for term in re.split(r"\s+", text.lower().strip())
        if len(term) >= _MIN_TERM and term not in _STOPWORDS
    ]


def _words(row: MemoryFactRow) -> set[str]:
    return set(re.split(r"\s+", f"{row.fact} {row.topic or ''}".lower().strip()))


def _matches(row: MemoryFactRow, terms: list[str]) -> bool:
    return bool(_words(row) & set(terms))


def _score(row: MemoryFactRow, terms: list[str]) -> int:
    return len(_words(row) & set(terms))


#: Stable floor for the recency tie-break when a created_at is somehow missing.
_MIN_DT = datetime.min


__all__ = [
    "MAX_FACTS_PER_CONTEXT",
    "list_facts",
    "nudge_text",
    "pattern_due",
    "refresh_patterns",
    "relevant_facts",
]
