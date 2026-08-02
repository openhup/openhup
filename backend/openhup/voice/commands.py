"""Deterministic voice command routing.

A transcript in, an action out. This is deliberately *not* an LLM router: every intent is matched
by keyword, so a spoken command behaves identically with or without a model, and a bad match is a
wrong guess rather than a confident hallucination (the same position the skill parser takes in
`skills/parse.py`).

The server owns the intents that need state - task commands act on the next task, queries read it,
dictation reuses the existing natural-language skill parser. Navigation is the one thing the client
handles itself, because it needs no state; `route_command` still recognises it so the reply can be
spoken consistently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from openhup_schemas import BUILTIN_DETECTORS, TaskState
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import MemoryFactRow
from ..memory import list_facts, refresh_patterns, relevant_facts
from ..notify import Dispatcher
from ..skills.parse import describe, parse_skill
from ..tasks import Executor

if TYPE_CHECKING:  # pragma: no cover - avoids an import cycle with the API package.
    from ..api.state import AppState

UTC = UTC

#: Longest a snooze may be, matching TaskUpdate's own ceiling.
_MAX_SNOOZE_MINUTES = 60 * 24 * 14


@dataclass
class CommandResult:
    """What the assistant should do and say in response to a transcript."""

    intent: str  # task_command | query | skill_dictation | navigate | memory | identity | unknown
    #: Text to speak back. Always present; the frontend feeds it straight to TTS.
    reply: str
    #: Task mutation performed, when `intent == "task_command"`.
    action: str | None = None
    task_id: str | None = None
    #: Draft skill (never enabled), when `intent == "skill_dictation"`.
    skill: dict[str, Any] | None = None
    explanation: str = ""
    confidence: float = 0.0
    unsupported: str | None = None
    problems: list[str] = field(default_factory=list)
    needs_confirmation: bool = True
    #: Route to navigate to, when `intent == "navigate"`.
    target: str | None = None
    #: Member id the client should remember as this device's speaker (self-ID).
    speaker_id: str | None = None


async def route_command(
    text: str,
    *,
    state: AppState,
    session: AsyncSession,
    speaker: str | None = None,
) -> CommandResult:
    """Turn a spoken sentence into an action and a spoken reply.

    `speaker` is the member id the client already knows (per-device "who am I"), sent with every
    command. It is a declaration, never a recognition: the person told the device who they are,
    and the device passes it along. Without one, personal queries in a multi-member household get
    the honest "I don't know who's asking" rather than a guess.
    """
    lowered = _normalise(text)
    if not lowered:
        return CommandResult(intent="unknown", reply="I didn't catch that.")

    target = navigation_target(lowered)
    if target is not None:
        label = _NAV_LABELS.get(target, target)
        return CommandResult(intent="navigate", target=target, reply=f"Opening {label}.")

    identity = await _identity_command(lowered, state, session, speaker=speaker)
    if identity is not None:
        return identity

    memory = await _memory_command(lowered, state, session)
    if memory is not None:
        return memory

    task = await _task_command(lowered, state, session)
    if task is not None:
        return task

    query = await _query(lowered, state, session, speaker=speaker)
    if query is not None:
        return query

    if _looks_like_skill(lowered):
        return await _dictate(text, state)

    return CommandResult(
        intent="unknown",
        reply=(
            "I didn't catch that. You can say \u201cwhat should I do\u201d, \u201cdone\u201d, "
            "\u201cshow tasks\u201d, or \u201cremind me when the trash is full\u201d."
        ),
    )


# --------------------------------------------------------------------------------------
# Pure detection helpers (unit-tested in isolation)
# --------------------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop trailing punctuation that speech often adds."""
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    return lowered.rstrip(".,!?")


_NAV_TARGETS: dict[str, str] = {
    "tasks": "/tasks",
    "my tasks": "/tasks",
    "task list": "/tasks",
    "skills": "/skills",
    "cameras": "/cameras",
    "camera": "/cameras",
    "habits": "/metrics",
    "metrics": "/metrics",
    "settings": "/settings",
    "alerts": "/",
    "home": "/",
    "today": "/",
}
_NAV_LABELS = {
    "/tasks": "tasks",
    "/skills": "skills",
    "/cameras": "cameras",
    "/metrics": "habits",
    "/settings": "settings",
    "/": "today",
}
_NAV_PREFIXES = ("show me ", "show ", "open ", "go to ", "take me to ", "navigate to ", "view ")


def navigation_target(lowered: str) -> str | None:
    """Map \"show tasks\", \"open cameras\", or a bare \"tasks\" to a route."""
    candidate = lowered
    for prefix in _NAV_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    return _NAV_TARGETS.get(candidate)


#: (action, trigger phrases). Order matters: longer/less ambiguous first.
_COMPLETE = (
    "done",
    "complete",
    "finish",
    "finished",
    "completed",
    "i did it",
    "i'm done",
    "im done",
    "it's done",
    "its done",
    "all done",
    "mark it done",
    "mark done",
)
_START = ("start", "begin", "starting", "i'll start", "ill start", "i'm starting", "im starting")
_SNOOZE = ("snooze", "later", "not now", "postpone", "remind me later", "snooze it")
_DISMISS = ("dismiss", "dismiss it", "ignore", "ignore it")
_FALSE_POSITIVE = (
    "not a real task",
    "false positive",
    "that's not right",
    "that's wrong",
    "this is wrong",
    "not real",
    "wrong",
)

#: Query phrases about what to do next.
_QUERIES = (
    "what should i do",
    "what do i do",
    "what do i need to do",
    "what needs doing",
    "what's next",
    "whats next",
    "what is next",
    "what's on my list",
    "whats on my list",
    "what's on the list",
    "whats on the list",
    "read my task",
    "read my tasks",
    "tell me what to do",
    "anything to do",
    "any tasks",
)

#: Keywords that mark a sentence as "define a rule", not a command.
_SKILL_HINTS = (
    "remind me",
    "tell me when",
    "let me know",
    "alert me",
    "notify me",
    "warn me",
    "watch for",
    "watch the",
    "keep an eye",
    "when the",
    "when my",
    "when i",
    "if the",
    "if my",
)


def command_action(lowered: str) -> tuple[str, int] | None:
    """Classify a short imperative as a task command. Returns ``(action, snooze_minutes)``."""
    if lowered in _COMPLETE or lowered.startswith("done ") or lowered.startswith("complete "):
        return ("complete", 0)
    if lowered in _START or lowered.startswith("start ") or lowered.startswith("begin "):
        return ("start", 0)
    if lowered in _SNOOZE or lowered.startswith("snooze ") or lowered.startswith("later "):
        return ("snooze", snooze_minutes(lowered))
    if lowered in _DISMISS or lowered.startswith("dismiss ") or lowered.startswith("ignore "):
        return ("dismiss", 0)
    if lowered in _FALSE_POSITIVE:
        return ("false_positive", 0)
    return None


def snooze_minutes(lowered: str, default: int = 60) -> int:
    """Pull a duration out of \"snooze for 30 minutes\" or \"later for 2 hours\"."""
    for number, unit in re.findall(r"(\d+)\s*(minute|minutes|min|hour|hours|hr|day|days)", lowered):
        minutes = int(number)
        if unit.startswith("hour"):
            minutes *= 60
        elif unit.startswith("day"):
            minutes *= 60 * 24
        return min(max(minutes, 1), _MAX_SNOOZE_MINUTES)
    return default


def is_query(lowered: str) -> bool:
    return lowered in _QUERIES


def _looks_like_skill(lowered: str) -> bool:
    return any(hint in lowered for hint in _SKILL_HINTS)


#: Prefixes that teach a fact. Order matters: longest/most specific first.
_REMEMBER_PREFIXES = (
    "i want you to remember that ",
    "remember that ",
    "remember: ",
    "note that ",
    "keep in mind that ",
    "from now on ",
    "remember ",
)

#: "what do you remember about <topic>" asks about one thing; the bare forms ask for the store.
_RECALL_ABOUT_PREFIXES = ("what do you remember about ", "what do you know about ")
_RECALL_ALL = (
    "what do you remember",
    "what do you know",
    "what have i told you",
    "what have you been told",
    "what are you supposed to remember",
    "what do you know so far",
)

_FORGET_PREFIXES = (
    "stop remembering that ",
    "never mind that ",
    "you can forget ",
    "forget that ",
    "forget ",
)

#: Phrases that wipe the store rather than one fact.
_FORGET_ALL = ("everything", "all", "all facts", "everything you know")


def _extract_fact(lowered: str) -> str | None:
    """The claim after a teach prefix, or None when this is not a teach command.

    "remember to ..." is a reminder, not a fact, so it deliberately falls through to the skill
    dictation path ("remind me when ...").
    """
    for prefix in _REMEMBER_PREFIXES:
        if lowered.startswith(prefix):
            fact = lowered[len(prefix) :].strip().strip(".,!?")
            if not fact or fact.startswith("to "):
                return None
            return fact
    return None


def _extract_recall(lowered: str) -> str | None:
    """The topic of a recall question, "" for the whole store, or None when not a recall."""
    for prefix in _RECALL_ABOUT_PREFIXES:
        if lowered.startswith(prefix):
            topic = lowered[len(prefix) :].strip().strip(".,!?")
            return topic or None
    if lowered in _RECALL_ALL:
        return ""
    return None


#: Phrases that ask what the assistant has *learned* (patterns), as opposed to what it was told.
_PATTERN_RECALL_ABOUT_PREFIXES = (
    "what have you noticed about ",
    "what patterns have you noticed about ",
    "what patterns have you seen about ",
)
_PATTERN_RECALL_ALL = (
    "what have you noticed",
    "what patterns have you noticed",
    "what patterns have you seen",
    "have you noticed anything",
    "what have you seen",
    "what do you see happening regularly",
    "anything interesting",
)


def _extract_pattern_recall(lowered: str) -> str | None:
    """The topic of a learned-pattern question, "" for everything, or None when not one."""
    for prefix in _PATTERN_RECALL_ABOUT_PREFIXES:
        if lowered.startswith(prefix):
            topic = lowered[len(prefix) :].strip().strip(".,!?")
            return topic or None
    if lowered in _PATTERN_RECALL_ALL:
        return ""
    return None


def _extract_forget(lowered: str) -> str | None:
    """The phrase to forget, or None when this is not a forget command."""
    for prefix in _FORGET_PREFIXES:
        if lowered.startswith(prefix):
            phrase = lowered[len(prefix) :].strip().strip(".,!?")
            return phrase or None
    return None


def _pattern_matches(row, topic: str) -> bool:
    """Word-set match of a recall topic against a pattern's summary and subject."""
    words = {w for w in re.split(r"\s+", topic.lower()) if len(w) >= 3}
    if not words:
        return True
    text = f"{row.summary} {row.anchor_id}".lower()
    return bool(set(re.split(r"\s+", text)) & words)


# --------------------------------------------------------------------------------------
# Identity: self-declaration and consent (ADR-016)
# --------------------------------------------------------------------------------------

#: "it's Sam" / "I'm Sam" / "call me Sam" - the person declares who they are. Never a guess.
_SELF_ID_PREFIXES = (
    "i'm ",
    "im ",
    "it's ",
    "its ",
    "this is ",
    "call me ",
    "i am ",
)
_SELF_ID_BARE = {"it's me", "its me", "i'm me", "im me"}
#: "what's my name" / "who am i" - the system reports what it was *told*, never what it sees.
_WHO_AM_I = ("who am i", "what's my name", "whats my name", "what is my name")
#: Consent answers. "yes, remember me" and "no thanks" both work; a bare "yes"/"no" only counts
#: when a consent ask is actually pending, which the client knows but the router must not assume.
_CONSENT_YES = ("yes", "sure", "okay", "remember me", "yes remember me")
_CONSENT_NO = ("no", "no thanks", "no thank you", "don't remember me", "dont remember me")


def _extract_self_id(lowered: str) -> str | None:
    """The name in a self-declaration, or None when this is not one."""
    for prefix in _SELF_ID_PREFIXES:
        if lowered.startswith(prefix):
            name = lowered[len(prefix) :].strip().strip(".,!?")
            return name or None
    if lowered in _SELF_ID_BARE:
        return ""
    return None


async def _identity_command(
    lowered: str,
    state: AppState,
    session: AsyncSession,
    *,
    speaker: str | None,
) -> CommandResult | None:
    """Self-declaration and consent answers. Deterministic, like everything else here.

    Identity is declared, never inferred: the router only ever matches the words the person
    actually said. A declaration is confirmed against the enrolled members and the client is told
    which id to remember for this device; there is no face matching anywhere in this path.
    """
    from sqlalchemy import select

    from ..db import MemberRow

    if lowered in _WHO_AM_I:
        if not speaker:
            return CommandResult(
                intent="identity",
                reply="I don't know who's asking. Say \u201cit's \u201d and your name.",
            )
        row = await session.get(MemberRow, speaker)
        if row is None:
            return CommandResult(intent="identity", reply="I don't have that name on record yet.")
        return CommandResult(intent="identity", reply=f"You told me you're {row.name}.")

    if lowered in _CONSENT_YES or lowered in _CONSENT_NO:
        # A bare "yes"/"no" without a pending ask is not a task command and not identity; leave
        # it for the consent handoff the client drives. The router stays stateless on purpose.
        if lowered in _CONSENT_YES:
            return CommandResult(
                intent="identity",
                reply="Good. What should I call you? Say \u201ccall me \u201d and a name.",
            )
        return CommandResult(intent="identity", reply="No problem. I won't ask again today.")

    name = _extract_self_id(lowered)
    if name is None:
        return None
    if not name:
        return CommandResult(
            intent="identity", reply="And what's your name? Say \u201ccall me \u201d and it."
        )

    rows = (await session.execute(select(MemberRow).where(MemberRow.active))).scalars().all()
    member = next((r for r in rows if r.name.lower() == name.lower()), None)
    if member is None:
        return CommandResult(
            intent="identity",
            reply=(
                f"I don't know anyone called {name} yet. "
                "If I've asked before and you said yes, check Settings; "
                "otherwise the next time I see you I'll ask."
            ),
        )
    return CommandResult(
        intent="identity",
        speaker_id=member.id,
        reply=f"Got it - I'll remember you as {member.name} on this device.",
    )


# --------------------------------------------------------------------------------------
# Stateful handlers
# --------------------------------------------------------------------------------------


async def _memory_command(
    lowered: str,
    state: AppState,
    session: AsyncSession,
) -> CommandResult | None:
    """Teach, recall, or forget household memory: facts (taught) and patterns (learned).

    Deterministic, like everything else here: no model decides what was said or what to reply. The
    store is local Postgres; nothing leaves the house except a fragment inside a phrasing prompt,
    which is gated and audited like any other LLM egress.
    """
    fact = _extract_fact(lowered)
    if fact is not None:
        session.add(MemoryFactRow(fact=fact, topic=None, source="voice"))
        await session.flush()
        return CommandResult(intent="memory", action="remember", reply=f"Got it: {fact}")

    topic = _extract_recall(lowered)
    if topic is not None:
        if topic:
            facts = await relevant_facts(session, query=topic)
            if not facts:
                return CommandResult(
                    intent="memory",
                    action="recall",
                    reply=f"I don't remember anything about {topic}.",
                )
            reply = "I remember: " + "; ".join(facts)
        else:
            rows = await list_facts(session, limit=5)
            if not rows:
                return CommandResult(
                    intent="memory",
                    action="recall",
                    reply=(
                        "I don't remember anything yet. Say \u201cremember that\u201d to teach me."
                    ),
                )
            reply = "I remember: " + "; ".join(row.fact for row in rows)
        return CommandResult(intent="memory", action="recall", reply=reply)

    phrase = _extract_forget(lowered)
    if phrase is not None:
        deleted = await _delete_facts(session, phrase)
        if deleted:
            return CommandResult(intent="memory", action="forget", reply="Forgotten.")
        return CommandResult(intent="memory", action="forget", reply="I don't remember that.")

    pattern_topic = _extract_pattern_recall(lowered)
    if pattern_topic is not None:
        labels = {anchor_id: anchor.label for anchor_id, anchor in state.anchors.items()}
        rows = await refresh_patterns(session, now=datetime.now(tz=UTC), labels=labels)
        rows = [row for row in rows if _pattern_matches(row, pattern_topic)]
        if not rows:
            if pattern_topic:
                return CommandResult(
                    intent="memory",
                    action="recall_patterns",
                    reply=f"I haven't noticed anything about {pattern_topic} yet.",
                )
            return CommandResult(
                intent="memory",
                action="recall_patterns",
                reply=(
                    "Nothing yet. Give me a couple of weeks of data "
                    "and I'll start noticing patterns."
                ),
            )
        reply = "I've noticed: " + "; ".join(row.summary for row in rows[:3])
        return CommandResult(intent="memory", action="recall_patterns", reply=reply)
    return None


async def _delete_facts(session: AsyncSession, phrase: str) -> int:
    """Delete facts matching a forget phrase. Returns how many went.

    Matching is both directions on the fact text ("forget bin day" finds "bin day is Tuesday", and
    "forget that bin day is Tuesday" finds it too), done in Python because the store is a few
    hundred rows at most and the semantics are easier to read than an OR of LIKEs. Deletion is
    per-row and immediate - a forgotten fact is gone, not hidden.
    """
    needle = phrase.lower()
    rows = (await session.execute(select(MemoryFactRow))).scalars().all()
    if needle in _FORGET_ALL:
        matching = rows
    else:
        matching = [row for row in rows if needle in row.fact.lower() or row.fact.lower() in needle]
    for row in matching:
        await session.delete(row)
    return len(matching)


def _executor(state: AppState, session: AsyncSession) -> Executor:
    return Executor(
        session=session,
        renderer=state.renderer,
        bus=state.bus,
        dispatcher=state.dispatcher or Dispatcher(channels={}),
    )


async def _task_command(
    lowered: str,
    state: AppState,
    session: AsyncSession,
) -> CommandResult | None:
    classified = command_action(lowered)
    if classified is None:
        return None
    action, minutes = classified

    row = await _executor(state, session).next_task()
    if row is None:
        return CommandResult(
            intent="task_command", action=action, reply="There's nothing to do right now."
        )

    now = datetime.now(tz=UTC)
    if action == "complete":
        skill = state.skills.get(row.skill_id)
        if skill is None:
            row.state = TaskState.RESOLVED_MANUAL.value
            row.completed_at = now
        else:
            await _executor(state, session).complete_manually(
                row.id, skill=skill, now=now, still_matching=None
            )
        reply = "Done. Nice work."
    elif action == "start":
        row.state = TaskState.IN_PROGRESS.value
        reply = "Started. I'll keep an eye on it."
    elif action == "snooze":
        row.state = TaskState.SNOOZED.value
        row.snoozed_until = now + timedelta(minutes=minutes)
        reply = f"Snoozed for {_say_minutes(minutes)}."
    elif action == "dismiss":
        row.state = TaskState.DISMISSED.value
        row.completed_at = now
        reply = "Dismissed."
    else:  # false_positive
        row.false_positive = True
        row.state = TaskState.DISMISSED.value
        row.completed_at = now
        row.note = "marked as a false positive by voice"
        reply = "Marked as not a real task. Thanks - that helps me tune the threshold."
    await session.flush()
    return CommandResult(intent="task_command", action=action, task_id=row.id, reply=reply)


async def _query(
    lowered: str,
    state: AppState,
    session: AsyncSession,
    *,
    speaker: str | None = None,
) -> CommandResult | None:
    if not is_query(lowered):
        return None
    member_count = await _member_count(session)
    if member_count > 1 and not speaker:
        # The honest reply: the next task belongs to the household, not to whoever happened to
        # ask. Guessing would hand someone the wrong person's chore.
        return CommandResult(
            intent="query",
            reply=(
                "I don't know who's asking. Say \u201cit's \u201d and your name, "
                "or check who you are in Settings."
            ),
        )
    row = await _executor(state, session).next_task()
    if row is None:
        return CommandResult(intent="query", reply="Nothing right now. You're clear.")
    steps = row.micro_steps
    current = (
        steps[row.current_step]["text"]
        if steps and 0 <= row.current_step < len(steps)
        else row.plain_text
    )
    return CommandResult(intent="query", task_id=row.id, reply=current)


async def _dictate(text: str, state: AppState) -> CommandResult:
    result = await parse_skill(
        text,
        provider=state.provider,
        anchors=state.anchors,
        registry=BUILTIN_DETECTORS,
        timeout_s=state.settings.llm.timeout.total_seconds(),
    )
    if result.skill is not None:
        explanation = result.explanation or describe(result.skill)
        reply = f"Here's what I understood: {explanation} Say enable to arm it."
    else:
        reply = result.summary()
    return CommandResult(
        intent="skill_dictation",
        reply=reply,
        skill=result.skill.to_yaml_dict() if result.skill else None,
        explanation=result.explanation or result.summary(),
        confidence=result.confidence,
        unsupported=result.unsupported,
        problems=result.problems,
        needs_confirmation=True,
    )


async def _member_count(session: AsyncSession) -> int:
    """Enrolled, active household members. 0 means identity is off or nobody has consented."""
    from sqlalchemy import func

    from ..db import MemberRow

    result = await session.execute(
        select(func.count()).select_from(MemberRow).where(MemberRow.active)
    )
    return int(result.scalar() or 0)


def _say_minutes(minutes: int) -> str:
    if minutes % (60 * 24) == 0:
        return f"{minutes // (60 * 24)} day" + ("s" if minutes // (60 * 24) != 1 else "")
    if minutes % 60 == 0:
        return f"{minutes // 60} hour" + ("s" if minutes // 60 != 1 else "")
    return f"{minutes} minute" + ("s" if minutes != 1 else "")


__all__ = [
    "CommandResult",
    "command_action",
    "is_query",
    "navigation_target",
    "route_command",
    "snooze_minutes",
]
