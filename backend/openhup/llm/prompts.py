"""Prompt templates.

Kept in one module so the whole surface a model sees is reviewable in a single file - which matters
when the promise is "you can audit what leaves your house".

Two rules run through all of them:

* **Facts in, wording out.** The model is never asked what to do, only how to say what has already
  been decided. It cannot invent a task, escalate an urgency, or claim something the detectors did
  not report.
* **Short.** Every phrasing prompt caps the output hard. A 30-word ceiling is not just style: it
  keeps local inference fast enough to be usable on a mini PC, and it makes the filter's job easy.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from openhup_schemas import DetectorRegistry, Personality, Urgency

from .base import Message, system, user

# --------------------------------------------------------------------------------------
# Skill parsing
# --------------------------------------------------------------------------------------

SKILL_PARSE_SYSTEM = """\
You convert a household request into an OpenHup skill definition.

OpenHup watches named regions of a home ("anchors") with cameras and sensors, and turns what it sees
into tasks, alerts, or metrics. Your only job is to produce the JSON definition. You are not
deciding whether the request is a good idea.

Rules:
1. Use only the anchors and detectors listed below. Never invent an anchor id, detector name, or
   signal name. If nothing fits, set "unsupported" to a short explanation and leave "skill" null.
2. Choose the effect type by intent:
   - task   : something the person should do, and the camera can see when it is done
   - alert  : something unsafe or urgent that needs attention now
   - metric : something to measure over time, with no nagging ("help me cook more")
3. Always give trigger conditions a temporal qualifier ("for", "within", "absent_for") so a single
   noisy frame cannot fire the skill. Two minutes is a sensible floor; fifteen for clutter.
4. Trigger and resolve thresholds must not overlap. If the trigger is `gte 0.6`, resolve well
   below it, for example `lte 0.25`. Overlapping thresholds make a task open and close forever and
   the skill will be rejected.
5. Safety requests (stove, smoke, falls, blocked exits, doors left open) are alerts with urgency
   "high" or "critical", and get no personality.
6. Set limits.cooldown and limits.max_per_day on every task skill. Err towards calm.
7. Prefer one clear condition over three clever ones.

Respond with a single JSON object. No prose, no code fences.\
"""

SKILL_PARSE_USER = """\
Request: {request}

Available anchors:
{anchors}

Available detectors:
{detectors}

{extra}\
"""

#: The response envelope. Deliberately not the bare Skill: a model needs a way to say "I can't do
#: that with these detectors", and a confidence signal drives whether the UI pre-fills or asks.
SKILL_PARSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["skill", "unsupported", "confidence", "explanation"],
    "properties": {
        "skill": {
            "type": ["object", "null"],
            "description": "An OpenHup skill definition, or null if the request cannot be met.",
        },
        "unsupported": {
            "type": ["string", "null"],
            "description": "Why the request cannot be met with the available detectors.",
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {
            "type": "string",
            "description": "One sentence, for the user, explaining what this skill will do.",
        },
    },
}


def skill_parse_messages(
    request: str,
    anchors: Sequence[tuple[str, str]],
    registry: DetectorRegistry,
    *,
    extra_guidance: str = "",
) -> list[Message]:
    """Build the skill-parsing prompt.

    Anchors are passed as (id, label) pairs and detectors are rendered from the registry, so a
    deployment with a custom detector gets it offered automatically with no prompt edit.
    """
    anchor_lines = "\n".join(f"- {anchor_id}: {label}" for anchor_id, label in anchors) or "- none"
    detector_lines = []
    for detector in registry.detectors:
        if detector.dynamic:
            signals = f"any {detector.dynamic_kind} signal you name (see params)"
        else:
            signals = ", ".join(f"{s.key} ({s.kind})" for s in detector.signals)
        required = [p.name for p in detector.params if p.required]
        needs = f" [requires params: {', '.join(required)}]" if required else ""
        detector_lines.append(f"- {detector.name}: {signals}{needs}\n    {detector.description}")

    return [
        system(SKILL_PARSE_SYSTEM),
        user(
            SKILL_PARSE_USER.format(
                request=request.strip(),
                anchors=anchor_lines,
                detectors="\n".join(detector_lines),
                extra=extra_guidance,
            )
        ),
    ]


def repair_messages(original: Sequence[Message], bad_output: str, errors: str) -> list[Message]:
    """One retry, with the validation errors handed back.

    Capped at a single attempt by the gateway. If a model cannot produce a valid skill twice, the UI
    shows the structured form instead - which is a better experience than a third guess.
    """
    return [
        *original,
        Message(role=original[-1].role, content=bad_output),
        user(
            "That did not validate. Fix exactly these problems and return the corrected JSON "
            f"object only:\n{errors}"
        ),
    ]


# --------------------------------------------------------------------------------------
# Phrasing
# --------------------------------------------------------------------------------------

PHRASING_SYSTEM = """\
You write one very short line of text for a home assistant, in a specific voice.

Hard limits:
- {max_words} words maximum. Shorter is better.
- One line. No preamble, no quotes, no explanation, no emoji unless allowed.
- Describe only what you are told. Never invent details, objects, times, or counts.
- Never mention how long something has been undone, how many things are unfinished, or any
  streak the person has broken.
- Never comment on the person's body, habits, character, mental health, or household members.
- Never threaten, guilt, or compare them to anyone.

Voice: {voice_name}
Tone: {tone}
{flavor}
{avoid}
{style}

Output the line and nothing else.\
"""

TASK_PHRASING_USER = """\
Write the line for this task.

What needs doing: {title_hint}
Where: {anchor_label}
{objects}
Urgency: {urgency}
{memory}\
"""

MICRO_STEP_USER = """\
Break this into {count} tiny steps, each doable in under two minutes.

Task: {title_hint}
Where: {anchor_label}
{objects}

Rules:
- Order them easiest first. The first step must feel almost too small to refuse.
- 12 words maximum per step.
- Each step must be a physical action on a real thing, not "tidy up" or "sort out".
- Return a JSON array of strings and nothing else.\
"""

MICRO_STEP_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string", "maxLength": 90},
    "minItems": 1,
    "maxItems": 10,
}

WIN_USER = """\
Write the line celebrating this.

The {anchor} has stayed clear of its task for {days}.
{record}

Rules:
- One short line, in this voice.
- The celebration is forward-facing: how long things have been good.
- Never mention what was left undone, missed, skipped, or how anything was before.
- No comparisons to other people, no counting of unfinished work.
- If the voice is gruff, sarcastic, or shy, the win still gets acknowledged - grudgingly or
  quietly is fine, but it is acknowledged.\
"""

WEEKLY_USER = """\
Write a short weekly note about the household's week.

Numbers you may refer to:
{facts}

Rules:
- 60 words maximum, 2 to 4 sentences.
- Lead with something that went well. There is always something.
- At most one suggestion for next week, and only if the numbers support it.
- Never mention what was missed, skipped, or left undone.
- No lists, no headings.\
"""

#: Alerts are phrased by code, not by a model. Kept here so the whole text surface is in one file.
ALERT_TEMPLATE = "{summary}"


def phrasing_messages(
    personality: Personality,
    *,
    title_hint: str,
    anchor_label: str,
    urgency: Urgency,
    objects: Sequence[str] = (),
    memory: Sequence[str] = (),
) -> list[Message]:
    """Prompt for rendering one task line in a personality's voice.

    `memory` carries the relevant household facts retrieved from the local memory store. They are
    context, never instructions: the model may use them to phrase the task, but the hard limits in
    the system prompt (describe only what you are told, never invent details, no backlog counts)
    apply to them like everything else, and the output filter runs regardless.
    """
    return [
        system(_voice_system(personality)),
        user(
            TASK_PHRASING_USER.format(
                title_hint=title_hint,
                anchor_label=anchor_label,
                objects=_objects_line(objects),
                urgency=urgency.value,
                memory=_memory_block(memory),
            )
        ),
    ]


def micro_step_messages(
    personality: Personality,
    *,
    title_hint: str,
    anchor_label: str,
    count: int,
    objects: Sequence[str] = (),
) -> list[Message]:
    return [
        system(_voice_system(personality)),
        user(
            MICRO_STEP_USER.format(
                count=count,
                title_hint=title_hint,
                anchor_label=anchor_label,
                objects=_objects_line(objects),
            )
        ),
    ]


def win_messages(
    personality: Personality,
    *,
    anchor_label: str,
    days: float,
    record: bool,
) -> list[Message]:
    """Prompt for one win line, in a personality's voice."""
    record_line = (
        "This is the longest clear stretch for this place in the last 90 days." if record else ""
    )
    return [
        system(_voice_system(personality)),
        user(
            WIN_USER.format(
                anchor=anchor_label,
                days=days_str(days),
                record=record_line,
            )
        ),
    ]


def days_str(days: float) -> str:
    """ "3.0" -> "3 days". Shared by prompts and the plain/template rendering."""
    rounded = round(days, 1)
    if rounded == 1:
        return "1 day"
    if rounded.is_integer():
        return f"{int(rounded)} days"
    return f"{rounded} days"


def weekly_messages(personality: Personality, facts: dict[str, Any]) -> list[Message]:
    lines = "\n".join(f"- {key}: {value}" for key, value in facts.items())
    return [system(_voice_system(personality)), user(WEEKLY_USER.format(facts=lines))]


def _voice_system(personality: Personality) -> str:
    flavor = (
        f"Words you may lean on: {', '.join(personality.flavor_words)}"
        if personality.flavor_words
        else ""
    )
    avoid = (
        f"Words to avoid entirely: {', '.join(personality.avoid_words)}"
        if personality.avoid_words
        else ""
    )
    return PHRASING_SYSTEM.format(
        max_words=personality.boundaries.max_words,
        voice_name=personality.display_name,
        tone=", ".join(personality.tone) or "neutral",
        flavor=flavor,
        avoid=avoid,
        style=personality.style_prompt,
    ).replace("\n\n\n", "\n\n")


def _objects_line(objects: Sequence[str]) -> str:
    if not objects:
        return ""
    listed = ", ".join(objects[:6])
    return f"Things the camera can see there: {listed}"


def _memory_block(memory: Sequence[str]) -> str:
    """Render retrieved household facts as prompt context, or nothing at all."""
    if not memory:
        return ""
    lines = "\n".join(f"- {line.strip()}" for line in memory)
    return f"Things to keep in mind (from what you have been told):\n{lines}"


def describe_scene_prompt(anchor_label: str) -> str:
    """Vision-model prompt for an optional scene description.

    Notably absent: anything about people. The prompt asks about surfaces and objects only, because
    a system that starts describing the humans in a room has become something else.
    """
    return (
        f"Describe only the objects and surfaces visible in this photo of {anchor_label}, in one "
        "sentence of at most 25 words. List what is out of place. Do not describe or mention any "
        "people, pets, screens, or documents. Do not guess at anything you cannot see clearly."
    )


def schema_hint(schema: dict[str, Any]) -> str:
    """Compact schema rendering for providers without native JSON mode."""
    return json.dumps(schema, separators=(",", ":"))


__all__ = [
    "ALERT_TEMPLATE",
    "MICRO_STEP_SCHEMA",
    "PHRASING_SYSTEM",
    "SKILL_PARSE_SCHEMA",
    "SKILL_PARSE_SYSTEM",
    "days_str",
    "describe_scene_prompt",
    "micro_step_messages",
    "phrasing_messages",
    "repair_messages",
    "schema_hint",
    "skill_parse_messages",
    "weekly_messages",
    "win_messages",
]
