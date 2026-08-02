"""Natural language → draft Skill.

"Remind me when the trash is full" becomes a structured skill the user can see, edit, simulate, and
only then enable. Three properties make this safe to ship:

* **Nothing auto-arms.** The result is a *draft*: `enabled=False`, `origin=llm`, with the original
  sentence retained. The UI shows the compiled meaning and the simulation before anything watches
  anything.
* **The model is never trusted.** Output is validated against the Pydantic schema and then compiled
  against the real detector registry and anchors. One repair attempt, then it gives up and hands the
  user a structured form - a worse experience than magic, and a much better one than a skill that
  silently does the wrong thing.
* **It degrades to nothing.** With no LLM configured, `parse_skill` returns a `ParseResult` marked
  unsupported and the UI falls back to the builder. No feature depends on this working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openhup_schemas import (
    BUILTIN_DETECTORS,
    AlertEffect,
    Anchor,
    DetectorRegistry,
    Skill,
    SkillOrigin,
    TaskEffect,
)
from pydantic import ValidationError

from ..llm import prompts
from ..llm.base import LLMError, LLMProvider
from .compile import CompiledSkill, Lint, SkillCompileError, compile_skill

#: The model gets one correction. See the module docstring for why not more.
MAX_REPAIRS = 1


@dataclass
class ParseResult:
    """Everything the UI needs to show "here is what I think you meant"."""

    request: str
    skill: Skill | None = None
    compiled: CompiledSkill | None = None
    #: One sentence for the user: "This will add a task when the counter stays cluttered for 15m."
    explanation: str = ""
    confidence: float = 0.0
    #: Set when the request cannot be met with the available detectors.
    unsupported: str | None = None
    #: Schema or compile problems that survived the repair attempt.
    problems: list[str] = field(default_factory=list)
    warnings: list[Lint] = field(default_factory=list)
    attempts: int = 0
    #: True when the answer came from the deterministic keyword fallback rather than a model.
    heuristic: bool = False
    #: Why the model path was abandoned, kept for display even when the fallback succeeded. The
    #: user should know they got a guess, without the guess being reported as a failure.
    fallback_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.skill is not None and not self.problems

    @property
    def needs_confirmation(self) -> bool:
        """Always true. Kept as a property so the intent is explicit at the call site."""
        return True

    def summary(self) -> str:
        if self.unsupported:
            return f"Can't do that yet: {self.unsupported}"
        skill = self.skill
        if skill is None or self.problems:
            return "Couldn't turn that into a skill. Try the builder instead."
        return self.explanation or f"Draft skill {skill.id!r} ready to review."


async def parse_skill(
    request: str,
    *,
    provider: LLMProvider | None,
    anchors: dict[str, Anchor],
    registry: DetectorRegistry | None = None,
    timeout_s: float = 45.0,
) -> ParseResult:
    """Turn a sentence into a draft skill.

    Args:
        request: what the user typed.
        provider: the LLM. None or unreachable falls back to the keyword heuristic.
        anchors: known anchors, used both in the prompt and for compile-time validation.
        registry: available detectors. Defaults to the built-ins.
        timeout_s: generous, because a local 7B model on CPU is not fast.
    """
    registry = registry or BUILTIN_DETECTORS
    result = ParseResult(request=request.strip())

    if not result.request:
        result.unsupported = "empty request"
        return result

    if provider is None:
        return _heuristic(result, anchors, registry)

    anchor_pairs = [(a.id, a.label) for a in anchors.values() if a.enabled]
    messages = prompts.skill_parse_messages(result.request, anchor_pairs, registry)

    for attempt in range(MAX_REPAIRS + 1):
        result.attempts = attempt + 1
        try:
            completion = await provider.complete(
                messages,
                json_schema=prompts.SKILL_PARSE_SCHEMA,
                max_tokens=1200,
                temperature=0.2,
                timeout_s=timeout_s,
            )
        except (LLMError, TimeoutError, OSError) as exc:
            result.problems.append(f"LLM unavailable: {exc}")
            return _heuristic(result, anchors, registry)

        try:
            envelope = completion.json()
        except (ValueError, TypeError) as exc:
            decode_error = f"response was not valid JSON: {exc}"
            if attempt >= MAX_REPAIRS:
                result.problems.append(decode_error)
                return _heuristic(result, anchors, registry)
            messages = prompts.repair_messages(messages, completion.text, decode_error)
            continue

        if not isinstance(envelope, dict):
            result.problems.append("response was not a JSON object")
            return _heuristic(result, anchors, registry)

        result.confidence = float(envelope.get("confidence") or 0.0)
        result.explanation = str(envelope.get("explanation") or "").strip()

        if envelope.get("unsupported"):
            result.unsupported = str(envelope["unsupported"])
            return result

        raw_skill = envelope.get("skill")
        if not isinstance(raw_skill, dict):
            result.problems.append("response contained no skill object")
            return _heuristic(result, anchors, registry)

        errors = _validate(raw_skill, anchors, registry, result)
        if errors is None:
            result.problems.clear()
            return result
        if attempt >= MAX_REPAIRS:
            result.problems = errors.splitlines()
            return result
        messages = prompts.repair_messages(messages, completion.text, errors)

    return result


def _validate(
    raw_skill: dict[str, Any],
    anchors: dict[str, Anchor],
    registry: DetectorRegistry,
    result: ParseResult,
) -> str | None:
    """Validate then compile. Returns an error string to feed back, or None on success."""
    # A draft is never enabled, whatever the model says.
    draft = {**raw_skill, "enabled": False, "origin": SkillOrigin.LLM.value}
    draft.setdefault("source_text", result.request)
    draft["source_text"] = result.request

    try:
        skill = Skill.model_validate(draft)
    except ValidationError as exc:
        return "\n".join(
            f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}" for error in exc.errors()
        )

    try:
        compiled = compile_skill(skill, registry=registry, anchors=anchors)
    except SkillCompileError as exc:
        return "\n".join(finding.message for finding in exc.findings)

    result.skill = skill
    result.compiled = compiled
    result.warnings = list(compiled.warnings)
    if not result.explanation:
        result.explanation = describe(skill)
    return None


def describe(skill: Skill) -> str:
    """Plain-language rendering of a compiled skill.

    Shown next to the YAML in the review screen. Some users will read this and never look at the
    YAML, which is the point: the structure should be inspectable without being intimidating.
    """
    from openhup_schemas import iter_predicates

    where = ", ".join(w.anchor or f"all of {w.camera}" for w in skill.watch)
    triggers = [p.describe() for p in iter_predicates(skill.conditions)]

    if isinstance(skill.effect, TaskEffect):
        action = f"add a task ({skill.effect.title_hint})"
    elif isinstance(skill.effect, AlertEffect):
        action = f"raise a {skill.urgency.value} alert"
    else:
        action = f"record the metric {skill.effect.metric}"

    sentence = f"Watching {where}: when {' and '.join(triggers)}, {action}."
    if skill.resolve is not None and skill.resolve.conditions is not None:
        closers = [p.describe() for p in iter_predicates(skill.resolve.conditions)]
        sentence += f" It clears itself when {' or '.join(closers)}."
    elif skill.resolve is not None and skill.resolve.manual_only:
        sentence += " You mark it done yourself."

    caps = []
    if skill.limits.max_per_day:
        caps.append(f"at most {skill.limits.max_per_day}x a day")
    if skill.limits.quiet_hours:
        caps.append(f"never during {skill.limits.quiet_hours}")
    if caps:
        sentence += f" Limits: {', '.join(caps)}."
    return sentence


# --------------------------------------------------------------------------------------
# Deterministic fallback
# --------------------------------------------------------------------------------------

#: Keyword → (detector, signal, effect kind). Covers the requests people actually make first, so a
#: deployment with no LLM at all is still usable rather than being a YAML editor.
_HEURISTICS: list[tuple[tuple[str, ...], str]] = [
    (("trash", "bin", "rubbish", "garbage"), "trash"),
    (("clutter", "mess", "messy", "tidy", "clear the", "counter", "shelf", "desk"), "clutter"),
    (("stove", "burner", "hob", "oven"), "stove"),
    (("tv", "television", "screen", "watching"), "tv"),
    (("dish", "sink", "washing up"), "dishes"),
    (("bowl", "pet", "cat", "dog", "water"), "bowl"),
    (("door", "window"), "door"),
    (("fall", "fallen", "collapsed"), "fall"),
]


def _heuristic(
    result: ParseResult,
    anchors: dict[str, Anchor],
    registry: DetectorRegistry,
) -> ParseResult:
    """Best-effort keyword match, used when no model is available or the model failed.

    It does not try to be clever. It picks a template, points it at the most plausible anchor, and
    tells the user plainly that it guessed - which is far better than an error page.
    """
    result.heuristic = True
    if result.problems:
        # The model path failed, but that is not the user's problem if the fallback works. Keep the
        # reason for display and stop treating it as a failure of this request.
        result.fallback_reason = "; ".join(result.problems)
        result.problems.clear()
    lowered = result.request.lower()
    topic = next((name for words, name in _HEURISTICS if any(w in lowered for w in words)), None)
    if topic is None:
        result.unsupported = (
            "No LLM is configured and this request did not match a known pattern. "
            "Use the skill builder to define it directly."
        )
        return result

    anchor = _guess_anchor(lowered, anchors, topic)
    if anchor is None:
        result.unsupported = (
            f"Recognised this as a {topic} skill, but no anchor matched. "
            f"Create an anchor for the area first."
        )
        return result

    draft = _TEMPLATES[topic](anchor)
    errors = _validate(draft, anchors, registry, result)
    if errors is not None:
        result.problems = errors.splitlines()
        return result

    result.confidence = 0.35
    if result.skill is not None:
        result.explanation = (
            f"Guessed from keywords (no LLM available): {describe(result.skill)} "
            f"Check the thresholds before enabling."
        )
    return result


def _guess_anchor(text: str, anchors: dict[str, Anchor], topic: str) -> Anchor | None:
    """Score anchors by label and id overlap with the request."""
    hints = {
        "trash": ("trash", "bin", "kitchen"),
        "clutter": ("counter", "shelf", "desk", "table", "floor"),
        "stove": ("stove", "hob", "cooker", "kitchen"),
        "tv": ("tv", "living", "lounge"),
        "dishes": ("sink", "kitchen", "dish"),
        "bowl": ("bowl", "pet", "kitchen"),
        "door": ("door", "entry", "hall"),
        "fall": ("walkway", "hall", "living", "floor"),
    }[topic]

    best: tuple[int, Anchor] | None = None
    for anchor in anchors.values():
        if not anchor.enabled:
            continue
        haystack = f"{anchor.id} {anchor.label}".lower()
        score = sum(1 for word in text.split() if len(word) > 3 and word in haystack)
        score += sum(2 for hint in hints if hint in haystack)
        if score and (best is None or score > best[0]):
            best = (score, anchor)
    return best[1] if best else None


def _clutter_template(anchor: Anchor) -> dict[str, Any]:
    return {
        "id": f"{anchor.id.replace('.', '-')}-clutter",
        "description": f"Keep {anchor.label} clear.",
        "watch": [{"anchor": anchor.id}],
        "signals": [
            {
                "id": "clutter",
                "detector": "clutter_score",
                "signal": "clutter_level",
                "params": {"reference": "baseline" if anchor.baseline_ref else "none"},
            }
        ],
        "conditions": {"signal": "clutter", "op": "gte", "value": 0.6, "for": "15m"},
        "effect": {
            "type": "task",
            "mode": "single_task_focus",
            "title_hint": f"clear {anchor.label.lower()}",
            "urgency": "low",
        },
        "resolve": {
            "conditions": {"signal": "clutter", "op": "lte", "value": 0.25, "for": "2m"},
            "grace": "5m",
        },
        "limits": {"cooldown": "45m", "max_per_day": 4},
    }


def _trash_template(anchor: Anchor) -> dict[str, Any]:
    return {
        "id": f"{anchor.id.replace('.', '-')}-trash-full",
        "description": "Take the trash out when it has been full for a while.",
        "watch": [{"anchor": anchor.id}],
        "signals": [
            {
                "id": "fill",
                "detector": "fill_level",
                "signal": "fill_level",
                "params": {"container": "kitchen trash can"},
            }
        ],
        "conditions": {"signal": "fill", "op": "gte", "value": 0.85, "for": "2h"},
        "effect": {"type": "task", "title_hint": "take the trash out", "urgency": "normal"},
        "resolve": {
            "conditions": {"signal": "fill", "op": "lte", "value": 0.3, "for": "1m"},
            "grace": "2m",
        },
        "limits": {"cooldown": "4h", "max_per_day": 2},
    }


def _stove_template(anchor: Anchor) -> dict[str, Any]:
    return {
        "id": f"{anchor.id.replace('.', '-')}-burner-safety",
        "description": "Alert if a burner is left on with nobody around.",
        "watch": [{"anchor": anchor.id}],
        "signals": [
            {
                "id": "burner",
                "detector": "zero_shot_state",
                "signal": "burner_state",
                "params": {
                    "probes": {
                        "on": "a lit stove burner with a flame or glowing element",
                        "off": "an unlit stove burner",
                    }
                },
            },
            {"id": "people", "detector": "object_inventory", "signal": "person_count"},
        ],
        "conditions": {
            "all": [
                {"signal": "burner", "op": "eq", "value": "on", "for": "10m"},
                {"signal": "people", "op": "gte", "value": 1, "absent_for": "5m"},
            ]
        },
        "effect": {"type": "alert", "urgency": "high", "requires_ack": True},
        "resolve": {
            "conditions": {
                "any": [
                    {"signal": "burner", "op": "eq", "value": "off", "for": "30s"},
                    {"signal": "people", "op": "gte", "value": 1},
                ]
            }
        },
        "limits": {"cooldown": "5m"},
    }


def _tv_template(anchor: Anchor) -> dict[str, Any]:
    return {
        "id": f"{anchor.id.replace('.', '-')}-screen-time",
        "description": "Track screen time without nagging about it.",
        "watch": [{"anchor": anchor.id}],
        "signals": [{"id": "screen", "detector": "screen_on", "signal": "screen_on"}],
        "conditions": {"signal": "screen", "op": "eq", "value": True, "for": "2m"},
        "effect": {
            "type": "metric",
            "metric": "tv_on_minutes_per_day",
            "aggregation": "duration_minutes",
            "unit": "min",
        },
        "resolve": {"conditions": {"signal": "screen", "op": "eq", "value": False, "for": "5m"}},
        "snapshot": {"attach": False},
    }


def _dishes_template(anchor: Anchor) -> dict[str, Any]:
    template = _clutter_template(anchor)
    template["id"] = f"{anchor.id.replace('.', '-')}-dishes"
    template["effect"]["title_hint"] = "deal with the dishes"
    return template


def _bowl_template(anchor: Anchor) -> dict[str, Any]:
    return {
        "id": f"{anchor.id.replace('.', '-')}-bowl-empty",
        "description": "Refill when the bowl runs dry.",
        "watch": [{"anchor": anchor.id}],
        "signals": [
            {
                "id": "bowl",
                "detector": "fill_level",
                "signal": "fill_level",
                "params": {"container": "pet water bowl"},
            }
        ],
        "conditions": {"signal": "bowl", "op": "lte", "value": 0.15, "for": "10m"},
        "effect": {"type": "task", "title_hint": "refill the bowl", "urgency": "normal"},
        "resolve": {"conditions": {"signal": "bowl", "op": "gte", "value": 0.6, "for": "1m"}},
        "limits": {"cooldown": "2h", "max_per_day": 3},
    }


def _door_template(anchor: Anchor) -> dict[str, Any]:
    return {
        "id": f"{anchor.id.replace('.', '-')}-left-open",
        "description": "Say something when it is left open.",
        "watch": [{"anchor": anchor.id}],
        "signals": [{"id": "door", "detector": "door_state", "signal": "door_state"}],
        "conditions": {"signal": "door", "op": "eq", "value": "open", "for": "10m"},
        "effect": {"type": "alert", "urgency": "normal"},
        "resolve": {"conditions": {"signal": "door", "op": "eq", "value": "closed", "for": "30s"}},
        "limits": {"cooldown": "15m"},
    }


def _fall_template(anchor: Anchor) -> dict[str, Any]:
    return {
        "id": f"{anchor.id.replace('.', '-')}-fall",
        "description": "Alert if someone is on the floor and not moving.",
        "watch": [{"anchor": anchor.id}],
        "personality": "brief",
        "signals": [
            {"id": "down", "detector": "pose_fall", "signal": "person_down"},
            {"id": "motion", "detector": "pose_fall", "signal": "motion_level"},
        ],
        "conditions": {
            "all": [
                {"signal": "down", "op": "eq", "value": True, "for": "30s"},
                {"signal": "motion", "op": "lte", "value": 0.05, "for": "30s"},
            ]
        },
        "effect": {"type": "alert", "urgency": "critical", "requires_ack": True},
        "resolve": {"conditions": {"signal": "down", "op": "eq", "value": False, "for": "10s"}},
        "limits": {"cooldown": "2m"},
    }


_TEMPLATES = {
    "clutter": _clutter_template,
    "trash": _trash_template,
    "stove": _stove_template,
    "tv": _tv_template,
    "dishes": _dishes_template,
    "bowl": _bowl_template,
    "door": _door_template,
    "fall": _fall_template,
}


__all__ = ["MAX_REPAIRS", "ParseResult", "describe", "parse_skill"]
