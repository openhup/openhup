"""The personality renderer: facts in, one line of voice out.

This is the module that enforces the two promises made in ADR-009 and the README:

* `urgency >= high` never reaches a model. Safety messages are assembled by code from the facts the
  detectors reported, so a burner alert reads the same on every install and cannot be flavoured.
* Anything a model does produce is filtered, and a filtered line falls back to the deterministic
  template rather than being retried.

Every function here returns a `Rendered`, which always carries a usable `text` *and* the `plain`
version. The UI shows `plain` for screen readers, for search, and whenever a household member has
turned personality off - so the personality layer can never be load-bearing for comprehension.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openhup_schemas import (
    Boundaries,
    Personality,
    PersonalitySettings,
    TextSource,
    Urgency,
)

from . import prompts, safety
from .base import Completion, LLMError, LLMProvider, UsageLog


@dataclass(frozen=True, slots=True)
class Rendered:
    """A phrase, plus everything needed to explain or audit it."""

    text: str
    plain: str
    source: TextSource
    #: Set when a model produced text that was rejected, so the personality editor can show why.
    filtered: tuple[str, ...] = ()
    adjustments: tuple[str, ...] = ()

    @property
    def used_llm(self) -> bool:
        return self.source is TextSource.LLM


PLAIN = Personality(
    id="plain",
    display_name="Plain",
    #: Internal safety net, not a choice: what an unknown personality id resolves to, and the
    #: deterministic fallback when the model is down, slow, or filtered. The shipped quiet voice
    #: is `brief` - templates_only is never offered as a personality (ADR-008).
    description=(
        "Internal deterministic fallback. Templates only, no model calls - never offered "
        "as a personality choice."
    ),
    intensity=1,
    templates_only=True,
    boundaries=Boundaries.strict(),
)


class PersonalityRenderer:
    """Renders task, alert, and report text.

    Construct one per request or per engine loop; it holds no state beyond its collaborators.
    """

    def __init__(
        self,
        provider: LLMProvider | None,
        *,
        settings: PersonalitySettings | None = None,
        personalities: dict[str, Personality] | None = None,
        usage: UsageLog | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        self.provider = provider
        self.settings = settings or PersonalitySettings()
        self.personalities = personalities or {"plain": PLAIN}
        self.usage = usage or UsageLog()
        self.timeout_s = timeout_s

    # -- personality resolution --------------------------------------------------------

    def resolve(self, personality_id: str | None) -> Personality:
        """Look up a personality, apply the operator's ceiling, fall back to plain.

        An unknown id resolves to the configured default rather than raising: a renamed personality
        must not stop tasks from being created.
        """
        wanted = personality_id or self.settings.default_personality
        personality = self.personalities.get(wanted) or self.personalities.get(
            self.settings.default_personality
        )
        if personality is None:
            return PLAIN
        clamped = personality.clamped(self.settings.humor_ceiling)
        if clamped.intensity >= 4 and not self.settings.roast_consent:
            # Roast-flavoured voices require explicit consent. Without it, tone down rather than
            # refuse - the user still gets their task.
            return clamped.clamped(3)
        return clamped

    # -- tasks -------------------------------------------------------------------------

    async def task(
        self,
        *,
        title_hint: str,
        anchor_label: str,
        urgency: Urgency = Urgency.LOW,
        personality_id: str | None = None,
        objects: Sequence[str] = (),
        memory: Sequence[str] = (),
    ) -> Rendered:
        """Phrase a task.

        `memory` is the relevant household facts retrieved from the local memory store - context for
        the model, never instructions. It never reaches the `plain` line, which stays tone-free and
        factual no matter what anyone told the assistant.
        """
        personality = self.resolve(personality_id)
        plain = _plain_task(title_hint, anchor_label)

        # Two ways to skip the model: the personality steps aside for urgency, or there is no
        # provider at all. Both must still produce a usable line.
        if not personality.applies_to(urgency) or self.provider is None:
            if personality.templates_only or not personality.applies_to(urgency):
                text = plain
            else:
                text = personality.templates.render(
                    "task",
                    title_hint=title_hint,
                    plain_text=plain,
                    anchor=anchor_label,
                    objects=_objects(objects),
                    step="",
                    facts="; ".join(memory),
                    duration="",
                )
            return Rendered(text=text, plain=plain, source=TextSource.TEMPLATE)

        messages = prompts.phrasing_messages(
            personality,
            title_hint=title_hint,
            anchor_label=anchor_label,
            urgency=urgency,
            objects=objects,
            memory=memory,
        )
        completion = await self._try(messages, purpose="task_phrasing")
        if completion is None:
            return Rendered(text=plain, plain=plain, source=TextSource.TEMPLATE)

        result = safety.check(completion.text, personality.boundaries)
        if not result.allowed:
            # Fall back, do not retry. See the module docstring.
            return Rendered(
                text=plain,
                plain=plain,
                source=TextSource.TEMPLATE,
                filtered=result.violations,
            )
        return Rendered(
            text=result.text,
            plain=plain,
            source=TextSource.LLM,
            adjustments=result.adjustments,
        )

    async def micro_steps(
        self,
        *,
        title_hint: str,
        anchor_label: str,
        count: int,
        personality_id: str | None = None,
        objects: Sequence[str] = (),
        subregion_labels: Sequence[str] = (),
    ) -> list[str]:
        """Produce a ladder of small steps.

        Spatial laddering wins when the anchor has subregions, because "the left third of the shelf"
        is verifiable by the camera and needs no model at all. The LLM is the fallback, not the
        first choice.
        """
        if subregion_labels:
            return [f"Just clear {label.lower()}" for label in subregion_labels[:count]]

        personality = self.resolve(personality_id)
        if self.provider is None or personality.templates_only:
            return _plain_steps(title_hint, count)

        messages = prompts.micro_step_messages(
            personality,
            title_hint=title_hint,
            anchor_label=anchor_label,
            count=count,
            objects=objects,
        )
        completion = await self._try(
            messages, purpose="micro_steps", json_schema=prompts.MICRO_STEP_SCHEMA
        )
        if completion is None:
            return _plain_steps(title_hint, count)

        try:
            steps = completion.json()
        except (ValueError, TypeError):
            return _plain_steps(title_hint, count)
        if not isinstance(steps, list) or not steps:
            return _plain_steps(title_hint, count)

        cleaned: list[str] = []
        for step in steps[:count]:
            if not isinstance(step, str):
                continue
            checked = safety.check(step, personality.boundaries)
            cleaned.append(checked.text if checked.allowed else step.strip()[:80])
        return cleaned or _plain_steps(title_hint, count)

    # -- alerts ------------------------------------------------------------------------

    def alert(
        self,
        *,
        facts: Sequence[str],
        anchor_label: str,
        urgency: Urgency = Urgency.HIGH,
        personality_id: str | None = None,
        summary: str | None = None,
    ) -> Rendered:
        """Phrase an alert. Synchronous, because at high urgency no model is involved.

        A burner alert must not wait on a 7B model warming up, and it must not be witty. The text is
        assembled from the detector facts: "Stove top: burner eq 'on' for 10m; nobody present 5m."
        """
        plain = _plain_alert(summary, facts, anchor_label)
        personality = self.resolve(personality_id)
        if not personality.applies_to(urgency):
            return Rendered(text=plain, plain=plain, source=TextSource.TEMPLATE)
        rendered = personality.templates.render(
            "alert",
            plain_text=plain,
            title_hint=summary or anchor_label,
            anchor=anchor_label,
            facts="; ".join(facts),
            objects="",
            step="",
            duration="",
        )
        checked = safety.check(rendered, personality.boundaries)
        return Rendered(
            text=checked.text if checked.allowed else plain,
            plain=plain,
            source=TextSource.TEMPLATE,
            filtered=checked.violations,
        )

    # -- wins --------------------------------------------------------------------------

    async def win(
        self,
        *,
        anchor_label: str,
        days: float,
        record: bool = False,
        personality_id: str | None = None,
    ) -> Rendered:
        """Celebrate an anchor staying clear for whole days.

        The caring half of the personality layer: the system notices progress, not just problems.
        The claim shape is enforced to be forward-facing - how long things have been good - and
        `backlog_counts` filters any model that drifts backwards. Record wins say "longest yet";
        they never mention the streaks that were broken to get there.
        """
        personality = self.resolve(personality_id)
        plain = _plain_win(anchor_label, days, record)
        if self.provider is None or personality.templates_only:
            text = personality.templates.render(
                "win",
                plain_text=plain,
                anchor=anchor_label,
                days=prompts.days_str(days),
            )
            return Rendered(text=text, plain=plain, source=TextSource.TEMPLATE)

        completion = await self._try(
            prompts.win_messages(
                personality,
                anchor_label=anchor_label,
                days=days,
                record=record,
            ),
            purpose="win_note",
        )
        if completion is None:
            text = personality.templates.render(
                "win",
                plain_text=plain,
                anchor=anchor_label,
                days=prompts.days_str(days),
            )
            return Rendered(text=text, plain=plain, source=TextSource.TEMPLATE)

        result = safety.check(completion.text, personality.boundaries)
        if not result.allowed:
            return Rendered(
                text=plain,
                plain=plain,
                source=TextSource.TEMPLATE,
                filtered=result.violations,
            )
        return Rendered(
            text=result.text,
            plain=plain,
            source=TextSource.LLM,
            adjustments=result.adjustments,
        )

    # -- weekly report -----------------------------------------------------------------

    async def weekly(
        self,
        facts: dict[str, Any],
        *,
        personality_id: str | None = None,
        plain_summary: str = "",
    ) -> Rendered:
        personality = self.resolve(personality_id)
        plain = plain_summary or _plain_weekly(facts)
        if self.provider is None or personality.templates_only:
            return Rendered(text=plain, plain=plain, source=TextSource.TEMPLATE)

        completion = await self._try(
            prompts.weekly_messages(personality, facts), purpose="weekly_report", max_tokens=200
        )
        if completion is None:
            return Rendered(text=plain, plain=plain, source=TextSource.TEMPLATE)

        # Reports are allowed to be a few sentences, so the per-line word cap is relaxed here while
        # the boundary rules stay exactly as strict.
        relaxed = personality.boundaries.model_copy(update={"max_words": 80})
        result = safety.check(completion.text.replace("\n", " "), relaxed)
        if not result.allowed:
            return Rendered(
                text=plain, plain=plain, source=TextSource.TEMPLATE, filtered=result.violations
            )
        return Rendered(text=result.text, plain=plain, source=TextSource.LLM)

    # -- preview -----------------------------------------------------------------------

    async def preview(self, personality_id: str) -> dict[str, str]:
        """Sample output for the personality editor, so tone can be chosen by seeing it."""
        samples = {}
        task = await self.task(
            title_hint="clear the kitchen counter",
            anchor_label="Kitchen counter",
            personality_id=personality_id,
            objects=["mug", "cereal box", "post"],
        )
        samples["task"] = task.text
        samples["alert"] = self.alert(
            facts=["burner eq 'on' for 10m", "nobody present for 5m"],
            anchor_label="Stove top",
            personality_id=personality_id,
            summary="Front burner still on",
        ).text
        samples["note"] = (
            "Safety alerts always read like the one above, whatever personality is chosen."
        )
        return samples

    # -- internals ---------------------------------------------------------------------

    async def _try(
        self,
        messages: Sequence[Any],
        *,
        purpose: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 120,
    ) -> Completion | None:
        """Call the provider, swallowing failure. None means "use the template"."""
        if self.provider is None:
            return None
        prompt_bytes = sum(len(m.content.encode()) for m in messages)
        try:
            completion = await self.provider.complete(
                messages,
                json_schema=json_schema,
                max_tokens=max_tokens,
                temperature=0.7,
                timeout_s=self.timeout_s,
            )
        except (LLMError, TimeoutError, OSError):
            self.usage.record(
                provider=self.provider.caps.name,
                model="?",
                purpose=purpose,
                local=self.provider.caps.local,
                prompt_bytes=prompt_bytes,
                response_bytes=0,
                ok=False,
            )
            return None

        self.usage.record(
            provider=completion.provider,
            model=completion.model,
            purpose=purpose,
            local=self.provider.caps.local,
            prompt_bytes=prompt_bytes,
            response_bytes=len(completion.text.encode()),
        )
        return completion


# --------------------------------------------------------------------------------------
# Deterministic text
# --------------------------------------------------------------------------------------


def _plain_task(title_hint: str, anchor_label: str) -> str:
    hint = title_hint.strip().rstrip(".")
    if not hint:
        return f"Something to do at {anchor_label}."
    return f"{hint[0].upper()}{hint[1:]}."


def _plain_alert(summary: str | None, facts: Sequence[str], anchor_label: str) -> str:
    head = (summary or anchor_label).strip().rstrip(".")
    if not facts:
        return f"{head}."
    return f"{head}: {'; '.join(facts)}."


def _plain_steps(title_hint: str, count: int) -> list[str]:
    """Fallback ladder. Generic, but honest and never absent."""
    base = title_hint.strip().rstrip(".")
    if count <= 1:
        return [f"{base[0].upper()}{base[1:]}."]
    return [f"Start on {base} - just two minutes."] + [
        f"Keep going on {base} ({index + 2} of {count})." for index in range(count - 1)
    ]


def _plain_win(anchor_label: str, days: float, record: bool) -> str:
    line = f"{anchor_label} has stayed clear for {prompts.days_str(days)}."
    if record:
        line += " That is its longest clear stretch in the last 90 days."
    return line


def _plain_weekly(facts: dict[str, Any]) -> str:
    parts = [f"{key.replace('_', ' ')}: {value}" for key, value in facts.items()]
    return "This week - " + ", ".join(parts) + "."


def _objects(objects: Sequence[str]) -> str:
    return ", ".join(objects[:6])


__all__ = ["PLAIN", "PersonalityRenderer", "Rendered"]
