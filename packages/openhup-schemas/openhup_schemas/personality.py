"""Personality configuration.

Personality is a *rendering layer*, not a system prompt (ADR-009). The facts are decided first by
the skill engine; tone is applied afterwards and then filtered. Two consequences that are enforced
in code rather than requested politely of a model:

* `urgency >= high` bypasses personality entirely. A burner alert is a plain declarative sentence.
* Output that trips a boundary rule falls back to the neutral template. It is not retried with a
  stern reminder - comedy is not worth a second round trip.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import Slug, StrEnum, Urgency


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class BoundaryKind(StrEnum):
    """Categories a personality can be forbidden from touching.

    The regex/heuristic implementation of each lives in `openhup.llm.safety` on the backend; the
    names live here so personality files and the filter cannot drift apart.
    """

    SHAME_LANGUAGE = "shame_language"
    BODY_OR_APPEARANCE = "body_or_appearance_comments"
    MENTAL_HEALTH_DIAGNOSIS = "mental_health_diagnosis"
    STRONG_PROFANITY = "strong_profanity"
    MILD_PROFANITY = "mild_profanity"
    COMPARISONS_TO_OTHERS = "comparisons_to_other_people"
    #: "you've failed 6 times this week" - the single fastest way to get uninstalled.
    BACKLOG_COUNTS = "backlog_counts"
    #: Threats, ultimatums, or fake consequences.
    COERCION = "coercion"
    #: Anything about household members other than the addressee.
    THIRD_PARTY_REMARKS = "third_party_remarks"


class EmojiPolicy(StrEnum):
    NONE = "none"
    SPARING = "sparing"
    ALLOWED = "allowed"


class Boundaries(_Base):
    """Hard limits applied after generation."""

    never: list[BoundaryKind] = Field(default_factory=list)
    max_words: int = Field(default=30, ge=5, le=120)
    emoji: EmojiPolicy = EmojiPolicy.SPARING
    #: Extra deny-list substrings for household-specific sensitivities. Case-insensitive.
    forbidden_phrases: list[str] = Field(default_factory=list)

    @classmethod
    def strict(cls) -> Boundaries:
        """Everything off. What the internal fallback uses, and the floor personalities inherit."""
        return cls(never=list(BoundaryKind), max_words=25, emoji=EmojiPolicy.NONE)


class Templates(_Base):
    """Deterministic fallbacks, used whenever the LLM is unavailable, slow, or filtered.

    Every template must be a complete, usable sentence on its own. These are not placeholders:
    when the model is unavailable they are the entire user-facing voice, and the product still
    works.

    Available fields: {title_hint} {plain_text} {anchor} {objects} {facts} {step} {duration}
    """

    task: str = "{title_hint}"
    task_step: str = "{step}"
    alert: str = "{plain_text}"
    task_done: str = "Done: {title_hint}"
    weekly: str = "{plain_summary}"
    nudge: str = "Still there: {title_hint}"
    #: A milestone win: an anchor staying clear for whole days. Always forward-facing - the claim
    #: is about how long things have been good, never how long anything was left undone.
    win: str = "{plain_text}"

    def render(self, key: str, **values: object) -> str:
        template = getattr(self, key, None)
        if template is None:
            raise KeyError(f"no template named {key!r}")
        try:
            return template.format(**values)
        except KeyError as exc:
            # A personality file referencing an unavailable field must not take down a task.
            raise ValueError(f"template {key!r} references unknown field {exc.args[0]!r}") from exc


class Personality(_Base):
    """A voice.

    Ships with: kind_coach, deadpan_butler, chaos_goblin, drill_sergeant_lite, brief, and the
    gamble pool (ADR-014): friendly, shy, sassy, sarcastic, angry - all at intensity 3 or below,
    so a random draw is never silently clamped by the default humor_ceiling. There is no shipped
    personality that disables the model: `brief` is the quietest voice, and the deterministic
    template fallback is resilience, not a choice (ADR-008).
    """

    id: Slug
    display_name: str
    description: str = ""

    #: 1 gentle … 5 unhinged. User-adjustable, and clamped by the deployment's humor_ceiling.
    intensity: int = Field(default=2, ge=1, le=5)
    tone: list[str] = Field(default_factory=list)
    #: Words to lean on and words to avoid. Fed to the LLM and checked on the way back.
    flavor_words: list[str] = Field(default_factory=list)
    avoid_words: list[str] = Field(default_factory=list)

    boundaries: Boundaries = Field(default_factory=Boundaries)
    templates: Templates = Field(default_factory=Templates)

    #: When True, no LLM call is made at all - templates only. Only the internal fallback sets
    #: this; no shipped personality does (the AI layer is core, ADR-008).
    templates_only: bool = False
    #: Urgency at or above which this personality is bypassed in favour of plain phrasing.
    bypass_at_urgency: Urgency = Urgency.HIGH
    #: Extra instruction appended to the phrasing prompt. Not a jailbreak surface: it is appended
    #: after the boundary instructions, and the output filter runs regardless of what it says.
    style_prompt: str = ""

    @model_validator(mode="after")
    def _high_intensity_needs_boundaries(self) -> Self:
        """A loud personality with no stated limits is a bug. Backfill the non-negotiables."""
        if self.intensity >= 4:
            required = {
                BoundaryKind.SHAME_LANGUAGE,
                BoundaryKind.BODY_OR_APPEARANCE,
                BoundaryKind.MENTAL_HEALTH_DIAGNOSIS,
                BoundaryKind.COMPARISONS_TO_OTHERS,
                BoundaryKind.BACKLOG_COUNTS,
                BoundaryKind.COERCION,
            }
            self.boundaries.never = sorted(set(self.boundaries.never) | required)
        return self

    def clamped(self, ceiling: int) -> Personality:
        """Return a copy with intensity capped, for household members under an operator ceiling."""
        if self.intensity <= ceiling:
            return self
        return self.model_copy(update={"intensity": max(1, ceiling)}, deep=True)

    def applies_to(self, urgency: Urgency) -> bool:
        """False when the personality must step aside for a safety message.

        `templates_only` personalities never apply - that flag exists only on the internal
        fallback, not on shipped voices.
        """
        if self.templates_only:
            return False
        return urgency.rank < self.bypass_at_urgency.rank


class PersonalitySettings(_Base):
    """Deployment-wide personality policy, from `config.yaml`."""

    default_personality: Slug = "kind_coach"
    #: Nobody in the household can be raised past this, whatever their own setting says.
    humor_ceiling: int = Field(default=3, ge=1, le=5)
    #: Explicit opt-in gate for roast-flavoured personalities (intensity >= 4).
    roast_consent: bool = False
    #: Never joke inside these hours, regardless of personality.
    serious_hours: list[str] = Field(default_factory=list)
    #: The personality gamble (ADR-014): on a fresh install, one of the gamble pool is drawn at
    #: random and becomes the default without being announced. Revealed in Settings, documented in
    #: the ADR and the preset file - a mystery to live with, not a secret to keep.
    gamble: bool = False
    #: The pool the draw picks from. Every entry must be a loaded personality; unknown ids are
    #: skipped, and an empty pool disables the gamble.
    gamble_pool: list[str] = Field(
        default_factory=lambda: ["friendly", "shy", "sassy", "sarcastic", "angry"]
    )

    @model_validator(mode="after")
    def _consent_gates_ceiling(self) -> Self:
        if not self.roast_consent:
            self.humor_ceiling = min(self.humor_ceiling, 3)
        return self


__all__ = [
    "Boundaries",
    "BoundaryKind",
    "EmojiPolicy",
    "Personality",
    "PersonalitySettings",
    "Templates",
]
