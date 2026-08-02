"""The output filter.

Personality is a config object applied *after* the facts are decided, and this module is what makes
that safe. It is a deny-list over generated text, not a prompt instruction, because prompt text
cannot be trusted to hold a line and because a filter is testable.

Design decisions worth defending:

* **Fail closed, do not retry.** Filtered output falls back to the deterministic template. Asking
  the model again with a sterner prompt costs another few seconds of local inference and produces
  something equally likely to trip; comedy is not worth the round trip.
* **`backlog_counts` is a boundary.** "You've left this for six days" is the single most reliable
  way to make someone stop opening an app about their house. It is filtered even in roast mode.
* **The word list is conservative and reviewable.** False positives cost a joke. False negatives
  cost trust from someone who told us their sensitivities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from openhup_schemas import Boundaries, BoundaryKind, EmojiPolicy

#: Compiled patterns per boundary. Word-boundary anchored to avoid catching substrings
#: ("classic" must not trip on "ass").
_PATTERNS: dict[BoundaryKind, list[str]] = {
    BoundaryKind.SHAME_LANGUAGE: [
        r"\byou\s+(?:should|ought to)\s+be\s+ashamed\b",
        r"\b(?:lazy|slob|pathetic|useless|hopeless|worthless|disgusting|filthy)\b",
        r"\bwhat(?:'s| is)\s+wrong\s+with\s+you\b",
        r"\b(?:can't|cannot|couldn't)\s+even\b",
        r"\bof\s+course\s+you\s+didn'?t\b",
        r"\btypical\b.{0,20}\byou\b",
        r"\byou\s+never\b",
        r"\byou\s+always\b",
        r"\bagain\?\s*$",
    ],
    BoundaryKind.BODY_OR_APPEARANCE: [
        r"\b(?:fat|obese|skinny|ugly|smelly)\b",
        r"\byou\s+(?:smell|stink)\b",
        r"\byour\s+(?:body|weight|face|hair|skin)\b",
        # "clean the shower" is a legitimate task, so only personal hygiene instructions match.
        r"\b(?:take|have|go\s+take)\s+a\s+(?:shower|bath)\b",
    ],
    BoundaryKind.MENTAL_HEALTH_DIAGNOSIS: [
        # Note: no bare "add" - it collides with the most ordinary instruction in the product
        # ("add the mug to the sink"). ADHD is matched explicitly instead.
        r"\b(?:adhd|a\.d\.h\.d|autis(?:m|tic)|depress(?:ed|ion)|anxiety|ocd|bipolar)\b",
        r"\byour\s+(?:executive\s+function|dopamine|brain\s+chemistry)\b",
        r"\bexecutive\s+dysfunction\b",
        r"\bmedicat(?:ed|ion)\b",
    ],
    BoundaryKind.STRONG_PROFANITY: [
        r"\bf+u+c+k+\w*\b",
        r"\bc+u+n+t+\b",
        r"\bmotherfuck\w*\b",
        r"\bbitch\w*\b",
        r"\bbastard\b",
    ],
    BoundaryKind.MILD_PROFANITY: [
        r"\b(?:damn|hell|crap|shit\w*|arse|ass)\b",
    ],
    BoundaryKind.COMPARISONS_TO_OTHERS: [
        r"\b(?:most|normal|other|real)\s+(?:people|adults|households|families)\b",
        r"\beveryone\s+else\b",
        r"\bunlike\s+you\b",
        r"\bnobody\s+else\b",
    ],
    BoundaryKind.BACKLOG_COUNTS: [
        # "3 unfinished tasks", "5 days in a row", "the 4th time this week"
        r"\b\d+\s+(?:unfinished|pending|outstanding|open|overdue|incomplete)\b",
        r"\b(?:for|in)\s+\d+\s+(?:days?|weeks?)\s+(?:now|straight|in a row)\b",
        r"\b\d+(?:st|nd|rd|th)\s+time\s+this\s+(?:week|month)\b",
        r"\byou(?:'ve| have)\s+(?:ignored|skipped|missed)\s+(?:this|it)\s+\d+\b",
        r"\bstill\s+not\s+done\s+after\b",
    ],
    BoundaryKind.COERCION: [
        r"\bor\s+(?:else|i(?:'ll| will))\b",
        r"\bi(?:'m| am)\s+(?:telling|reporting)\b",
        r"\byou\s+(?:have|need)\s+to\s+.{0,30}\bnow\s+or\b",
        r"\blast\s+warning\b",
        r"\bi\s+will\s+(?:not\s+)?(?:stop|keep)\s+(?:asking|nagging)\s+until\b",
    ],
    BoundaryKind.THIRD_PARTY_REMARKS: [
        r"\byour\s+(?:partner|husband|wife|roommate|flatmate|kids?|children|mother|father)\b",
        r"\bwhoever\s+(?:left|did)\s+(?:this|that)\b",
    ],
}

_COMPILED: dict[BoundaryKind, list[re.Pattern[str]]] = {
    kind: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for kind, patterns in _PATTERNS.items()
}

#: Emoji, roughly. Good enough to enforce a policy; not a Unicode conformance exercise.
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff\u2b00-\u2bff]"
)


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Outcome of filtering one generated string."""

    text: str
    allowed: bool
    #: Which boundary rules tripped, for the audit log and for the personality-tuning UI.
    violations: tuple[str, ...] = ()
    #: Non-blocking adjustments that were applied (emoji stripped, text truncated).
    adjustments: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return self.allowed and not self.adjustments


def check(text: str, boundaries: Boundaries) -> FilterResult:
    """Apply a personality's boundaries to generated text.

    Word-count and emoji limits are *repaired* (truncate, strip) because they are stylistic.
    Boundary violations are *rejected*, because they are not.
    """
    violations: list[str] = []
    adjustments: list[str] = []
    result = text.strip()

    for kind in boundaries.never:
        for pattern in _COMPILED.get(kind, []):
            if pattern.search(result):
                violations.append(kind.value)
                break

    lowered = result.lower()
    for phrase in boundaries.forbidden_phrases:
        if phrase.strip() and phrase.strip().lower() in lowered:
            violations.append(f"forbidden_phrase:{phrase.strip()}")

    if boundaries.emoji is EmojiPolicy.NONE and _EMOJI.search(result):
        result = _EMOJI.sub("", result)
        result = re.sub(r"\s{2,}", " ", result).strip()
        adjustments.append("emoji_removed")
    elif boundaries.emoji is EmojiPolicy.SPARING:
        found = _EMOJI.findall(result)
        if len(found) > 1:
            for extra in found[1:]:
                result = result.replace(extra, "", 1)
            result = re.sub(r"\s{2,}", " ", result).strip()
            adjustments.append("emoji_reduced")

    words = result.split()
    if len(words) > boundaries.max_words:
        result = " ".join(words[: boundaries.max_words]).rstrip(",;:-") + "…"
        adjustments.append("truncated")

    # A model that ignored "one line" and wrote a paragraph is not on-spec; keep the first line.
    if "\n" in result:
        result = result.splitlines()[0].strip()
        adjustments.append("first_line_only")

    return FilterResult(
        text=result,
        allowed=not violations,
        violations=tuple(dict.fromkeys(violations)),
        adjustments=tuple(adjustments),
    )


def explain(kind: BoundaryKind) -> str:
    """Human-readable reason, shown in the personality editor when a preview is rejected."""
    return {
        BoundaryKind.SHAME_LANGUAGE: "Shaming language. It reliably makes people avoid the app.",
        BoundaryKind.BODY_OR_APPEARANCE: "Comments about the person's body or appearance.",
        BoundaryKind.MENTAL_HEALTH_DIAGNOSIS: (
            "Naming or implying a diagnosis. OpenHup is not qualified and it is not welcome."
        ),
        BoundaryKind.STRONG_PROFANITY: "Strong profanity.",
        BoundaryKind.MILD_PROFANITY: "Mild profanity, which this personality has switched off.",
        BoundaryKind.COMPARISONS_TO_OTHERS: (
            "Comparison to other people. Nobody needs a household benchmark."
        ),
        BoundaryKind.BACKLOG_COUNTS: (
            "Counting unfinished work back at the user. The fastest route to uninstalling."
        ),
        BoundaryKind.COERCION: "Threats, ultimatums, or invented consequences.",
        BoundaryKind.THIRD_PARTY_REMARKS: "Remarks about other members of the household.",
    }[kind]


def audit_personality(text: str) -> tuple[str, ...]:
    """Check text against *every* boundary, ignoring which are enabled.

    Used by the personality editor to warn an author that their custom template would be rejected
    by a stricter household member's settings before they save it.
    """
    tripped = []
    for kind, patterns in _COMPILED.items():
        if any(pattern.search(text) for pattern in patterns):
            tripped.append(kind.value)
    return tuple(tripped)


__all__ = ["FilterResult", "audit_personality", "check", "explain"]
