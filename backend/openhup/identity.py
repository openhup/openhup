"""Consent-gated household identity (ADR-016).

The whole feature is three small rules, kept in one pure module so they can be reasoned about and
tested without a camera, a database, or a model:

1. **An embedding exists only for a member.** There is no unknown-face store anywhere. The vision
   service computes an embedding, matches it against the enrolled gallery, and emits either member
   ids or "unknown" - it never persists the unknown itself, and neither does anything here.
2. **Ask at most once per anchor per day.** `ConsentAskRow` stores that a question was asked and
   answered, never what the person looked like. Saying "no" once does not nag again today.
3. **Presence is a room fact, never an accusation.** A presence window says Sam was *in the
   kitchen*; nothing here ever emits "Sam left the plates". Identity annotates, it does not blame.

Matching is plain cosine similarity against the enrolled gallery. That is deliberately simple:
face embeddings are high-dimensional and L2-normalised, cosine is the standard distance for them,
and a threshold (default 0.55) is a far more honest gate than a softmax over one or two enrolled
faces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Default cosine floor for a gallery match, matching the face_id detector spec.
DEFAULT_MATCH_THRESHOLD = 0.55


@dataclass(frozen=True, slots=True)
class Member:
    """The enrolled identity the matcher sees: id, name, and embedding vector."""

    id: str
    name: str
    embedding: list[float]


def match(
    embedding: list[float],
    gallery: list[Member],
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Member | None:
    """Best gallery match above the threshold, or None for an unknown face.

    Returns the single best match only when its cosine similarity clears the threshold. Below it,
    the face is unknown - which is exactly the state that triggers the consent question. A gallery
    of one is not a licence to guess: if the distance is too far, it is an unknown person, not a
    confident match.
    """
    if not embedding or not gallery:
        return None
    best: Member | None = None
    best_score = -1.0
    for member in gallery:
        score = cosine(embedding, member.embedding)
        if score > best_score:
            best_score = score
            best = member
    return best if best_score >= threshold else None


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, 0.0 on degenerate input."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def should_ask(unknown_face: bool, asked_here_today: bool) -> bool:
    """Decide whether an unknown face earns the consent question.

    * No unknown face → nothing to ask.
    * Already asked in this anchor today → do not nag (the 24-hour marker; see ADR-016).
    """
    return unknown_face and not asked_here_today


def consent_question(name_hint: str | None = None) -> str:
    """The plain wording of the consent question, before any personality flavours it.

    `name_hint` is the name the person just said, when they volunteered it before being asked -
    e.g. "I'm Sam" - so the question can confirm rather than interrogate.
    """
    if name_hint:
        return (
            f"So you're {name_hint}? I can remember you from now on, so I'll know who's "
            "in the house. You can say no - I'll ask again tomorrow. Want me to remember you?"
        )
    return (
        "I don't recognize you. Do you want me to remember you, so I'll know who's in the "
        "house? You can say no - I'll ask again tomorrow."
    )


def consent_accept(name: str) -> str:
    return f"Got it - I'll remember you as {name}."


def consent_decline() -> str:
    return "No problem. I won't ask again today."


def presence_line(names: list[str], anchor_label: str) -> str:
    """A room fact, never an accusation: who was in a place.

    "Sam was in the kitchen" - and nothing more. Identity annotates presence; the attribution of
    an action to a person is something only the person can declare, never something this derives.
    """
    if not names:
        return ""
    if len(names) == 1:
        return f"{names[0]} was in {anchor_label}."
    if len(names) == 2:
        return f"{names[0]} and {names[1]} were in {anchor_label}."
    return f"{', '.join(names[:-1])}, and {names[-1]} were in {anchor_label}."


__all__ = [
    "DEFAULT_MATCH_THRESHOLD",
    "Member",
    "consent_accept",
    "consent_decline",
    "consent_question",
    "cosine",
    "match",
    "presence_line",
    "should_ask",
]
