"""Consent-gated identity (ADR-016): the pure decision rules.

The tests that matter here are the ones proving the privacy machinery cannot be bent: an unknown
face never matches, a near-miss is not a guess, and presence wording is a room fact rather than
an accusation.
"""

from __future__ import annotations

from openhup.identity import (
    Member,
    consent_accept,
    consent_decline,
    consent_question,
    cosine,
    match,
    presence_line,
    should_ask,
)


def _member(seed: float, member_id: str = "m1", name: str = "Sam") -> Member:
    """A deterministic unit vector, so matches and misses are exact rather than probabilistic."""
    import math

    vector = [math.sin(seed + i) for i in range(16)]
    norm = math.sqrt(sum(x * x for x in vector))
    return Member(id=member_id, name=name, embedding=[x / norm for x in vector])


def test_match_finds_the_right_member() -> None:
    sam = _member(1.0, "sam", "Sam")
    lee = _member(2.0, "lee", "Lee")
    gallery = [sam, lee]
    # Sam's exact embedding matches Sam, well above the threshold.
    assert match(sam.embedding, gallery).id == "sam"


def test_match_unknown_below_threshold_is_none() -> None:
    """A distant embedding must not match, even against a single-member gallery."""
    sam = _member(1.0, "sam", "Sam")
    # A vector pointing a different way entirely (perpendicular-ish), not a scaled Sam.
    stranger = [-sam.embedding[i] if i % 2 else sam.embedding[(i + 8) % 16] for i in range(16)]
    assert match(stranger, [sam], threshold=0.55) is None


def test_match_empty_gallery_is_none() -> None:
    assert match([0.1, 0.2], [], threshold=0.55) is None
    assert match([], [_member(1.0)], threshold=0.55) is None


def test_match_threshold_is_a_gate_not_a_softmax() -> None:
    """A two-member gallery must not force a winner: below threshold means unknown."""
    # Two orthogonal members: Sam is e1, Lee is e2.
    sam = Member(id="sam", name="Sam", embedding=[1.0, 0.0] + [0.0] * 14)
    lee = Member(id="lee", name="Lee", embedding=[0.0, 1.0] + [0.0] * 14)
    # A vector halfway between the two has cosine cos(45°) with each - below a strict threshold,
    # so it must stay unknown rather than being forced onto whichever is nearest.
    between = [0.5, 0.5] + [0.0] * 14
    assert match(between, [sam, lee], threshold=0.9) is None


def test_cosine_identical_is_one_orthogonal_is_zero() -> None:
    assert abs(cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9
    assert cosine([], []) == 0.0


def test_should_ask_rules() -> None:
    # An unknown face with no marker earns the question.
    assert should_ask(unknown_face=True, asked_here_today=False)
    # Already asked today: no nag.
    assert not should_ask(unknown_face=True, asked_here_today=True)
    # No unknown face: nothing to ask.
    assert not should_ask(unknown_face=False, asked_here_today=False)


def test_consent_question_offers_an_out() -> None:
    q = consent_question()
    assert "remember" in q
    assert "no" in q  # the out is stated, not hidden


def test_consent_replies_are_unambiguous() -> None:
    assert "Sam" in consent_accept("Sam")
    assert "again today" in consent_decline()


def test_presence_line_is_a_room_fact_never_an_accusation() -> None:
    assert presence_line(["Sam"], "the kitchen") == "Sam was in the kitchen."
    assert presence_line(["Sam", "Lee"], "the kitchen") == "Sam and Lee were in the kitchen."
    assert "left" not in presence_line(["Sam"], "the kitchen")
    assert "forgot" not in presence_line(["Sam"], "the kitchen")
    assert presence_line([], "the kitchen") == ""
