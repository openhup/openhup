"""Natural-language skill parsing.

The point of these tests is that a model's output is never trusted: it is validated, compiled, and
left disabled, and when it is wrong the user gets a form rather than a broken skill.
"""

from __future__ import annotations

import json

from openhup_schemas import SkillOrigin

from openhup.llm import EchoProvider
from openhup.skills.parse import describe, parse_skill

GOOD_SKILL = {
    "skill": {
        "id": "kitchen-counter-clutter",
        "watch": [{"anchor": "kitchen.counter"}],
        "signals": [
            {"id": "clutter", "detector": "clutter_score", "signal": "clutter_level"},
        ],
        "conditions": {"signal": "clutter", "op": "gte", "value": 0.6, "for": "15m"},
        "effect": {"type": "task", "title_hint": "clear the kitchen counter", "urgency": "low"},
        "resolve": {
            "conditions": {"signal": "clutter", "op": "lte", "value": 0.25, "for": "2m"},
            "grace": "5m",
        },
        "limits": {"cooldown": "45m", "max_per_day": 4},
    },
    "unsupported": None,
    "confidence": 0.9,
    "explanation": "Adds a task when the counter stays cluttered for 15 minutes.",
}


def provider_returning(payload) -> EchoProvider:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return EchoProvider({"Request:": text})


async def test_parses_a_good_response(anchors) -> None:
    result = await parse_skill(
        "remind me when the kitchen counter is a mess",
        provider=provider_returning(GOOD_SKILL),
        anchors=anchors,
    )
    assert result.ok
    assert result.skill.id == "kitchen-counter-clutter"
    assert result.confidence == 0.9
    assert result.compiled is not None


async def test_drafts_are_never_enabled(anchors) -> None:
    """Nothing starts watching the house because a model said so."""
    enabled = {**GOOD_SKILL, "skill": {**GOOD_SKILL["skill"], "enabled": True}}
    result = await parse_skill(
        "watch the counter", provider=provider_returning(enabled), anchors=anchors
    )
    assert result.skill.enabled is False
    assert result.skill.origin is SkillOrigin.LLM
    assert result.needs_confirmation


async def test_original_sentence_is_retained(anchors) -> None:
    request = "remind me when the kitchen counter is a disaster"
    result = await parse_skill(request, provider=provider_returning(GOOD_SKILL), anchors=anchors)
    assert result.skill.source_text == request


async def test_unsupported_request_is_reported_not_faked(anchors) -> None:
    result = await parse_skill(
        "tell me when the neighbour's cat is in the garden",
        provider=provider_returning(
            {
                "skill": None,
                "unsupported": "no anchor covers the garden",
                "confidence": 0.1,
                "explanation": "",
            }
        ),
        anchors=anchors,
    )
    assert result.skill is None
    assert "garden" in result.unsupported
    assert "Can't do that yet" in result.summary()


async def test_invalid_skill_gets_one_repair_attempt(anchors) -> None:
    """The echo provider returns the same broken payload twice, so the repair loop must give up."""
    broken = {**GOOD_SKILL, "skill": {**GOOD_SKILL["skill"], "signals": []}}
    result = await parse_skill(
        "watch the counter", provider=provider_returning(broken), anchors=anchors
    )
    assert not result.ok
    assert result.attempts == 2  # initial + one repair, then stop
    assert result.problems


async def test_overlapping_thresholds_are_rejected_even_from_the_llm(anchors) -> None:
    """The compile-time anti-flap check applies to generated skills too."""
    flapping = {
        **GOOD_SKILL,
        "skill": {
            **GOOD_SKILL["skill"],
            "resolve": {
                "conditions": {"signal": "clutter", "op": "lte", "value": 0.7, "for": "2m"},
                "grace": "5m",
            },
        },
    }
    result = await parse_skill(
        "watch the counter", provider=provider_returning(flapping), anchors=anchors
    )
    assert not result.ok
    assert any("open and close repeatedly" in p for p in result.problems)


async def test_non_json_response_falls_back_to_heuristics(anchors) -> None:
    result = await parse_skill(
        "remind me when the kitchen counter is cluttered",
        provider=EchoProvider({"Request:": "Sure! I'd be happy to help with that."}),
        anchors=anchors,
    )
    assert result.heuristic
    assert result.ok
    assert result.skill.watch[0].anchor == "kitchen.counter"


async def test_no_provider_still_produces_a_usable_draft(anchors) -> None:
    """A deployment with no LLM at all is not reduced to hand-editing YAML."""
    result = await parse_skill(
        "remind me when the kitchen counter is a mess", provider=None, anchors=anchors
    )
    assert result.heuristic
    assert result.ok
    assert result.confidence < 0.5
    assert "no LLM available" in result.explanation


async def test_heuristic_covers_the_named_examples(anchors) -> None:
    requests = {
        "alert me if the stove burner stays on": "kitchen.stove",
        "help me watch less tv": "living.tv",
        "one task at a time for this shelf": "office.shelf",
    }
    for request, expected_anchor in requests.items():
        result = await parse_skill(request, provider=None, anchors=anchors)
        assert result.ok, f"{request}: {result.problems or result.unsupported}"
        assert result.skill.watch[0].anchor == expected_anchor, request


async def test_unmatched_request_without_llm_says_so_plainly(anchors) -> None:
    result = await parse_skill("book me a dentist appointment", provider=None, anchors=anchors)
    assert result.skill is None
    assert "skill builder" in result.unsupported


async def test_llm_failure_degrades_to_heuristics(anchors) -> None:
    result = await parse_skill(
        "the kitchen counter gets cluttered",
        provider=EchoProvider(fail=True),
        anchors=anchors,
    )
    assert result.heuristic
    assert result.ok


async def test_empty_request(anchors) -> None:
    result = await parse_skill("   ", provider=None, anchors=anchors)
    assert result.unsupported == "empty request"


def test_describe_renders_a_skill_in_plain_language(anchors) -> None:
    from openhup_schemas import Skill

    skill = Skill.model_validate({**GOOD_SKILL["skill"], "enabled": False})
    sentence = describe(skill)
    assert "Watching kitchen.counter" in sentence
    assert "clutter gte 0.6 for 15m" in sentence
    assert "clears itself" in sentence
    assert "at most 4x a day" in sentence
