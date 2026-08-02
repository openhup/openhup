"""The personality layer and its safety filter.

The tests that matter here are the ones proving OpenHup cannot be talked into being cruel, and
cannot be prevented from being clear about a burner left on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openhup_schemas import (
    Boundaries,
    BoundaryKind,
    EmojiPolicy,
    Personality,
    PersonalitySettings,
    TextSource,
    Urgency,
)

from openhup.llm import EchoProvider, PersonalityRenderer, audit_personality, check
from openhup.llm.render import PLAIN

PRESETS = Path(__file__).resolve().parents[2] / "examples" / "personalities" / "personalities.yaml"


@pytest.fixture
def presets() -> dict[str, Personality]:
    raw = yaml.safe_load(PRESETS.read_text())
    return {entry["id"]: Personality.model_validate(entry) for entry in raw}


@pytest.fixture
def goblin(presets) -> Personality:
    return presets["chaos_goblin"]


def renderer(
    responses=None, *, personalities=None, settings=None, fail=False
) -> PersonalityRenderer:
    return PersonalityRenderer(
        EchoProvider(responses or {}, fail=fail),
        personalities=personalities or {"plain": PLAIN},
        settings=settings or PersonalitySettings(roast_consent=True, humor_ceiling=5),
    )


# ------------------------------------------------------------------ the shipped presets


def test_all_presets_parse(presets) -> None:
    assert set(presets) == {
        "kind_coach",
        "deadpan_butler",
        "chaos_goblin",
        "drill_sergeant_lite",
        "brief",
        "friendly",
        "shy",
        "sassy",
        "sarcastic",
        "angry",
    }


def test_gamble_pool_is_never_clamped_in_secret() -> None:
    """The gamble draws from five presets, all at intensity 3 or below.

    This is the invariant that keeps a mystery voice honest: under the default humor_ceiling of 3
    with no roast_consent, a drawn personality must be exactly what was drawn - never a quieter
    version of itself (ADR-014).
    """
    from openhup.personality import GAMBLE_POOL

    assert set(GAMBLE_POOL) == {"friendly", "shy", "sassy", "sarcastic", "angry"}


def test_no_preset_template_would_trip_its_own_filter(presets) -> None:
    """A shipped personality whose own fallback text gets filtered would be embarrassing."""
    for personality in presets.values():
        for field in ("task", "task_step", "alert", "task_done", "nudge", "win"):
            text = getattr(personality.templates, field)
            rendered = text.format(
                title_hint="clear the counter",
                plain_text="Counter is cluttered.",
                anchor="kitchen counter",
                objects="mug",
                facts="clutter high",
                step="move the mug",
                duration="15m",
                plain_summary="a normal week",
                days="3 days",
            )
            result = check(rendered, personality.boundaries)
            assert result.allowed, f"{personality.id}.{field} trips {result.violations}"


def test_loud_presets_have_the_non_negotiable_boundaries(presets) -> None:
    for personality in presets.values():
        if personality.intensity >= 4:
            never = set(personality.boundaries.never)
            assert BoundaryKind.SHAME_LANGUAGE in never
            assert BoundaryKind.BACKLOG_COUNTS in never
            assert BoundaryKind.COERCION in never


def test_every_preset_steps_aside_for_safety(presets) -> None:
    for personality in presets.values():
        assert not personality.applies_to(Urgency.HIGH), personality.id
        assert not personality.applies_to(Urgency.CRITICAL), personality.id


# ------------------------------------------------------------------ the filter


@pytest.mark.parametrize(
    ("text", "boundary"),
    [
        ("Don't be so lazy about the counter", BoundaryKind.SHAME_LANGUAGE),
        ("You never finish anything", BoundaryKind.SHAME_LANGUAGE),
        ("Maybe take a shower while you're up", BoundaryKind.BODY_OR_APPEARANCE),
        ("Classic ADHD counter situation", BoundaryKind.MENTAL_HEALTH_DIAGNOSIS),
        ("Your executive function is losing", BoundaryKind.MENTAL_HEALTH_DIAGNOSIS),
        ("Most people manage this fine", BoundaryKind.COMPARISONS_TO_OTHERS),
        ("You have 6 unfinished tasks waiting", BoundaryKind.BACKLOG_COUNTS),
        ("This has been here for 4 days now", BoundaryKind.BACKLOG_COUNTS),
        ("Do it now or else", BoundaryKind.COERCION),
        ("I will keep nagging until you move", BoundaryKind.COERCION),
        ("Tell your roommate to help", BoundaryKind.THIRD_PARTY_REMARKS),
    ],
)
def test_filter_rejects_the_things_it_must(text: str, boundary: BoundaryKind) -> None:
    result = check(text, Boundaries(never=[boundary]))
    assert not result.allowed
    assert boundary.value in result.violations


def test_filter_allows_playful_text_aimed_at_the_mess(goblin) -> None:
    result = check(
        "The counter has grown a forbidden artifact hoard. Evict three items. Go.",
        goblin.boundaries,
    )
    assert result.allowed


def test_filter_does_not_trip_on_innocent_substrings() -> None:
    """ "classic" must not match "ass"; "add" must not match a diagnosis."""
    boundaries = Boundaries(never=list(BoundaryKind))
    for phrase in ["A classic mug situation", "Add the mug to the sink", "Assorted items here"]:
        assert check(phrase, boundaries).allowed, phrase


def test_word_cap_truncates_rather_than_rejects() -> None:
    long_text = " ".join(["word"] * 60)
    result = check(long_text, Boundaries(max_words=10))
    assert result.allowed
    assert "truncated" in result.adjustments
    assert len(result.text.split()) <= 11  # 10 words plus the ellipsis token


def test_emoji_policy_none_strips() -> None:
    result = check("Clear the counter 🧹✨", Boundaries(emoji=EmojiPolicy.NONE))
    assert "🧹" not in result.text
    assert "emoji_removed" in result.adjustments


def test_emoji_policy_sparing_keeps_one() -> None:
    result = check("Counter 🧹 needs help ✨🎉", Boundaries(emoji=EmojiPolicy.SPARING))
    assert result.text.count("🧹") == 1
    assert "✨" not in result.text


def test_multiline_output_is_reduced_to_one_line() -> None:
    result = check("Clear the counter.\nThen wipe it.\nThen relax.", Boundaries())
    assert "\n" not in result.text
    assert "first_line_only" in result.adjustments


def test_forbidden_phrases_are_household_specific() -> None:
    boundaries = Boundaries(forbidden_phrases=["the shed"])
    result = check("Something about the shed again", boundaries)
    assert not result.allowed
    assert any("forbidden_phrase" in v for v in result.violations)


def test_audit_reports_every_boundary_regardless_of_settings() -> None:
    tripped = audit_personality("Don't be lazy, most people manage this")
    assert "shame_language" in tripped
    assert "comparisons_to_other_people" in tripped


# ------------------------------------------------------------------ rendering


async def test_task_uses_the_llm_when_it_behaves(goblin) -> None:
    render = renderer(
        {"clear the kitchen counter": "The counter has formed a small hoard. Evict two items."},
        personalities={"chaos_goblin": goblin},
    )
    result = await render.task(
        title_hint="clear the kitchen counter",
        anchor_label="Kitchen counter",
        personality_id="chaos_goblin",
    )
    assert result.source is TextSource.LLM
    assert "hoard" in result.text
    # The plain version is always available, whatever the model said.
    assert result.plain == "Clear the kitchen counter."


async def test_filtered_output_falls_back_to_the_template(goblin) -> None:
    """The model got mean. The user must never see it, and we must not retry."""
    provider = EchoProvider({"clear the kitchen counter": "Don't be so lazy about this."})
    render = PersonalityRenderer(
        provider,
        personalities={"chaos_goblin": goblin},
        settings=PersonalitySettings(roast_consent=True, humor_ceiling=5),
    )
    result = await render.task(
        title_hint="clear the kitchen counter",
        anchor_label="Kitchen counter",
        personality_id="chaos_goblin",
    )
    assert result.source is TextSource.TEMPLATE
    assert "shame_language" in result.filtered
    assert "lazy" not in result.text
    assert len(provider.calls) == 1  # no retry


async def test_llm_failure_still_produces_a_task(goblin) -> None:
    render = renderer(fail=True, personalities={"chaos_goblin": goblin})
    result = await render.task(
        title_hint="clear the kitchen counter",
        anchor_label="Kitchen counter",
        personality_id="chaos_goblin",
    )
    assert result.text == "Clear the kitchen counter."
    assert result.source is TextSource.TEMPLATE


async def test_no_provider_at_all_still_produces_a_task(goblin) -> None:
    render = PersonalityRenderer(None, personalities={"chaos_goblin": goblin})
    result = await render.task(title_hint="clear the counter", anchor_label="Counter")
    assert result.text
    assert result.source is TextSource.TEMPLATE


async def test_brief_personality_calls_the_model(presets) -> None:
    """The quietest shipped voice still uses the AI layer - no personality turns it off."""
    provider = EchoProvider({"clear": "Clear the counter."})
    render = PersonalityRenderer(provider, personalities={"brief": presets["brief"]})
    result = await render.task(
        title_hint="clear the counter", anchor_label="Counter", personality_id="brief"
    )
    assert provider.calls  # the model was asked, not skipped
    assert result.source is TextSource.LLM


async def test_internal_fallback_never_calls_the_model() -> None:
    """The deterministic safety net (unknown personality, model down) is templates-only."""
    provider = EchoProvider({"clear": "something witty"})
    render = PersonalityRenderer(provider, personalities={"plain": PLAIN})
    result = await render.task(
        title_hint="clear the counter", anchor_label="Counter", personality_id="plain"
    )
    assert provider.calls == []
    assert result.text == "Clear the counter."


async def test_high_urgency_task_bypasses_the_model(goblin) -> None:
    provider = EchoProvider({"stove": "hehe the stove is a dragon now"})
    render = PersonalityRenderer(
        provider,
        personalities={"chaos_goblin": goblin},
        settings=PersonalitySettings(roast_consent=True, humor_ceiling=5),
    )
    result = await render.task(
        title_hint="turn off the stove",
        anchor_label="Stove top",
        urgency=Urgency.HIGH,
        personality_id="chaos_goblin",
    )
    assert provider.calls == []
    assert result.text == "Turn off the stove."


def test_alerts_are_assembled_from_facts_not_generated(goblin) -> None:
    render = renderer(personalities={"chaos_goblin": goblin})
    result = render.alert(
        facts=["burner eq 'on' for 10m", "nobody present for 5m"],
        anchor_label="Stove top",
        summary="Front burner still on",
        personality_id="chaos_goblin",
    )
    assert result.text == "Front burner still on: burner eq 'on' for 10m; nobody present for 5m."
    assert result.source is TextSource.TEMPLATE


def test_low_urgency_alert_may_carry_personality(goblin) -> None:
    render = renderer(personalities={"chaos_goblin": goblin})
    result = render.alert(
        facts=["door open 10m"],
        anchor_label="Front door",
        urgency=Urgency.LOW,
        personality_id="chaos_goblin",
    )
    assert result.plain.startswith("Front door:")


# ------------------------------------------------------------------ intensity policy


def test_humor_ceiling_clamps_intensity(goblin) -> None:
    render = PersonalityRenderer(
        None,
        personalities={"chaos_goblin": goblin},
        settings=PersonalitySettings(roast_consent=True, humor_ceiling=2),
    )
    assert render.resolve("chaos_goblin").intensity == 2


def test_roast_needs_consent(goblin) -> None:
    render = PersonalityRenderer(
        None,
        personalities={"chaos_goblin": goblin},
        settings=PersonalitySettings(roast_consent=False, humor_ceiling=5),
    )
    assert render.resolve("chaos_goblin").intensity <= 3


def test_unknown_personality_falls_back_instead_of_raising() -> None:
    render = PersonalityRenderer(
        None,
        personalities={"plain": PLAIN},
        settings=PersonalitySettings(default_personality="plain"),
    )
    assert render.resolve("was_deleted_yesterday").id == "plain"


# ------------------------------------------------------------------ micro steps


async def test_spatial_ladder_needs_no_model() -> None:
    provider = EchoProvider()
    render = PersonalityRenderer(provider, personalities={"plain": PLAIN})
    steps = await render.micro_steps(
        title_hint="clear the shelf",
        anchor_label="Office shelf",
        count=3,
        subregion_labels=["Left third", "Middle third", "Right third"],
    )
    assert steps == ["Just clear left third", "Just clear middle third", "Just clear right third"]
    assert provider.calls == []


async def test_semantic_ladder_uses_the_model(goblin) -> None:
    render = renderer(
        {"Break this into 3 tiny steps": '["Move the three mugs", "Bin the wrappers", "Wipe it"]'},
        personalities={"chaos_goblin": goblin},
    )
    steps = await render.micro_steps(
        title_hint="clear the shelf",
        anchor_label="Office shelf",
        count=3,
        personality_id="chaos_goblin",
        objects=["mug", "wrapper"],
    )
    assert steps == ["Move the three mugs", "Bin the wrappers", "Wipe it"]


async def test_ladder_falls_back_when_the_model_returns_nonsense(goblin) -> None:
    render = renderer(
        {"Break this into": "I'm afraid I can't help with that"},
        personalities={"chaos_goblin": goblin},
    )
    steps = await render.micro_steps(
        title_hint="clear the shelf",
        anchor_label="Shelf",
        count=3,
        personality_id="chaos_goblin",
    )
    assert len(steps) == 3
    assert all(isinstance(step, str) and step for step in steps)


# ------------------------------------------------------------------ weekly report


async def test_weekly_report_is_generated_then_filtered(goblin) -> None:
    render = renderer(
        {"weekly note": "Three cooked meals and a clear counter most mornings. Nice week."},
        personalities={"chaos_goblin": goblin},
    )
    result = await render.weekly(
        {"cook_sessions": 3, "clean_streak_days": 4}, personality_id="chaos_goblin"
    )
    assert "cooked meals" in result.text
    assert result.plain.startswith("This week - ")


async def test_weekly_report_rejects_backlog_shaming(goblin) -> None:
    render = renderer(
        {"weekly note": "You left 7 unfinished tasks this week."},
        personalities={"chaos_goblin": goblin},
    )
    result = await render.weekly({"tasks": 7}, personality_id="chaos_goblin")
    assert result.source is TextSource.TEMPLATE
    assert "backlog_counts" in result.filtered


# ------------------------------------------------------------------ audit trail


async def test_usage_log_records_every_call(goblin) -> None:
    render = renderer({"clear": "Counter has a hoard."}, personalities={"chaos_goblin": goblin})
    await render.task(
        title_hint="clear the counter", anchor_label="Counter", personality_id="chaos_goblin"
    )
    assert len(render.usage.entries) == 1
    entry = render.usage.entries[0]
    assert entry["purpose"] == "task_phrasing"
    assert entry["local"] is True
    assert entry["prompt_bytes"] > 0
    assert render.usage.remote_calls == 0


async def test_preview_shows_both_voices(goblin) -> None:
    render = renderer(
        {"clear the kitchen counter": "Counter: hoard detected."},
        personalities={"chaos_goblin": goblin},
    )
    preview = await render.preview("chaos_goblin")
    assert "hoard" in preview["task"]
    assert preview["alert"].startswith("Front burner still on:")


# ------------------------------------------------------------------ wins


async def test_win_uses_the_model_when_it_behaves(presets) -> None:
    """A win line is voice-flavoured when a model is there, but the plain version always exists."""
    render = renderer(
        {"Kitchen counter": "The counter has behaved itself for 3 days. Fancy that."},
        personalities={"sassy": presets["sassy"]},
    )
    result = await render.win(anchor_label="Kitchen counter", days=3.0, personality_id="sassy")
    assert result.source is TextSource.LLM
    assert result.plain.startswith("Kitchen counter has stayed clear for 3 days")


async def test_win_filtered_line_falls_back_to_plain(presets) -> None:
    """A model that turns a win into a dig gets nothing - not even a second try."""
    render = renderer(
        {"Kitchen counter": "At last! You're not completely useless after all."},
        personalities={"sassy": presets["sassy"]},
    )
    result = await render.win(anchor_label="Kitchen counter", days=3.0, personality_id="sassy")
    assert result.source is TextSource.TEMPLATE
    assert "shame_language" in result.filtered
    assert "useless" not in result.text


async def test_win_record_mentions_longest_never_backwards() -> None:
    render = PersonalityRenderer(None, personalities={"plain": PLAIN})
    result = await render.win(anchor_label="Kitchen counter", days=12.4, record=True)
    assert "12.4 days" in result.text
    assert "longest clear stretch in the last 90 days" in result.text
    assert "left" not in result.text


async def test_win_shy_template_is_short_and_quiet(presets) -> None:
    render = PersonalityRenderer(None, personalities={"shy": presets["shy"]})
    result = await render.win(anchor_label="Kitchen counter", days=3.0, personality_id="shy")
    assert result.text == "Kitchen counter stayed clear 3 days. That's... good."
