"""Schema-level tests: parsing, validation, and the guardrails baked into the models."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

import pytest
from pydantic import ValidationError

from openhup_schemas import (
    AllOf,
    Anchor,
    Camera,
    Goal,
    GoalDirection,
    GoalStatus,
    Observation,
    Op,
    Personality,
    Signal,
    SignalKey,
    SignalKind,
    SignalPredicate,
    Skill,
    TimeWindow,
    Urgency,
    format_duration,
    load_skill_yaml,
    new_ulid,
    parse_duration,
    ulid_timestamp,
)

UTC = UTC


# ---------------------------------------------------------------------------- durations


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30s", timedelta(seconds=30)),
        ("15m", timedelta(minutes=15)),
        ("1h30m", timedelta(hours=1, minutes=30)),
        ("2d", timedelta(days=2)),
        ("4h", timedelta(hours=4)),
        ("500ms", timedelta(milliseconds=500)),
        ("1d2h3m4s", timedelta(days=1, hours=2, minutes=3, seconds=4)),
    ],
)
def test_parse_duration(text: str, expected: timedelta) -> None:
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "15", "banana", "1w", "m15", "-5m"])
def test_parse_duration_rejects_garbage(text: str) -> None:
    """A unit-less `for: 15` in YAML is a bug, not "15 seconds"."""
    with pytest.raises((ValueError, TypeError)):
        parse_duration(text)


@pytest.mark.parametrize("text", ["30s", "15m", "1h30m", "2d", "1d2h3m4s", "500ms"])
def test_duration_roundtrip(text: str) -> None:
    assert format_duration(parse_duration(text)) == text


# ---------------------------------------------------------------------------- ULIDs


def test_ulid_is_sortable_by_time() -> None:
    early = new_ulid(ts_ms=1_000_000_000_000)
    late = new_ulid(ts_ms=1_000_000_001_000)
    assert early < late
    assert len(early) == 26


def test_ulid_timestamp_roundtrip() -> None:
    stamp = 1_755_000_000_000
    recovered = ulid_timestamp(new_ulid(ts_ms=stamp))
    assert recovered == datetime.fromtimestamp(stamp / 1000, tz=UTC)


# ---------------------------------------------------------------------------- time windows


def test_time_window_between_shorthand() -> None:
    window = TimeWindow.model_validate({"between": ["07:00", "22:00"], "tz": "UTC"})
    assert window.start == time(7, 0)
    assert window.end == time(22, 0)
    assert not window.wraps_midnight


def test_time_window_wraps_midnight() -> None:
    quiet = TimeWindow.model_validate({"between": ["22:00", "07:00"], "tz": "UTC"})
    assert quiet.wraps_midnight
    assert quiet.contains(datetime(2026, 8, 17, 23, 30, tzinfo=UTC))
    assert quiet.contains(datetime(2026, 8, 17, 3, 0, tzinfo=UTC))
    assert not quiet.contains(datetime(2026, 8, 17, 12, 0, tzinfo=UTC))


def test_time_window_weekday_filter() -> None:
    weekdays = TimeWindow.model_validate(
        {"between": ["09:00", "17:00"], "tz": "UTC", "days": ["mon", "tue"]}
    )
    monday = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)
    saturday = datetime(2026, 8, 22, 10, 0, tzinfo=UTC)
    assert weekdays.contains(monday)
    assert not weekdays.contains(saturday)


def test_time_window_rejects_naive_datetime() -> None:
    window = TimeWindow.model_validate({"between": ["09:00", "17:00"]})
    with pytest.raises(ValueError, match="timezone-aware"):
        window.contains(datetime(2026, 8, 17, 10, 0))


# ---------------------------------------------------------------------------- observations


def test_signal_kind_must_match_value() -> None:
    Signal(key="clutter_level", kind=SignalKind.SCALAR, value=0.7)
    with pytest.raises(ValidationError):
        Signal(key="clutter_level", kind=SignalKind.SCALAR, value="very messy")
    with pytest.raises(ValidationError):
        Signal(key="person_count", kind=SignalKind.COUNT, value=-1)
    with pytest.raises(ValidationError):
        Signal(key="screen_on", kind=SignalKind.BOOLEAN, value=1.0)


def test_observation_parses_wire_format() -> None:
    raw = {
        "schema": "openhup.observation/v1",
        "id": new_ulid(),
        "ts": "2026-08-17T12:34:56.789Z",
        "source": {"camera_id": "kitchen", "anchor_id": "kitchen.counter", "frame_seq": 918273},
        "detector": {
            "name": "clutter_score",
            "version": "clip-vit-b32-int8@1.2",
            "backend": "onnxruntime-openvino",
        },
        "signals": [
            {"key": "clutter_level", "kind": "scalar", "value": 0.72, "confidence": 0.81},
            {"key": "object_count", "kind": "count", "value": 11},
            {"key": "objects", "kind": "set", "value": ["cup", "plate"]},
        ],
        "media": {"snapshot_ref": "snap://2026/08/17/kitchen/x.jpg", "ttl_s": 604800},
        "cost_ms": 42,
    }
    obs = Observation.model_validate(raw)
    assert obs.signal("clutter_level").value == pytest.approx(0.72)
    assert SignalKey("kitchen.counter", "clutter_score", "objects") in obs.keys()


def test_observation_rejects_duplicate_signal_keys() -> None:
    with pytest.raises(ValidationError, match="duplicate signal"):
        Observation.model_validate(
            {
                "source": {"camera_id": "k", "anchor_id": "k.counter"},
                "detector": {"name": "d", "version": "1"},
                "signals": [
                    {"key": "x", "kind": "scalar", "value": 1.0},
                    {"key": "x", "kind": "scalar", "value": 2.0},
                ],
            }
        )


def test_signal_key_roundtrip() -> None:
    key = SignalKey("kitchen.counter", "clutter_score", "clutter_level")
    assert SignalKey.parse(str(key)) == key


# ---------------------------------------------------------------------------- skills

KITCHEN_SKILL = """
id: kitchen-clutter-buster
enabled: true
description: Keep the counter clear during waking hours.
watch:
  - anchor: kitchen.counter
signals:
  - id: clutter
    detector: clutter_score
    signal: clutter_level
    params: {reference: baseline}
conditions:
  all:
    - {signal: clutter, op: gte, value: 0.6, for: 15m}
    - {time_window: {between: ["07:00", "22:00"], tz: UTC}}
effect:
  type: task
  mode: single_task_focus
  title_hint: clear the kitchen counter
  micro_steps: auto:3
  urgency: low
resolve:
  conditions:
    all:
      - {signal: clutter, op: lte, value: 0.25, for: 2m}
  grace: 5m
limits:
  cooldown: 45m
  max_per_day: 4
snapshot:
  attach: true
  retention: 7d
  redact: [faces]
"""


def test_kitchen_skill_parses() -> None:
    skill = load_skill_yaml(KITCHEN_SKILL)
    assert skill.id == "kitchen-clutter-buster"
    assert skill.effect.type == "task"
    assert skill.effect.micro_steps.count == 3
    assert skill.urgency is Urgency.LOW
    # Horizon is the deepest history any predicate needs, across both trees.
    assert skill.horizon == timedelta(minutes=15)


def test_skill_roundtrips_through_yaml_dict() -> None:
    skill = load_skill_yaml(KITCHEN_SKILL)
    again = Skill.model_validate(skill.to_yaml_dict())
    assert again == skill
    # `for` must survive serialisation under its alias, not leak as `for_`.
    dumped = skill.to_yaml_dict()
    assert dumped["conditions"]["all"][0]["for"] == "15m"


def test_skill_rejects_undeclared_signal() -> None:
    broken = KITCHEN_SKILL.replace("signal: clutter, op: gte", "signal: cluter, op: gte")
    with pytest.raises(ValidationError, match="undeclared signal"):
        load_skill_yaml(broken)


def test_skill_rejects_unknown_key() -> None:
    """extra=forbid everywhere: a typo in a skill file fails on save, not at 3am."""
    with pytest.raises(ValidationError):
        load_skill_yaml(KITCHEN_SKILL + "\nurgencyy: high\n")


def test_task_skill_requires_resolve_block() -> None:
    without_resolve = KITCHEN_SKILL.split("resolve:")[0]
    with pytest.raises(ValidationError, match="needs a 'resolve' block"):
        load_skill_yaml(without_resolve)


def test_resolve_needs_a_way_out() -> None:
    with pytest.raises(ValidationError, match="episode can never end"):
        load_skill_yaml(
            KITCHEN_SKILL.replace(
                """resolve:
  conditions:
    all:
      - {signal: clutter, op: lte, value: 0.25, for: 2m}
  grace: 5m""",
                """resolve:
  grace: 5m""",
            )
        )


def test_predicate_rejects_two_temporal_qualifiers() -> None:
    with pytest.raises(ValidationError, match="multiple temporal qualifiers"):
        SignalPredicate.model_validate(
            {"signal": "clutter", "op": "gte", "value": 0.6, "for": "15m", "within": "1h"}
        )


def test_predicate_max_gap_requires_for() -> None:
    with pytest.raises(ValidationError, match="max_gap only applies"):
        SignalPredicate.model_validate(
            {"signal": "clutter", "op": "gte", "value": 0.6, "max_gap": "1m"}
        )


def test_condition_union_discriminates_by_shape() -> None:
    node = AllOf.model_validate(
        {
            "all": [
                {"signal": "a", "op": "gte", "value": 1},
                {"any": [{"not": {"signal": "b", "op": "eq", "value": True}}]},
            ]
        }
    )
    assert isinstance(node.all_[0], SignalPredicate)
    assert node.all_[0].op is Op.GTE


def test_micro_steps_shorthands() -> None:
    base = load_skill_yaml(KITCHEN_SKILL)
    explicit = Skill.model_validate(
        {
            **base.to_yaml_dict(),
            "effect": {**base.to_yaml_dict()["effect"], "micro_steps": ["cups", "mail", "wipe"]},
        }
    )
    assert explicit.effect.micro_steps.steps == ["cups", "mail", "wipe"]
    none = Skill.model_validate(
        {**base.to_yaml_dict(), "effect": {**base.to_yaml_dict()["effect"], "micro_steps": "none"}}
    )
    assert not none.effect.micro_steps.enabled


def test_ephemeral_snapshots_cannot_be_attached() -> None:
    base = load_skill_yaml(KITCHEN_SKILL).to_yaml_dict()
    base["snapshot"] = {"attach": True, "mode": "ephemeral"}
    with pytest.raises(ValidationError, match="cannot be attached"):
        Skill.model_validate(base)


def test_predicate_describe_is_human_readable() -> None:
    predicate = SignalPredicate.model_validate(
        {"signal": "burner", "op": "eq", "value": "on", "for": "10m"}
    )
    assert predicate.describe() == "burner eq 'on' for 10m"


# ---------------------------------------------------------------------------- cameras


def test_camera_requires_a_locator_for_its_kind() -> None:
    with pytest.raises(ValidationError, match="requires 'device'"):
        Camera(id="webcam", name="Webcam", kind="usb")
    assert Camera(id="webcam", name="Webcam", kind="usb", device="/dev/video0").device


def test_camera_prefers_substream_for_detection() -> None:
    cam = Camera(
        id="kitchen",
        name="Kitchen",
        url="rtsp://cam/main",
        substream_url="rtsp://cam/sub",
        password_env="KITCHEN_CAM_PASSWORD",
    )
    assert cam.detect_url == "rtsp://cam/sub"


def test_anchor_polygon_accepts_pairs_and_normalises_weights() -> None:
    anchor = Anchor.model_validate(
        {
            "id": "kitchen.counter",
            "camera_id": "kitchen",
            "label": "Kitchen counter",
            "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.6], [0.1, 0.6]],
            "clutter_weights": {"baseline_diff": 2, "object_density": 1, "semantic": 1},
        }
    )
    assert anchor.polygon[1].x == pytest.approx(0.9)
    assert anchor.clutter_weights.baseline_diff == pytest.approx(0.5)
    assert not anchor.is_full_frame


# ---------------------------------------------------------------------------- personality


def test_loud_personality_gets_boundaries_backfilled() -> None:
    goblin = Personality(id="chaos_goblin", display_name="Chaos Goblin", intensity=5)
    assert "shame_language" in [b.value for b in goblin.boundaries.never]
    assert "backlog_counts" in [b.value for b in goblin.boundaries.never]


def test_personality_steps_aside_for_safety() -> None:
    goblin = Personality(id="chaos_goblin", display_name="Chaos Goblin", intensity=4)
    assert goblin.applies_to(Urgency.LOW)
    assert not goblin.applies_to(Urgency.HIGH)
    assert not goblin.applies_to(Urgency.CRITICAL)


def test_personality_clamped_by_ceiling() -> None:
    goblin = Personality(id="chaos_goblin", display_name="Chaos Goblin", intensity=5)
    assert goblin.clamped(3).intensity == 3
    assert goblin.clamped(5).intensity == 5


def test_templates_only_personality_never_applies() -> None:
    """The `templates_only` flag (internal fallback only) steps aside for every urgency."""
    fallback = Personality(id="fallback", display_name="Fallback", templates_only=True)
    assert not fallback.applies_to(Urgency.INFO)


# ---------------------------------------------------------------------------- goals


def test_goal_up_direction() -> None:
    goal = Goal(id="cook-more", label="Cook more", metric="cook_sessions_per_week", target=4)
    assert goal.evaluate(5).status is GoalStatus.ACHIEVED
    assert goal.evaluate(3).status is GoalStatus.ON_TRACK
    assert goal.evaluate(1).status is GoalStatus.BEHIND


def test_goal_down_direction() -> None:
    goal = Goal(
        id="less-tv",
        label="Watch less TV",
        metric="tv_on_minutes_per_day",
        target=90,
        direction=GoalDirection.DOWN,
    )
    assert goal.evaluate(60).status is GoalStatus.ACHIEVED
    assert goal.evaluate(120).status is GoalStatus.ON_TRACK
    assert goal.evaluate(400).status is GoalStatus.BEHIND


def test_goal_with_no_samples_is_learning_not_failing() -> None:
    goal = Goal(id="cook-more", label="Cook more", metric="cook_sessions_per_week", target=4)
    assert goal.evaluate(0, samples=0).status is GoalStatus.LEARNING
