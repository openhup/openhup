"""Skill compilation: the checks that stop a parseable skill from being a bad one."""

from __future__ import annotations

import pytest
from openhup_schemas import BUILTIN_DETECTORS, load_skill_yaml

from openhup.skills.compile import SkillCompileError, compile_all, compile_skill

CLUTTER = """
id: kitchen-clutter-buster
watch:
  - anchor: kitchen.counter
signals:
  - {id: clutter, detector: clutter_score, signal: clutter_level}
conditions:
  all:
    - {signal: clutter, op: gte, value: 0.6, for: 15m}
effect:
  type: task
  title_hint: clear the kitchen counter
  urgency: low
resolve:
  conditions:
    all:
      - {signal: clutter, op: lte, value: 0.25, for: 2m}
  grace: 5m
limits: {cooldown: 45m, max_per_day: 4}
"""


def compile_yaml(text: str, anchors=None, **kwargs):
    return compile_skill(load_skill_yaml(text), anchors=anchors, **kwargs)


def codes(findings) -> set[str]:
    return {f.code for f in findings}


# ------------------------------------------------------------------ happy path


def test_compiles_and_resolves_bindings(anchors) -> None:
    compiled = compile_yaml(CLUTTER, anchors)
    assert compiled.anchor_ids == ("kitchen.counter",)
    assert compiled.binding("clutter").detector == "clutter_score"
    assert compiled.binding("clutter").spec.kind == "scalar"
    keys = compiled.signal_keys("kitchen.counter")
    assert str(keys["clutter"]) == "kitchen.counter/clutter_score.clutter_level"
    assert compiled.instances == [("kitchen-clutter-buster", "kitchen.counter")]


def test_horizon_is_the_deepest_predicate_window(anchors) -> None:
    compiled = compile_yaml(CLUTTER, anchors)
    assert compiled.horizon.total_seconds() == 15 * 60


def test_camera_wildcard_expands_to_anchors(anchors) -> None:
    skill = CLUTTER.replace("- anchor: kitchen.counter", "- camera: kitchen").replace(
        "signal: clutter_level}", "signal: clutter_level, params: {reference: none}}"
    )
    compiled = compile_yaml(skill, anchors)
    assert set(compiled.anchor_ids) == {"kitchen.counter", "kitchen.stove"}


def test_compiles_without_anchors_for_static_linting() -> None:
    """The CLI lints skill files with no database available."""
    compiled = compile_yaml(CLUTTER)
    assert compiled.anchor_ids == ("kitchen.counter",)


# ------------------------------------------------------------------ hard errors


def test_unknown_detector_is_an_error(anchors) -> None:
    broken = CLUTTER.replace("detector: clutter_score", "detector: vibe_check")
    with pytest.raises(SkillCompileError) as exc:
        compile_yaml(broken, anchors)
    assert "unknown_detector" in codes(exc.value.findings)


def test_unknown_signal_is_an_error(anchors) -> None:
    broken = CLUTTER.replace("signal: clutter_level", "signal: messiness")
    with pytest.raises(SkillCompileError) as exc:
        compile_yaml(broken, anchors)
    assert "unknown_signal" in codes(exc.value.findings)


def test_operator_kind_mismatch_is_an_error(anchors) -> None:
    """`contains` against a scalar parses fine and can never be true."""
    broken = CLUTTER.replace(
        "{signal: clutter, op: gte, value: 0.6, for: 15m}",
        "{signal: clutter, op: contains, value: cup, for: 15m}",
    )
    with pytest.raises(SkillCompileError) as exc:
        compile_yaml(broken, anchors)
    assert "operator_kind_mismatch" in codes(exc.value.findings)


def test_threshold_outside_signal_range_is_an_error(anchors) -> None:
    broken = CLUTTER.replace("value: 0.6, for: 15m", "value: 6, for: 15m")
    with pytest.raises(SkillCompileError) as exc:
        compile_yaml(broken, anchors)
    assert "threshold_out_of_range" in codes(exc.value.findings)


def test_unknown_enum_value_is_an_error(anchors) -> None:
    skill = """
id: door-watch
watch: [{anchor: living.walkway}]
signals:
  - {id: door, detector: door_state, signal: door_state}
conditions: {signal: door, op: eq, value: halfway, for: 5m}
effect: {type: alert, urgency: normal}
resolve: {conditions: {signal: door, op: eq, value: closed}}
"""
    with pytest.raises(SkillCompileError) as exc:
        compile_yaml(skill, anchors)
    assert "unknown_enum_value" in codes(exc.value.findings)


def test_missing_baseline_is_an_error(anchors) -> None:
    """clutter_score defaults to comparing against a baseline; the stove has none."""
    skill = CLUTTER.replace("- anchor: kitchen.counter", "- anchor: kitchen.stove")
    with pytest.raises(SkillCompileError) as exc:
        compile_yaml(skill, anchors)
    finding = next(f for f in exc.value.findings if f.code == "missing_baseline")
    assert "POST /api/v1/anchors/kitchen.stove/baseline" in finding.message


def test_baseline_not_needed_when_reference_is_none(anchors) -> None:
    skill = CLUTTER.replace("- anchor: kitchen.counter", "- anchor: kitchen.stove").replace(
        "signal: clutter_level}", "signal: clutter_level, params: {reference: none}}"
    )
    assert compile_yaml(skill, anchors).anchor_ids == ("kitchen.stove",)


def test_missing_required_param_is_an_error(anchors) -> None:
    skill = """
id: trash-full
watch: [{anchor: kitchen.counter}]
signals:
  - {id: fill, detector: fill_level, signal: fill_level}
conditions: {signal: fill, op: gte, value: 0.9, for: 4h}
effect: {type: task, title_hint: take out the trash}
resolve: {conditions: {signal: fill, op: lte, value: 0.2, for: 1m}}
"""
    with pytest.raises(SkillCompileError) as exc:
        compile_yaml(skill, anchors)
    assert "missing_required_param" in codes(exc.value.findings)


def test_unknown_anchor_is_an_error(anchors) -> None:
    broken = CLUTTER.replace("kitchen.counter", "kitchen.ceiling")
    with pytest.raises(SkillCompileError) as exc:
        compile_yaml(broken, anchors)
    assert "unknown_anchor" in codes(exc.value.findings)


# ------------------------------------------------------------------ the hysteresis check


def test_overlapping_thresholds_are_rejected(anchors) -> None:
    """Trigger >= 0.6, resolve <= 0.7: any value in 0.6..0.7 is both. This flaps forever."""
    flapping = CLUTTER.replace("op: lte, value: 0.25", "op: lte, value: 0.7")
    with pytest.raises(SkillCompileError) as exc:
        compile_yaml(flapping, anchors)
    finding = next(f for f in exc.value.findings if f.code == "no_hysteresis")
    assert "open and close repeatedly" in finding.message
    assert "resolve <=" in finding.message  # the message suggests a concrete fix


def test_identical_thresholds_are_a_warning(anchors) -> None:
    tight = CLUTTER.replace("op: lte, value: 0.25", "op: lte, value: 0.6")
    compiled = compile_yaml(tight, anchors)
    assert "tight_hysteresis" in codes(compiled.warnings)


def test_inverted_direction_hysteresis(anchors) -> None:
    """Trigger when a level falls (pet bowl empty), resolve when it rises again."""
    good = """
id: pet-bowl
watch: [{anchor: kitchen.counter}]
signals:
  - {id: bowl, detector: fill_level, signal: fill_level, params: {container: pet water bowl}}
conditions: {signal: bowl, op: lte, value: 0.15, for: 10m}
effect: {type: task, title_hint: refill the water bowl, urgency: normal}
resolve: {conditions: {signal: bowl, op: gte, value: 0.6, for: 1m}}
limits: {cooldown: 2h, max_per_day: 3}
"""
    assert "no_hysteresis" not in codes(compile_yaml(good, anchors).warnings)

    bad = good.replace("op: gte, value: 0.6", "op: gte, value: 0.05")
    with pytest.raises(SkillCompileError) as exc:
        compile_yaml(bad, anchors)
    assert "no_hysteresis" in codes(exc.value.findings)


def test_hysteresis_check_ignores_unrelated_signals(anchors) -> None:
    """Trigger and resolve on *different* signals is normal and must not be flagged."""
    skill = """
id: stove-safety
watch: [{anchor: kitchen.stove}]
signals:
  - id: burner
    detector: zero_shot_state
    signal: burner_state
    params: {probes: {on: a lit burner, off: an unlit burner}}
  - {id: people, detector: object_inventory, signal: person_count}
conditions:
  all:
    - {signal: burner, op: eq, value: "on", for: 10m}
    - {signal: people, op: gte, value: 1, absent_for: 5m}
effect: {type: alert, urgency: high, channels: [ntfy]}
resolve:
  conditions:
    any:
      - {signal: burner, op: eq, value: "off", for: 30s}
      - {signal: people, op: gte, value: 1}
limits: {cooldown: 5m}
"""
    compiled = compile_yaml(skill, anchors)
    assert "no_hysteresis" not in codes(compiled.warnings)
    assert "tight_hysteresis" not in codes(compiled.warnings)


# ------------------------------------------------------------------ advice


def test_instantaneous_trigger_warns(anchors) -> None:
    twitchy = CLUTTER.replace(", for: 15m}", "}")
    compiled = compile_yaml(twitchy, anchors)
    assert "instantaneous_trigger" in codes(compiled.warnings)


def test_short_cooldown_and_missing_cap_warn(anchors) -> None:
    naggy = CLUTTER.replace("limits: {cooldown: 45m, max_per_day: 4}", "limits: {cooldown: 1m}")
    warnings = codes(compile_yaml(naggy, anchors).warnings)
    assert "short_cooldown" in warnings
    assert "no_daily_cap" in warnings


def test_task_at_alert_urgency_warns(anchors) -> None:
    urgent = CLUTTER.replace("urgency: low", "urgency: high")
    assert "task_at_alert_urgency" in codes(compile_yaml(urgent, anchors).warnings)


def test_personality_bypass_is_noted_not_hidden(anchors) -> None:
    """A user setting a jokey personality on a safety alert should be told it will be ignored."""
    skill = """
id: fall-watch
watch: [{anchor: living.walkway}]
personality: chaos_goblin
signals:
  - {id: down, detector: pose_fall, signal: person_down}
conditions: {signal: down, op: eq, value: true, for: 30s}
effect: {type: alert, urgency: critical}
resolve: {conditions: {signal: down, op: eq, value: false, for: 10s}}
"""
    warnings = codes(compile_yaml(skill, anchors).warnings)
    assert "personality_will_be_bypassed" in warnings
    assert "optional_detector" in warnings  # pose_fall needs opting in


def test_metric_skill_with_snapshots_warns(anchors) -> None:
    skill = """
id: tv-time
watch: [{anchor: living.tv}]
signals:
  - {id: screen, detector: screen_on, signal: screen_on}
conditions: {signal: screen, op: eq, value: true, for: 2m}
effect:
  type: metric
  metric: tv_on_minutes_per_day
  aggregation: duration_minutes
resolve: {conditions: {signal: screen, op: eq, value: false, for: 5m}}
snapshot: {attach: true}
"""
    assert "metric_with_snapshots" in codes(compile_yaml(skill, anchors).warnings)


# ------------------------------------------------------------------ batch compilation


def test_compile_all_isolates_failures(anchors) -> None:
    """One broken skill must not stop the other nineteen from running."""
    good = load_skill_yaml(CLUTTER)
    bad = load_skill_yaml(
        CLUTTER.replace("id: kitchen-clutter-buster", "id: broken").replace(
            "detector: clutter_score", "detector: nonsense"
        )
    )
    compiled, failures = compile_all([good, bad], anchors=anchors)
    assert [c.skill.id for c in compiled] == ["kitchen-clutter-buster"]
    assert set(failures) == {"broken"}


def test_strict_false_downgrades_errors(anchors) -> None:
    broken = CLUTTER.replace("detector: clutter_score", "detector: nonsense")
    compiled = compile_yaml(broken, anchors, strict=False)
    assert compiled.bindings == ()  # nothing resolved, but no exception


def test_registry_is_injectable() -> None:
    compiled = compile_skill(load_skill_yaml(CLUTTER), registry=BUILTIN_DETECTORS)
    assert compiled.skill.id == "kitchen-clutter-buster"
