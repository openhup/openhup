"""End-to-end replay: observations → windows → evaluate → FSM → actions.

This exercises the same loop the engine runs, so it catches the integration mistakes the unit tests
cannot: retention too short for a deep `for:`, ticks not firing, windows keyed wrongly.

It is also the test for a user-facing feature. `POST /skills/{id}/simulate` shows "this would have
fired 14 times last week" before a skill is ever armed.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from openhup_schemas import (
    DetectorInfo,
    Observation,
    ObservationSource,
    Signal,
    SignalKind,
    load_skill_yaml,
)

from openhup.skills.compile import compile_skill
from openhup.skills.simulate import simulate, suggest_thresholds

from .conftest import T0

CLUTTER_YAML = """
id: kitchen-clutter-buster
watch: [{anchor: kitchen.counter}]
signals:
  - {id: clutter, detector: clutter_score, signal: clutter_level}
conditions: {signal: clutter, op: gte, value: 0.6, for: 15m}
effect: {type: task, mode: single_task_focus, title_hint: clear the counter, urgency: low}
resolve:
  conditions: {signal: clutter, op: lte, value: 0.25, for: 2m}
  grace: 5m
limits: {cooldown: 45m, max_per_day: 4}
"""


def observation(
    minutes_before: float, value: float, *, anchor: str = "kitchen.counter"
) -> Observation:
    return Observation(
        ts=T0 - timedelta(minutes=minutes_before),
        source=ObservationSource(camera_id="kitchen", anchor_id=anchor, replay=True),
        detector=DetectorInfo(name="clutter_score", version="clip-vit-b32@1", backend="test"),
        signals=[Signal(key="clutter_level", kind=SignalKind.SCALAR, value=value, confidence=0.9)],
    )


def series(spans: list[tuple[float, float, float]], *, every: float = 2.0) -> list[Observation]:
    """Build observations from (start_minutes_before, end_minutes_before, value) spans."""
    out: list[Observation] = []
    for start, end, value in spans:
        cursor = start
        while cursor > end:
            out.append(observation(cursor, value))
            cursor -= every
    return out


@pytest.fixture
def clutter(anchors):
    return compile_skill(load_skill_yaml(CLUTTER_YAML), anchors=anchors)


# ------------------------------------------------------------------ the happy path


def test_mess_appears_task_is_created_then_auto_resolves(clutter) -> None:
    observations = series(
        [
            (180, 120, 0.10),  # tidy for an hour
            (120, 40, 0.80),  # then a mess for eighty minutes
            (40, 0, 0.08),  # then cleaned up
        ]
    )
    result = simulate(clutter, observations)

    assert result.tasks_created == 1
    assert result.tasks_auto_resolved == 1
    assert result.episodes == 1
    assert result.tasks_expired == 0
    # Triggered 15m after the mess began, resolved ~2m + 5m grace after it cleared.
    assert timedelta(minutes=55) < result.episode_durations[0] < timedelta(minutes=80)


def test_brief_mess_never_fires(clutter) -> None:
    """Someone put a mug down and picked it up again. `for: 15m` exists for this."""
    observations = series([(180, 60, 0.05), (60, 55, 0.85), (55, 0, 0.05)])
    result = simulate(clutter, observations)
    assert result.tasks_created == 0
    assert "Would not have fired" in result.verdict_line()


def test_mess_returning_inside_the_cooldown_is_dropped(clutter) -> None:
    """Cleaned, then messy again briefly while the 45m cooldown is still running.

    The second mess is gone before the cooldown expires, so it never becomes a task. This is the
    cooldown doing its actual job: absorbing the churn of a space in active use.
    """
    observations = series(
        [
            (300, 240, 0.05),
            (240, 200, 0.80),  # first mess -> one task
            (200, 180, 0.05),  # cleaned -> resolves, cooldown starts
            (180, 160, 0.80),  # messy again, entirely inside the cooldown
            (160, 0, 0.05),  # and cleared again before it expires
        ]
    )
    result = simulate(clutter, observations)
    assert result.tasks_created == 1
    assert result.suppressions >= 1
    assert any("cooling down" in n for n in result.notices)


def test_cooldown_delays_rather_than_cancels(clutter) -> None:
    """A mess that outlasts the cooldown does get a second task - just later.

    Suppression must not become amnesia: the counter really is still covered in stuff, and once the
    quiet period is over that deserves saying once.
    """
    observations = series(
        [
            (300, 240, 0.05),
            (240, 200, 0.80),
            (200, 180, 0.05),
            (180, 60, 0.80),  # two hours of mess, well past the 45m cooldown
            (60, 0, 0.05),
        ]
    )
    result = simulate(clutter, observations)
    assert result.tasks_created == 2
    assert result.suppressions >= 1


def test_sustained_mess_produces_exactly_one_task(clutter) -> None:
    """Four hours of clutter is one task, not one per observation."""
    observations = series([(300, 240, 0.05), (240, 0, 0.85)])
    result = simulate(clutter, observations)
    assert result.tasks_created == 1
    assert result.tasks_auto_resolved == 0  # never cleaned, so it stays open


# ------------------------------------------------------------------ data health


def test_camera_dying_mid_episode_does_not_close_the_task(clutter) -> None:
    """The mess is real, then the camera stops. The task must stay open."""
    observations = series([(300, 240, 0.05), (240, 180, 0.85)])  # nothing after t-180m
    result = simulate(clutter, observations)
    assert result.tasks_created == 1
    assert result.tasks_auto_resolved == 0


def test_gap_in_data_is_reported_as_stale(clutter) -> None:
    observations = [
        *series([(300, 260, 0.05)]),
        *series([(60, 0, 0.05)]),  # three-hour hole in the middle
    ]
    result = simulate(clutter, observations)
    assert result.stale_periods >= 1
    assert any("no fresh data" in n for n in result.notices)


# ------------------------------------------------------------------ threshold advice


def test_a_badly_tuned_skill_is_visibly_badly_tuned(anchors) -> None:
    """Trigger and resolve one twentieth apart, with a one-minute sustain, on a jittery signal.

    Compilation permits it (the ranges do not overlap) but simulation makes the consequence
    obvious: a stream of very short episodes. This is precisely the feedback that stops someone
    shipping this skill into their own kitchen.
    """
    twitchy = load_skill_yaml(
        CLUTTER_YAML.replace("value: 0.6, for: 15m", "value: 0.60, for: 1m")
        .replace("value: 0.25, for: 2m", "value: 0.55, for: 1m")
        .replace("grace: 5m", "grace: 0s")
        .replace("limits: {cooldown: 45m, max_per_day: 4}", "limits: {cooldown: 1m}")
    )
    compiled = compile_skill(twitchy, anchors=anchors)

    jitter = []
    for index in range(120):
        jitter.append(observation(240 - index * 2, 0.70 if index % 2 == 0 else 0.50))
    result = simulate(compiled, jitter)

    assert result.tasks_created > 5
    advice = suggest_thresholds(result, compiled)
    assert any("per day is a lot" in line or "under two minutes" in line for line in advice)


def test_suggestions_when_nothing_fires(clutter) -> None:
    result = simulate(clutter, series([(120, 0, 0.05)]))
    advice = suggest_thresholds(result, clutter)
    assert any("Nothing fired" in line for line in advice)


def test_verdict_line_summarises_for_humans(clutter) -> None:
    observations = series([(300, 240, 0.05), (240, 60, 0.85), (60, 0, 0.05)])
    line = simulate(clutter, observations).verdict_line()
    assert "Would have fired 1x" in line
    assert "typical episode" in line


# ------------------------------------------------------------------ mechanics


def test_observations_for_other_anchors_are_ignored(clutter) -> None:
    noise = series([(120, 0, 0.9)])
    for obs in noise:
        obs.source.anchor_id = "office.shelf"
    result = simulate(clutter, noise)
    assert result.observations_seen == 0
    assert result.tasks_created == 0


def test_ticks_happen_between_observations(clutter) -> None:
    """Sparse observations, deep `for:`. Only the timer tick can make this condition true."""
    sparse = [observation(m, 0.85) for m in range(240, 0, -4)]
    result = simulate(clutter, sparse, tick=timedelta(seconds=30))
    assert result.ticks > len(sparse)
    assert result.tasks_created == 1


def test_simulation_is_deterministic(clutter) -> None:
    observations = series([(300, 240, 0.05), (240, 60, 0.85), (60, 0, 0.05)])
    first = simulate(clutter, observations)
    second = simulate(clutter, observations)
    assert (first.tasks_created, first.episodes) == (second.tasks_created, second.episodes)
    assert [s.ts for s in first.steps] == [s.ts for s in second.steps]


def test_empty_input_is_not_an_error(clutter) -> None:
    result = simulate(clutter, [])
    assert result.tasks_created == 0
    assert result.observations_seen == 0
    assert isinstance(result.verdict_line(), str)


def test_out_of_order_observations_are_sorted(clutter) -> None:
    observations = series([(300, 240, 0.05), (240, 60, 0.85), (60, 0, 0.05)])
    shuffled = list(reversed(observations))
    assert simulate(clutter, shuffled).tasks_created == 1


def test_record_all_steps_captures_the_timeline(clutter) -> None:
    result = simulate(clutter, series([(60, 0, 0.85)]), record_all_steps=True)
    assert len(result.steps) > 100
    assert all(isinstance(step.ts, datetime) for step in result.steps)
