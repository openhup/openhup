"""Temporal operator semantics.

These are the tests that matter most in the whole repository. Every bug this file prevents is a bug
that would show up as a task appearing at 3am, a burner alert that never fires, or a to-do list that
flaps forty times an hour.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from openhup_schemas import Op

from openhup.skills import operators as ops
from openhup.skills.window import Sample

from .conftest import T0, at, ramp, samples, steady


@pytest.fixture
def cluttered():
    """Clutter above 0.6 for the last twenty minutes, sampled every minute."""
    return samples(steady(0.72, minutes=20))


# ------------------------------------------------------------------ matchers


@pytest.mark.parametrize(
    ("op", "target", "value", "expected"),
    [
        (Op.GTE, 0.6, 0.72, True),
        (Op.GTE, 0.6, 0.6, True),
        (Op.GTE, 0.6, 0.59, False),
        (Op.LTE, 0.25, 0.2, True),
        (Op.GT, 0.6, 0.6, False),
        (Op.LT, 1, 0, True),
        (Op.EQ, "on", "on", True),
        (Op.EQ, "on", "ON", True),
        (Op.EQ, "on", "off", False),
        (Op.NEQ, "on", "off", True),
        (Op.EQ, True, True, True),
        (Op.EQ, True, "true", True),
        (Op.EQ, 0, 0, True),
        (Op.CONTAINS, "cup", ["plate", "cup"], True),
        (Op.CONTAINS, "cup", ["plate", "bowl"], False),
        (Op.NOT_CONTAINS, "trash can", ["plate"], True),
        (Op.CONTAINS, "CUP", ["cup"], True),
    ],
)
def test_matchers(op: Op, target: object, value: object, expected: bool) -> None:
    assert ops.make_matcher(op, target)(value) is expected


def test_matcher_type_mismatch_is_false_not_an_exception() -> None:
    """A detector reporting 'unknown' for a numeric signal must not crash the engine."""
    matcher = ops.make_matcher(Op.GTE, 0.6)
    assert matcher("unknown") is False
    assert matcher(None) is False
    assert matcher(True) is False  # bools are not numbers here, deliberately


def test_numeric_matcher_rejects_non_numeric_threshold() -> None:
    with pytest.raises(ValueError, match="needs a numeric value"):
        ops.make_matcher(Op.GTE, "very messy")


def test_contains_matches_bbox_labels() -> None:
    class Box:
        label = "trash can"

    assert ops.make_matcher(Op.CONTAINS, "trash can")([Box()]) is True


# ------------------------------------------------------------------ held_for


def test_held_for_true_when_sustained(cluttered) -> None:
    matcher = ops.make_matcher(Op.GTE, 0.6)
    assert ops.held_for(cluttered, matcher, timedelta(minutes=15), T0) is True


def test_held_for_false_when_too_recent() -> None:
    history = samples(steady(0.72, minutes=5))
    matcher = ops.make_matcher(Op.GTE, 0.6)
    assert ops.held_for(history, matcher, timedelta(minutes=15), T0) is False


def test_held_for_requires_a_contiguous_run() -> None:
    """Someone walking past the camera resets the clock. This is the anti-flap core."""
    history = samples(
        [(20, 0.8), (15, 0.8), (10, 0.1), (5, 0.8), (0, 0.8)]  # brief clear at t-10
    )
    matcher = ops.make_matcher(Op.GTE, 0.6)
    # The current run only started 5 minutes ago, not 20.
    assert ops.held_for(history, matcher, timedelta(minutes=15), T0) is False
    assert ops.held_for(history, matcher, timedelta(minutes=5), T0) is True


def test_held_for_false_when_latest_sample_disagrees(cluttered) -> None:
    history = [*cluttered, Sample(ts=T0, value=0.1)]
    matcher = ops.make_matcher(Op.GTE, 0.6)
    assert ops.held_for(history, matcher, timedelta(minutes=15), T0) is False


def test_held_for_rejects_a_run_with_a_data_gap() -> None:
    """A camera that dropped out for 20 minutes has not observed a sustained condition.

    Without this, an outage is indistinguishable from a burner left on - the exact failure mode
    that would make a safety skill dangerous rather than merely wrong.
    """
    history = samples([(25, "on"), (0, "on")])  # two samples, 25 minutes apart
    matcher = ops.make_matcher(Op.EQ, "on")
    assert ops.held_for(history, matcher, timedelta(minutes=10), T0, timedelta(minutes=5)) is False
    # With a tolerance wide enough to cover the gap, it passes.
    assert ops.held_for(history, matcher, timedelta(minutes=10), T0, timedelta(minutes=30)) is True


def test_held_for_rejects_stale_tail() -> None:
    """Data stopped ten minutes ago; we cannot claim the condition holds *now*."""
    history = samples(steady(0.8, minutes=40))[:-10]  # nothing in the last ~10 minutes
    matcher = ops.make_matcher(Op.GTE, 0.6)
    assert ops.held_for(history, matcher, timedelta(minutes=15), T0, timedelta(minutes=5)) is False


def test_held_for_empty_history_is_false() -> None:
    assert ops.held_for([], ops.make_matcher(Op.GTE, 0.6), timedelta(minutes=1), T0) is False


def test_run_start_finds_the_transition() -> None:
    history = samples([(30, 0.1), (20, 0.8), (10, 0.8), (0, 0.8)])
    start = ops.run_start(history, ops.make_matcher(Op.GTE, 0.6))
    assert start == at(minutes=-20)


# ------------------------------------------------------------------ within / absent_for


def test_occurred_within() -> None:
    history = samples([(45, "on"), (40, "off"), (0, "off")])
    matcher = ops.make_matcher(Op.EQ, "on")
    assert ops.occurred_within(history, matcher, timedelta(hours=1), T0) is True
    assert ops.occurred_within(history, matcher, timedelta(minutes=30), T0) is False


def test_absent_for_true_when_nothing_matched() -> None:
    history = samples(steady(0, minutes=20))  # person_count 0 throughout
    matcher = ops.make_matcher(Op.GTE, 1)
    assert ops.absent_for(history, matcher, timedelta(minutes=9), T0) is True


def test_absent_for_false_when_something_matched_recently() -> None:
    history = samples([(20, 0), (5, 2), (0, 0)])  # someone was there 5 minutes ago
    matcher = ops.make_matcher(Op.GTE, 1)
    assert ops.absent_for(history, matcher, timedelta(minutes=9), T0) is False
    assert ops.absent_for(history, matcher, timedelta(minutes=3), T0) is True


def test_absent_for_with_no_data_is_true_by_design() -> None:
    """Documented semantics: absence of evidence satisfies absent_for.

    That is only safe because `evaluate()` reports the missing signal separately and the FSM
    refuses to act on an unevaluable verdict.
    """
    assert ops.absent_for([], ops.make_matcher(Op.GTE, 1), timedelta(minutes=9), T0) is True


# ------------------------------------------------------------------ edges and counting


def test_rising_edges_counts_transitions_not_samples() -> None:
    history = samples(ramp([False, True, True, True, False, True, True], every=2))
    edges = ops.rising_edges(history, ops.make_matcher(Op.EQ, True))
    assert len(edges) == 2


def test_count_over_needs_separate_occurrences() -> None:
    """Three separate emptyings, not three consecutive samples of 'empty'."""
    matcher = ops.make_matcher(Op.LTE, 0.05)
    sustained = samples(steady(0.0, minutes=30))
    assert ops.count_over(sustained, matcher, timedelta(hours=12), 3, T0) is False

    thrice = samples(ramp([0.0, 0.9, 0.0, 0.9, 0.0, 0.9], every=60))
    assert ops.count_over(thrice, matcher, timedelta(hours=12), 3, T0) is True


def test_count_over_respects_the_window() -> None:
    matcher = ops.make_matcher(Op.EQ, True)
    history = samples(ramp([True, False, True, False, True, False], every=120))
    assert ops.count_over(history, matcher, timedelta(days=1), 3, T0) is True
    assert ops.count_over(history, matcher, timedelta(hours=3), 3, T0) is False


def test_changed_to_needs_an_actual_change() -> None:
    matcher = ops.make_matcher(Op.EQ, "open")
    just_changed = samples([(10, "closed"), (0, "open")])
    already_open = samples([(10, "open"), (0, "open")])
    assert ops.changed_to(just_changed, matcher, T0) is True
    assert ops.changed_to(already_open, matcher, T0) is False


def test_changed_to_with_window_looks_back() -> None:
    matcher = ops.make_matcher(Op.EQ, "open")
    history = samples([(30, "closed"), (20, "open"), (0, "open")])
    assert ops.changed_to(history, matcher, T0, timedelta(minutes=25)) is True
    assert ops.changed_to(history, matcher, T0, timedelta(minutes=10)) is False


# ------------------------------------------------------------------ integration


def test_time_true_integrates_state_duration() -> None:
    """TV on for 30 of the last 60 minutes."""
    history = samples([(60, True), (30, False), (0, False)])
    total = ops.time_true(history, ops.make_matcher(Op.EQ, True), timedelta(hours=1), T0)
    assert total == timedelta(minutes=30)


def test_time_true_counts_the_open_interval_to_now() -> None:
    history = samples([(60, False), (20, True)])
    total = ops.time_true(history, ops.make_matcher(Op.EQ, True), timedelta(hours=1), T0)
    assert total == timedelta(minutes=20)


# ------------------------------------------------------------------ confidence filtering


def test_low_confidence_samples_are_excluded_not_negated() -> None:
    """An unsure detector leaves a gap; it does not assert the opposite."""
    history = [
        Sample(ts=at(minutes=-20), value=0.8, confidence=0.9),
        Sample(ts=at(minutes=-10), value=0.1, confidence=0.2),  # unsure, must be ignored
        Sample(ts=at(minutes=0), value=0.8, confidence=0.9),
    ]
    filtered = ops.usable(history, T0, min_confidence=0.5)
    assert len(filtered) == 2
    # With the unsure sample dropped, the run is unbroken across the full 20 minutes.
    assert (
        ops.held_for(
            filtered,
            ops.make_matcher(Op.GTE, 0.6),
            timedelta(minutes=15),
            T0,
            timedelta(minutes=30),
        )
        is True
    )


def test_usable_drops_future_samples() -> None:
    history = [*samples([(5, 1)]), Sample(ts=at(minutes=5), value=99)]
    assert len(ops.usable(history, T0)) == 1
