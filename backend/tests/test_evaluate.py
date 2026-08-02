"""Condition-tree evaluation: boolean logic, explanations, and data health."""

from __future__ import annotations

from datetime import timedelta

import pytest
from openhup_schemas import AllOf, AnyOf, NotOf, SignalPredicate

from openhup.skills.evaluate import evaluate, evaluate_both

from .conftest import T0, steady, window, windows


def predicate(**kwargs) -> SignalPredicate:
    return SignalPredicate.model_validate(kwargs)


# ------------------------------------------------------------------ single predicates


def test_bare_predicate_uses_the_latest_sample() -> None:
    view = windows(clutter=window([(5, 0.2), (0, 0.8)]))
    verdict = evaluate(predicate(signal="clutter", op="gte", value=0.6), view, T0)
    assert verdict.matched
    assert verdict.reasons[0].observed == pytest.approx(0.8)


def test_predicate_with_for_needs_sustain() -> None:
    view = windows(clutter=window(steady(0.8, minutes=20)))
    sustained = predicate(signal="clutter", op="gte", value=0.6, **{"for": "15m"})
    assert evaluate(sustained, view, T0).matched

    brief = windows(clutter=window(steady(0.8, minutes=5)))
    assert not evaluate(sustained, brief, T0).matched


def test_missing_signal_is_reported_and_does_not_match() -> None:
    verdict = evaluate(predicate(signal="clutter", op="gte", value=0.6), windows(), T0)
    assert not verdict.matched
    assert verdict.missing == ("clutter",)
    assert not verdict.is_evaluable


def test_empty_window_counts_as_missing() -> None:
    verdict = evaluate(
        predicate(signal="clutter", op="gte", value=0.6), windows(clutter=window([])), T0
    )
    assert verdict.missing == ("clutter",)


def test_stale_signal_is_flagged_without_changing_the_boolean() -> None:
    """The verdict still says 'cluttered'; it also says 'and you should not trust me'."""
    view = windows(clutter=window(steady(0.8, minutes=120, every=5)[:-6]))  # nothing recent
    verdict = evaluate(
        predicate(signal="clutter", op="gte", value=0.6),
        view,
        T0,
        staleness_timeout=timedelta(minutes=15),
    )
    assert verdict.stale == ("clutter",)
    assert not verdict.is_evaluable


def test_fresh_signal_is_not_stale() -> None:
    view = windows(clutter=window(steady(0.8, minutes=20)))
    verdict = evaluate(
        predicate(signal="clutter", op="gte", value=0.6),
        view,
        T0,
        staleness_timeout=timedelta(minutes=15),
    )
    assert verdict.is_evaluable


# ------------------------------------------------------------------ boolean trees


def test_all_of_requires_every_child() -> None:
    view = windows(
        clutter=window(steady(0.8, minutes=20)),
        people=window(steady(0, minutes=20)),
    )
    tree = AllOf.model_validate(
        {
            "all": [
                {"signal": "clutter", "op": "gte", "value": 0.6, "for": "15m"},
                {"signal": "people", "op": "eq", "value": 0},
            ]
        }
    )
    assert evaluate(tree, view, T0).matched

    busy = windows(
        clutter=window(steady(0.8, minutes=20)),
        people=window(steady(2, minutes=20)),
    )
    assert not evaluate(tree, busy, T0).matched


def test_any_of_needs_one() -> None:
    view = windows(
        burner=window(steady("off", minutes=10)),
        people=window(steady(3, minutes=10)),
    )
    tree = AnyOf.model_validate(
        {
            "any": [
                {"signal": "burner", "op": "eq", "value": "on"},
                {"signal": "people", "op": "gte", "value": 1},
            ]
        }
    )
    assert evaluate(tree, view, T0).matched


def test_not_inverts() -> None:
    view = windows(screen=window(steady(True, minutes=10)))
    tree = NotOf.model_validate({"not": {"signal": "screen", "op": "eq", "value": True}})
    assert not evaluate(tree, view, T0).matched


def test_nested_tree() -> None:
    view = windows(
        burner=window(steady("on", minutes=20)),
        people=window(steady(0, minutes=20)),
        door=window(steady("closed", minutes=20)),
    )
    tree = AllOf.model_validate(
        {
            "all": [
                {"signal": "burner", "op": "eq", "value": "on", "for": "10m"},
                {
                    "any": [
                        {"signal": "people", "op": "eq", "value": 0},
                        {"signal": "door", "op": "eq", "value": "open"},
                    ]
                },
            ]
        }
    )
    verdict = evaluate(tree, view, T0)
    assert verdict.matched
    # Container summaries precede the children they summarise, so `explain()` reads as a tree.
    kinds = [r.kind for r in verdict.reasons]
    assert kinds[0] == "all"
    assert "any" in kinds


def test_time_window_gates_the_tree() -> None:
    view = windows(clutter=window(steady(0.8, minutes=20)))
    tree = AllOf.model_validate(
        {
            "all": [
                {"signal": "clutter", "op": "gte", "value": 0.6},
                {"time_window": {"between": ["07:00", "22:00"], "tz": "UTC"}},
            ]
        }
    )
    assert evaluate(tree, view, T0).matched  # T0 is 14:00 UTC

    night = AllOf.model_validate(
        {
            "all": [
                {"signal": "clutter", "op": "gte", "value": 0.6},
                {"time_window": {"between": ["22:00", "07:00"], "tz": "UTC"}},
            ]
        }
    )
    assert not evaluate(night, view, T0).matched


# ------------------------------------------------------------------ explanations


def test_verdict_facts_are_the_satisfied_predicates() -> None:
    view = windows(
        burner=window(steady("on", minutes=20)),
        people=window(steady(0, minutes=20)),
    )
    tree = AllOf.model_validate(
        {
            "all": [
                {"signal": "burner", "op": "eq", "value": "on", "for": "10m"},
                {"signal": "people", "op": "gte", "value": 1, "absent_for": "5m"},
            ]
        }
    )
    verdict = evaluate(tree, view, T0)
    facts = verdict.facts()
    assert any("burner eq 'on' for 10m" in f for f in facts)
    assert len(facts) == 2


def test_verdict_failures_name_the_unmet_clause() -> None:
    view = windows(clutter=window(steady(0.3, minutes=20)))
    verdict = evaluate(predicate(signal="clutter", op="gte", value=0.6), view, T0)
    assert verdict.failures() == ["clutter gte 0.6"]


def test_explain_is_readable() -> None:
    view = windows(clutter=window(steady(0.8, minutes=20)))
    tree = AllOf.model_validate({"all": [{"signal": "clutter", "op": "gte", "value": 0.6}]})
    text = evaluate(tree, view, T0).explain()
    assert "✓" in text
    assert "clutter gte 0.6" in text


# ------------------------------------------------------------------ evaluate_both


def test_evaluate_both_uses_one_instant() -> None:
    """Trigger and resolve must be judged at the same `now`, or a skill can appear to be in two
    states at once purely from the microseconds between two clock reads."""
    view = windows(clutter=window(steady(0.8, minutes=20)))
    trigger = AllOf.model_validate(
        {"all": [{"signal": "clutter", "op": "gte", "value": 0.6, "for": "15m"}]}
    )
    resolve = AllOf.model_validate(
        {"all": [{"signal": "clutter", "op": "lte", "value": 0.25, "for": "2m"}]}
    )
    trigger_verdict, resolve_verdict = evaluate_both(trigger, resolve, view, T0)
    assert trigger_verdict.matched
    assert not resolve_verdict.matched


def test_evaluate_both_without_resolve_tree() -> None:
    view = windows(clutter=window(steady(0.8, minutes=20)))
    trigger = predicate(signal="clutter", op="gte", value=0.6)
    _, resolve_verdict = evaluate_both(trigger, None, view, T0)
    assert not resolve_verdict.matched


def test_evaluate_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate(
            predicate(signal="clutter", op="gte", value=0.6),
            windows(),
            T0.replace(tzinfo=None),
        )


# ------------------------------------------------------------------ the flap scenario


def test_hysteresis_prevents_flapping() -> None:
    """A surface hovering at 0.5 satisfies neither the trigger (>=0.6) nor the resolve (<=0.25).

    This is the whole reason trigger and resolve are separate trees. With one threshold, this
    history would open and close a task on every sample.
    """
    hovering = windows(clutter=window(steady(0.5, minutes=30)))
    trigger = predicate(signal="clutter", op="gte", value=0.6)
    resolve = predicate(signal="clutter", op="lte", value=0.25)
    trigger_verdict, resolve_verdict = evaluate_both(trigger, resolve, hovering, T0)
    assert not trigger_verdict.matched
    assert not resolve_verdict.matched
