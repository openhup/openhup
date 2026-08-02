"""Condition evaluation. The one function in OpenHup that must never be wrong.

`evaluate()` is pure: no I/O, no database, no clock. `now` is a parameter. That is not stylistic
purity - it is what makes the whole system testable against synthetic histories, and what makes
`POST /skills/{id}/simulate` possible by feeding it stored observations instead of live ones.

It returns a `Verdict`, not a bool. Every node's outcome is recorded, so the UI can show *why* a
task appeared, notifications can carry the facts, and a user tuning a threshold can see which
clause failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from openhup_schemas import (
    AllOf,
    AnyOf,
    Condition,
    NotOf,
    Op,
    SignalPredicate,
    TimeWindowCondition,
)

from . import operators as ops
from .window import BindingWindows, Sample


@dataclass(frozen=True, slots=True)
class Reason:
    """One node's contribution to the outcome, in words a person can read."""

    text: str
    satisfied: bool
    kind: str  # predicate | time_window | all | any | not
    #: Latest observed value, for predicates. Shown in the UI next to the threshold.
    observed: object | None = None
    depth: int = 0

    def __str__(self) -> str:
        mark = "✓" if self.satisfied else "✗"
        observed = f" (now {self.observed!r})" if self.observed is not None else ""
        return f"{'  ' * self.depth}{mark} {self.text}{observed}"


@dataclass(frozen=True, slots=True)
class Verdict:
    """The result of evaluating one condition tree at one instant."""

    matched: bool
    reasons: tuple[Reason, ...] = ()
    #: Bindings the tree referenced that have no window or no samples at all.
    missing: tuple[str, ...] = ()
    #: Bindings whose newest sample is older than the staleness timeout.
    stale: tuple[str, ...] = ()

    @property
    def is_evaluable(self) -> bool:
        """False when the data needed to judge this tree is absent or stale.

        The engine uses this to enter the STALE phase instead of quietly concluding that all is
        well: a dead camera must not be indistinguishable from a tidy house.
        """
        return not self.missing and not self.stale

    def facts(self) -> list[str]:
        """Satisfied predicate descriptions - the body of an alert notification."""
        return [r.text for r in self.reasons if r.satisfied and r.kind == "predicate"]

    def failures(self) -> list[str]:
        return [r.text for r in self.reasons if not r.satisfied and r.kind == "predicate"]

    def explain(self) -> str:
        return "\n".join(str(r) for r in self.reasons)


@dataclass
class _Context:
    windows: BindingWindows
    now: datetime
    staleness_timeout: timedelta | None
    reasons: list[Reason] = field(default_factory=list)
    missing: set[str] = field(default_factory=set)
    stale: set[str] = field(default_factory=set)


def evaluate(
    condition: Condition,
    windows: BindingWindows,
    now: datetime,
    *,
    staleness_timeout: timedelta | None = None,
) -> Verdict:
    """Evaluate a condition tree against signal history.

    Args:
        condition: the tree, from `skill.conditions` or `skill.resolve.conditions`.
        windows: signal history addressed by the skill's local binding names.
        now: evaluation instant. Injected so tick-driven evaluation and replay behave identically.
        staleness_timeout: how old the newest sample may be before a binding counts as stale.
            Reported on the verdict rather than changing the boolean outcome; the caller decides
            what stale data means for its phase.
    """
    if now.tzinfo is None:
        raise ValueError("evaluate() requires a timezone-aware `now`")

    context = _Context(windows=windows, now=now, staleness_timeout=staleness_timeout)
    matched = _eval(condition, context, depth=0)
    return Verdict(
        matched=matched,
        reasons=tuple(context.reasons),
        missing=tuple(sorted(context.missing)),
        stale=tuple(sorted(context.stale)),
    )


def _eval(node: Condition, ctx: _Context, depth: int) -> bool:
    if isinstance(node, AllOf):
        # Evaluate every child even after one fails: partial explanations are the point.
        results = [_eval(child, ctx, depth + 1) for child in node.all_]
        outcome = all(results)
        ctx.reasons.insert(
            _insertion_point(ctx, depth),
            Reason(f"all of {len(results)} conditions", outcome, "all", depth=depth),
        )
        return outcome

    if isinstance(node, AnyOf):
        results = [_eval(child, ctx, depth + 1) for child in node.any_]
        outcome = any(results)
        ctx.reasons.insert(
            _insertion_point(ctx, depth),
            Reason(f"any of {len(results)} conditions", outcome, "any", depth=depth),
        )
        return outcome

    if isinstance(node, NotOf):
        inner = _eval(node.not_, ctx, depth + 1)
        ctx.reasons.insert(
            _insertion_point(ctx, depth), Reason("not", not inner, "not", depth=depth)
        )
        return not inner

    if isinstance(node, TimeWindowCondition):
        outcome = node.time_window.contains(ctx.now)
        ctx.reasons.append(Reason(node.describe(), outcome, "time_window", depth=depth))
        return outcome

    if isinstance(node, SignalPredicate):
        return _eval_predicate(node, ctx, depth)

    raise TypeError(f"unknown condition node {type(node).__name__}")  # pragma: no cover


def _insertion_point(ctx: _Context, depth: int) -> int:
    """Place a container's summary before the children it summarises."""
    for index in range(len(ctx.reasons) - 1, -1, -1):
        if ctx.reasons[index].depth <= depth:
            return index + 1
    return 0


def _eval_predicate(node: SignalPredicate, ctx: _Context, depth: int) -> bool:
    window = ctx.windows.get(node.signal)
    raw: list[Sample] = window.all() if window is not None else []
    samples = ops.usable(raw, ctx.now, node.min_confidence)

    if not samples:
        ctx.missing.add(node.signal)
    elif ctx.staleness_timeout is not None:
        age = ctx.now - samples[-1].ts
        if age > ctx.staleness_timeout:
            ctx.stale.add(node.signal)

    matcher = ops.make_matcher(node.op, node.value)
    observed = samples[-1].value if samples else None

    if node.op is Op.CHANGED_TO:
        outcome = ops.changed_to(samples, matcher, ctx.now, node.within)
    elif node.for_ is not None:
        outcome = ops.held_for(samples, matcher, node.for_, ctx.now, node.max_gap)
    elif node.within is not None:
        outcome = ops.occurred_within(samples, matcher, node.within, ctx.now)
    elif node.absent_for is not None:
        outcome = ops.absent_for(samples, matcher, node.absent_for, ctx.now)
    elif node.count_over is not None:
        outcome = ops.count_over(
            samples, matcher, node.count_over.window, node.count_over.n, ctx.now
        )
    else:
        outcome = ops.latest_satisfies(samples, matcher)

    ctx.reasons.append(
        Reason(node.describe(), outcome, "predicate", observed=observed, depth=depth)
    )
    return outcome


def evaluate_both(
    trigger: Condition,
    resolve: Condition | None,
    windows: BindingWindows,
    now: datetime,
    *,
    staleness_timeout: timedelta | None = None,
) -> tuple[Verdict, Verdict]:
    """Evaluate the trigger and resolve trees at the same instant.

    Both must be judged against one `now`, or a skill could momentarily appear both triggered and
    resolved (or neither) purely because of the microseconds between two clock reads.
    """
    trigger_verdict = evaluate(trigger, windows, now, staleness_timeout=staleness_timeout)
    if resolve is None:
        return trigger_verdict, Verdict(matched=False)
    resolve_verdict = evaluate(resolve, windows, now, staleness_timeout=staleness_timeout)
    return trigger_verdict, resolve_verdict


__all__ = ["Reason", "Verdict", "evaluate", "evaluate_both"]
