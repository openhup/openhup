"""Temporal operators over signal history. Pure functions, no clock, no I/O.

These implement the operators in the skill schema, and they are where the interesting bugs in this
class of system live. Three decisions worth stating plainly:

1. **`for:` measures a contiguous run, not a window average.** A surface that is cluttered, briefly
   clear because someone walked in front of the camera, then cluttered again has *not* been
   cluttered for 15 minutes. The run is found by walking backwards from the newest sample until the
   predicate stops holding.

2. **A gap in the data breaks a run.** `max_gap` exists so that a camera which dropped out for
   twenty minutes cannot come back and satisfy `burner on for 10m` on the strength of two samples
   twenty minutes apart. Without this, an outage looks exactly like a sustained condition.

3. **`count_over` counts rising edges, not samples.** "The bowl was empty three times today" means
   three separate emptyings. Counting samples would satisfy it in fifteen seconds at a 5-second
   sampling interval.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any

from openhup_schemas import Op

from .window import Sample

Matcher = Callable[[Any], bool]

#: Default tolerated gap inside a `for` run. Generous enough for a 2-minute detector interval,
#: tight enough that a real outage does not masquerade as a sustained condition. Override per
#: predicate with `max_gap`.
DEFAULT_MAX_GAP = timedelta(minutes=5)


# --------------------------------------------------------------------------------------
# Matchers
# --------------------------------------------------------------------------------------


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def make_matcher(op: Op, target: Any) -> Matcher:
    """Build a value predicate from an operator and its comparison target.

    Type mismatches evaluate False rather than raising: a detector that reports `unknown` for an
    enum signal must not crash the engine, it must simply fail to match. Compile-time validation
    (`compile.py`) is where genuinely wrong skills are rejected.
    """
    if op in {Op.GTE, Op.LTE, Op.GT, Op.LT}:
        bound = _as_number(target)
        if bound is None:
            raise ValueError(f"operator {op} needs a numeric value, got {target!r}")

        def numeric(value: Any) -> bool:
            number = _as_number(value)
            if number is None:
                return False
            if op is Op.GTE:
                return number >= bound
            if op is Op.LTE:
                return number <= bound
            if op is Op.GT:
                return number > bound
            return number < bound

        return numeric

    if op in {Op.EQ, Op.NEQ, Op.CHANGED_TO}:

        def equality(value: Any) -> bool:
            same = _values_equal(value, target)
            return not same if op is Op.NEQ else same

        return equality

    if op in {Op.CONTAINS, Op.NOT_CONTAINS}:
        needle = str(target).strip().lower()

        def membership(value: Any) -> bool:
            if isinstance(value, (list, tuple, set, frozenset)):
                present = any(_label_of(item).strip().lower() == needle for item in value)
            elif isinstance(value, str):
                present = needle in value.strip().lower()
            else:
                present = False
            return (not present) if op is Op.NOT_CONTAINS else present

        return membership

    raise ValueError(f"unsupported operator {op!r}")  # pragma: no cover - enum is exhaustive


def _label_of(item: Any) -> str:
    """Set signals may carry plain strings or box-like objects with a label."""
    if isinstance(item, str):
        return item
    label = getattr(item, "label", None)
    if label is None and isinstance(item, dict):
        label = item.get("label", "")
    return str(label or "")


def _values_equal(value: Any, target: Any) -> bool:
    if isinstance(value, bool) or isinstance(target, bool):
        return _coerce_bool(value) == _coerce_bool(target)
    number_a, number_b = _as_number(value), _as_number(target)
    if number_a is not None and number_b is not None:
        return abs(number_a - number_b) < 1e-9
    return str(value).strip().lower() == str(target).strip().lower()


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "on", "1"}:
            return True
        if lowered in {"false", "no", "off", "0"}:
            return False
        return None
    number = _as_number(value)
    return bool(number) if number is not None else None


# --------------------------------------------------------------------------------------
# Sample filtering
# --------------------------------------------------------------------------------------


def usable(
    samples: Sequence[Sample],
    now: datetime,
    min_confidence: float | None = None,
) -> list[Sample]:
    """Samples eligible for evaluation: not from the future, confident enough.

    Low-confidence samples are *excluded* rather than treated as non-matching. A detector saying
    "I am not sure" is not evidence that the condition stopped holding; it is an absence of
    evidence, and it leaves a gap that `max_gap` will judge.
    """
    result = [s for s in samples if s.ts <= now]
    if min_confidence is not None:
        result = [s for s in result if s.confidence is None or s.confidence >= min_confidence]
    return result


# --------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------


def latest_satisfies(samples: Sequence[Sample], matcher: Matcher) -> bool:
    """Instantaneous test against the newest sample."""
    return bool(samples) and matcher(samples[-1].value)


def run_start(samples: Sequence[Sample], matcher: Matcher) -> datetime | None:
    """Timestamp of the first sample in the trailing contiguous matching run.

    None when the newest sample does not match, i.e. there is no current run. Conservative by
    design: the true transition happened somewhere between the last non-matching sample and this
    one, and we credit the later of the two.
    """
    if not samples or not matcher(samples[-1].value):
        return None
    start = samples[-1].ts
    for sample in reversed(samples[:-1]):
        if not matcher(sample.value):
            break
        start = sample.ts
    return start


def largest_gap(samples: Sequence[Sample]) -> timedelta:
    if len(samples) < 2:
        return timedelta(0)
    return max(b.ts - a.ts for a, b in pairwise(samples))


def held_for(
    samples: Sequence[Sample],
    matcher: Matcher,
    duration: timedelta,
    now: datetime,
    max_gap: timedelta | None = None,
) -> bool:
    """Has the predicate held continuously for at least `duration`?

    Requires: the newest sample matches; the run began at least `duration` ago; no gap inside the
    run (nor between the newest sample and `now`) exceeds `max_gap`.

    The final clause is what makes this correct on a timer tick rather than only on new data - the
    condition becomes true through the passage of time, but only while data is still flowing.
    """
    start = run_start(samples, matcher)
    if start is None:
        return False
    if now - start < duration:
        return False

    gap_limit = max_gap or DEFAULT_MAX_GAP
    run = [s for s in samples if s.ts >= start]
    if largest_gap(run) > gap_limit:
        return False
    return (now - run[-1].ts) <= gap_limit


def occurred_within(
    samples: Sequence[Sample],
    matcher: Matcher,
    window: timedelta,
    now: datetime,
) -> bool:
    """Did the predicate hold at least once in the trailing window?"""
    cutoff = now - window
    return any(matcher(s.value) for s in samples if s.ts >= cutoff)


def absent_for(
    samples: Sequence[Sample],
    matcher: Matcher,
    duration: timedelta,
    now: datetime,
) -> bool:
    """Has the predicate held at *no* point in the trailing window?

    An empty window satisfies this, deliberately: "nobody has been in the kitchen for 9 minutes" is
    exactly the case where no person was detected, and demanding positive evidence of absence would
    make the operator useless. The staleness machinery in `evaluate` is what stops this from turning
    an unplugged camera into a confident claim about an empty room.
    """
    cutoff = now - duration
    return not any(matcher(s.value) for s in samples if s.ts >= cutoff)


def rising_edges(
    samples: Sequence[Sample],
    matcher: Matcher,
    window: timedelta | None = None,
    now: datetime | None = None,
) -> list[datetime]:
    """Timestamps where the predicate went from not-holding to holding.

    The first sample counts as an edge only if it matches, since we cannot see before it.
    """
    edges: list[datetime] = []
    previous_matched: bool | None = None
    for sample in samples:
        matched = matcher(sample.value)
        if matched and previous_matched is not True:
            edges.append(sample.ts)
        previous_matched = matched

    if window is not None and now is not None:
        cutoff = now - window
        edges = [ts for ts in edges if ts >= cutoff]
    return edges


def count_over(
    samples: Sequence[Sample],
    matcher: Matcher,
    window: timedelta,
    n: int,
    now: datetime,
) -> bool:
    """Did the predicate rise at least `n` times in the trailing window?"""
    return len(rising_edges(samples, matcher, window, now)) >= n


def changed_to(
    samples: Sequence[Sample],
    matcher: Matcher,
    now: datetime,
    window: timedelta | None = None,
) -> bool:
    """Transition detection.

    Without a window: the newest sample matches and the one before it did not - "it just changed".
    With a window: any such transition occurred inside it. Used for enum and boolean signals where
    the *event* matters and the steady state does not (a burner already on when the skill was
    enabled should not fire `changed_to: on`).
    """
    if window is not None:
        return bool(rising_edges(samples, matcher, window, now))
    if len(samples) < 2:
        return False
    return matcher(samples[-1].value) and not matcher(samples[-2].value)


def time_true(
    samples: Sequence[Sample],
    matcher: Matcher,
    window: timedelta,
    now: datetime,
) -> timedelta:
    """Total time the predicate held inside the window, by piecewise-constant integration.

    Backs duration metrics ("TV minutes per day", "stove active minutes"): each sample is assumed
    to describe the state until the next one arrives.
    """
    cutoff = now - window
    relevant = [s for s in samples if s.ts >= cutoff]
    if not relevant:
        return timedelta(0)

    total = timedelta(0)
    for current, following in pairwise(relevant):
        if matcher(current.value):
            total += following.ts - current.ts
    if matcher(relevant[-1].value):
        total += now - relevant[-1].ts
    return total


__all__ = [
    "DEFAULT_MAX_GAP",
    "Matcher",
    "absent_for",
    "changed_to",
    "count_over",
    "held_for",
    "largest_gap",
    "latest_satisfies",
    "make_matcher",
    "occurred_within",
    "rising_edges",
    "run_start",
    "time_true",
    "usable",
]
