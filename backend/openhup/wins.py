"""Win moments: the assistant noticing when a place stays clean (ADR-015).

The caring half of the personality layer. Episodes in, at most one win out, pure - no I/O, no
LLM, testable at fixed times, exactly like `memory.patterns`. The engine calls this when a task
resolves and celebrates whatever comes back through the current personality.

The only claim shape that exists is forward-facing: *how long a place has been good*. There is no
code path that can produce \"you left it for six days\", so the safety filter never has to catch
one - tests assert this.

A win is one of:

* **`record_clear_days`** - the just-ended clear stretch is the longest for this anchor in the
  trailing 90 days. Wins over the band milestone: a record line already names the length.
* **`clear_days`** - the stretch crossed a whole-day band (1, 3, 7, 14, 30) without setting a
  record. The band floor is the dedupe value, so progress is celebrated once per band.

Both are deduped in the `win_milestones` ledger by (anchor, kind, value).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

#: Whole-day bands, in days. A stretch is celebrated for the highest band it crosses.
MILESTONES: tuple[int, ...] = (1, 3, 7, 14, 30)
#: Below this, a clear gap is ordinary life, not a win worth announcing.
MIN_CLEAR_DAYS = 1.0
#: A record must beat the previous best by at least this much (days), so float noise or a tied
#: stretch cannot re-celebrate the same record.
RECORD_MARGIN_DAYS = 0.5
#: Horizons for the record comparison.
LOOKBACK_DAYS = 90


@dataclass(frozen=True, slots=True)
class Win:
    """One claim worth celebrating, from the pure pass."""

    kind: str  # clear_days | record_clear_days
    #: Dedupe key for the ledger: the band floor, or the rounded record days.
    value: float
    #: The actual clear stretch, in days.
    days: float
    record: bool
    #: Tone-free summary, ready for the review screen: \"Kitchen counter stayed clear 3 days.\"
    summary: str


def clear_stretch(episodes: Sequence[object]) -> float | None:
    """Days the anchor stayed clear before its most recent episode opened.

    The gap between the previous episode closing and the current one opening. Requires two
    consecutive episodes and a previous episode that actually closed - absence of data never
    resolves anything, and a first-ever mess has no baseline to celebrate against.
    """
    if len(episodes) < 2:
        return None
    current = episodes[-1]
    previous = episodes[-2]
    opened_at = getattr(current, "opened_at", None)
    closed_at = getattr(previous, "closed_at", None)
    if opened_at is None or closed_at is None:
        return None
    opened_at = _aware(opened_at)
    closed_at = _aware(closed_at)
    if opened_at <= closed_at:
        return None  # overlapping episodes: bad data, not a win
    return (opened_at - closed_at).total_seconds() / 86400.0


def win_candidates(episodes: Sequence[object], *, label: str, now: datetime) -> list[Win]:
    """The win worth celebrating right now, or none.

    Only the most recent episode can produce a win - everything earlier was already celebrated
    when it closed. Records compare against the trailing 90 days of *earlier* clear stretches.
    """
    stretch = clear_stretch(episodes)
    if stretch is None or stretch < MIN_CLEAR_DAYS:
        return []
    if _is_record(stretch, episodes[:-1], now):
        value = round(stretch, 1)
        return [
            Win(
                kind="record_clear_days",
                value=value,
                days=stretch,
                record=True,
                summary=(
                    f"{label} stayed clear {_days(stretch)} - its longest clear stretch in "
                    f"the last {LOOKBACK_DAYS} days."
                ),
            )
        ]
    band = _highest_band(stretch)
    if band is None:
        return []
    return [
        Win(
            kind="clear_days",
            value=float(band),
            days=stretch,
            record=False,
            summary=f"{label} stayed clear {_days(stretch)}.",
        )
    ]


def _is_record(stretch: float, earlier: Sequence[object], now: datetime) -> bool:
    """True when `stretch` beats every earlier clear stretch inside the lookback window."""
    best: float | None = None
    lookback_start = _aware(now) - timedelta(days=LOOKBACK_DAYS)
    for index, episode in enumerate(earlier):
        window = earlier[: index + 1]
        gap = clear_stretch(window)
        if gap is None:
            continue
        if _aware(getattr(episode, "opened_at", None) or now) < lookback_start:
            continue  # too old to count as a benchmark
        if best is None or gap > best:
            best = gap
    return best is None or stretch > best + RECORD_MARGIN_DAYS


def _highest_band(stretch: float) -> int | None:
    crossed = [milestone for milestone in MILESTONES if stretch >= milestone]
    return crossed[-1] if crossed else None


def _days(days: float) -> str:
    from openhup.llm.prompts import days_str  # local import: avoids a package import cycle

    return days_str(days)


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment


__all__ = [
    "LOOKBACK_DAYS",
    "MILESTONES",
    "MIN_CLEAR_DAYS",
    "RECORD_MARGIN_DAYS",
    "Win",
    "clear_stretch",
    "win_candidates",
]
