"""Learned patterns: deterministic derivation from the household's own episodes.

This is the "learned memory" half of the memory feature (ADR-013), and it is deliberately a pure
module: episodes in, candidate claims out, no I/O. The engine and the API both call the same
discovery, so what the review screen shows and what the voice recalls are the same claims.

Three rules, all enforced here rather than requested:

* **Minimum sample before claiming anything.** A cadence needs at least four episodes spanning ten
  days, and a cadence shorter than a day or longer than a month is everyday noise, not a pattern.
  One episode is not a pattern and never will be treated as one.
* **Never per-person.** Patterns are keyed by (skill, anchor) - a place and a rule, never a person.
* **Forward language only.** A claim is "usually needs attention about every 3 days", never "you've
  left it for 6 days". There is no code path that produces a backwards-facing claim, so the safety
  filter never has to catch one.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any

from ..db import EpisodeRow

UTC = UTC

#: At least this many episodes before anything is claimed. Four is the floor for a median to mean
#: anything at all.
MIN_EPISODES = 4
#: And they must span at least this long - four episodes in one frantic day is not a habit.
MIN_SPAN = timedelta(days=10)
#: Only the last this much history is considered. Episodes are kept for 400 days; patterns should
#: describe the recent household, not the distant one.
LOOKBACK = timedelta(days=90)
#: A cadence shorter than this is everyday activity (cooking, the kettle), not a replenishment
#: cycle worth predicting. Longer than this and the claim is too vague to act on.
MIN_INTERVAL = timedelta(hours=30)
MAX_INTERVAL = timedelta(days=30)
#: The predicted window around the median interval during which a nudge is legitimate.
NUDGE_WINDOW = (0.75, 1.15)
#: Below this confidence a pattern is shown in the review screen but never nudged.
NUDGE_MIN_CONFIDENCE = 0.65

#: Hour buckets for the time-of-day pattern, newest start first. Night wraps the calendar day,
#: so it is the fall-through rather than a bucket that could swallow everything.
_DAYPARTS = ((18, "evening"), (12, "afternoon"), (6, "morning"))


@dataclass(frozen=True, slots=True)
class PatternCandidate:
    """One derived claim, ready to be reviewed, recalled, or nudged."""

    kind: str  # cadence | time_of_day
    skill_id: str
    anchor_id: str
    summary: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)


def discover_patterns(
    episodes: Sequence[EpisodeRow],
    *,
    now: datetime,
    labels: dict[str, str] | None = None,
) -> list[PatternCandidate]:
    """Derive pattern candidates from episodes, most confident first.

    `labels` maps anchor_id to a display name ("Kitchen counter") for the summary. Episodes are
    grouped by (skill, anchor); each group is considered independently, so two skills on one anchor
    produce separate claims about separate rules.
    """
    labels = labels or {}
    since = now - LOOKBACK
    # SQLite hands back naive datetimes for a timezone column; normalise once here so the rest of
    # the module never sees a naive/aware comparison.
    groups: dict[tuple[str, str], list[tuple[EpisodeRow, datetime]]] = defaultdict(list)
    for episode in episodes:
        opened = episode.opened_at
        if opened is None:
            continue
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=UTC)
        if opened >= since:
            groups[(episode.skill_id, episode.anchor_id)].append((episode, opened))

    candidates: list[PatternCandidate] = []
    for (skill_id, anchor_id), rows in groups.items():
        rows.sort(key=lambda pair: pair[1])
        subject = labels.get(anchor_id) or anchor_id
        cadence = _cadence(skill_id, anchor_id, subject, rows)
        if cadence is not None:
            candidates.append(cadence)
        daypart = _daypart(skill_id, anchor_id, subject, rows)
        if daypart is not None:
            candidates.append(daypart)
    return sorted(candidates, key=lambda c: c.confidence, reverse=True)


def _cadence(
    skill_id: str, anchor_id: str, subject: str, rows: list[tuple[EpisodeRow, datetime]]
) -> PatternCandidate | None:
    if len(rows) < MIN_EPISODES:
        return None
    first, last = rows[0][1], rows[-1][1]
    if last - first < MIN_SPAN:
        return None

    intervals = [
        (later[1] - earlier[1]).total_seconds() / 3600.0 for earlier, later in pairwise(rows)
    ]
    if not intervals:
        return None
    median_h = statistics.median(intervals)
    min_h = MIN_INTERVAL.total_seconds() / 3600.0
    max_h = MAX_INTERVAL.total_seconds() / 3600.0
    if not min_h <= median_h <= max_h:
        return None

    mean_h = statistics.fmean(intervals)
    cv = (statistics.stdev(intervals) / mean_h) if mean_h else 1.0
    days = median_h / 24.0
    day_word = "day" if round(days, 1) == 1.0 else "days"

    return PatternCandidate(
        kind="cadence",
        skill_id=skill_id,
        anchor_id=anchor_id,
        summary=(
            f"The {subject} usually needs attention about every {_round_days(days)} {day_word}."
        ),
        confidence=_confidence(len(rows), cv),
        evidence={
            "n_episodes": len(rows),
            "median_interval_h": round(median_h, 1),
            "mean_interval_h": round(mean_h, 1),
            "span_days": round((last - first).total_seconds() / 86400.0, 1),
            "last_episode_at": last.isoformat(),
            "last_episode_id": rows[-1][0].id,
            "window_days": LOOKBACK.days,
        },
    )


def _daypart(
    skill_id: str, anchor_id: str, subject: str, rows: list[tuple[EpisodeRow, datetime]]
) -> PatternCandidate | None:
    if len(rows) < MIN_EPISODES:
        return None
    counts: dict[str, int] = defaultdict(int)
    for _episode, opened in rows:
        counts[_bucket(opened.hour)] += 1
    peak, peak_count = max(counts.items(), key=lambda item: item[1])
    ratio = peak_count / len(rows)
    if ratio < 0.5:
        return None  # spread through the day; there is no "usual" time

    return PatternCandidate(
        kind="time_of_day",
        skill_id=skill_id,
        anchor_id=anchor_id,
        summary=f"The {subject} usually needs attention in the {peak}.",
        confidence=_confidence(len(rows), 0.5),
        evidence={
            "n_episodes": len(rows),
            "peak_daypart": peak,
            "peak_ratio": round(ratio, 2),
        },
    )


def _bucket(hour: int) -> str:
    for start, name in _DAYPARTS:
        if hour >= start:
            return name
    return "night"  # hour < 6


def _round_days(days: float) -> str:
    rounded = round(days, 1)
    if abs(rounded - round(rounded)) < 0.05:
        return str(round(rounded))
    return f"{rounded}"


def _confidence(n: int, cv: float) -> float:
    """More episodes and less spread mean a stronger claim. Clamped to a sane band."""
    confidence = 0.55 + 0.08 * (n - MIN_EPISODES)
    if cv <= 0.3:
        confidence += 0.1
    elif cv > 0.75:
        confidence -= 0.15
    return round(min(0.95, max(0.4, confidence)), 2)


__all__ = [
    "LOOKBACK",
    "MAX_INTERVAL",
    "MIN_EPISODES",
    "MIN_INTERVAL",
    "MIN_SPAN",
    "NUDGE_MIN_CONFIDENCE",
    "NUDGE_WINDOW",
    "PatternCandidate",
    "discover_patterns",
]
