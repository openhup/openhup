"""Habits, goals, and the weekly report.

Metrics exist so the system can say something true about a month instead of only about right now.
The one to notice is `nag_index`: notifications sent per completed task. It is an *anti*-metric -
if it climbs, the thresholds are wrong and OpenHup is becoming the thing it was built to avoid.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import ULID, Duration, Ident, Slug, StrEnum, new_ulid, utcnow


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class MetricPoint(_Base):
    """One bucketed value. Written by the rollup worker, never by hand."""

    metric: Ident
    ts: datetime
    value: float
    anchor_id: Slug | None = None
    skill_id: Slug | None = None
    bucket: Duration = Field(default_factory=lambda: timedelta(days=1))
    #: Free-form dimensions, e.g. {"room": "kitchen"}. Kept small; this is not a metrics platform.
    labels: dict[str, str] = Field(default_factory=dict)


class GoalDirection(StrEnum):
    #: "cook more", "longer clean streaks"
    UP = "up"
    #: "watch less TV", "fewer trash overflows"
    DOWN = "down"
    #: "keep it around here"
    MAINTAIN = "maintain"


class GoalStatus(StrEnum):
    ON_TRACK = "on_track"
    BEHIND = "behind"
    ACHIEVED = "achieved"
    #: Not enough data yet. Shown as "still learning", never as failure.
    LEARNING = "learning"


class Goal(_Base):
    """A KPI target. Deliberately thin: metric, target, direction, window."""

    id: Slug
    label: str
    metric: Ident
    target: float
    direction: GoalDirection = GoalDirection.UP
    window: Duration = Field(default_factory=lambda: timedelta(days=7))
    anchor_id: Slug | None = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    #: Show progress in the weekly report even when the goal is behind. Off by default for
    #: goals the user marks private.
    include_in_report: bool = True

    def evaluate(self, actual: float, samples: int = 1) -> GoalProgress:
        """Score progress. Never returns a "failed" state - only ACHIEVED, ON_TRACK, or BEHIND."""
        if samples == 0:
            return GoalProgress(
                goal_id=self.id,
                actual=actual,
                target=self.target,
                ratio=0.0,
                status=GoalStatus.LEARNING,
            )

        if self.direction is GoalDirection.UP:
            ratio = actual / self.target if self.target else 1.0
            status = (
                GoalStatus.ACHIEVED
                if actual >= self.target
                else (GoalStatus.ON_TRACK if ratio >= 0.7 else GoalStatus.BEHIND)
            )
        elif self.direction is GoalDirection.DOWN:
            # Lower is better: ratio is how much headroom is left under the ceiling.
            ratio = (self.target / actual) if actual else 2.0
            status = (
                GoalStatus.ACHIEVED
                if actual <= self.target
                else (GoalStatus.ON_TRACK if ratio >= 0.7 else GoalStatus.BEHIND)
            )
        else:
            spread = abs(actual - self.target) / (self.target or 1.0)
            ratio = max(0.0, 1.0 - spread)
            status = (
                GoalStatus.ACHIEVED
                if spread <= 0.15
                else (GoalStatus.ON_TRACK if spread <= 0.35 else GoalStatus.BEHIND)
            )
        return GoalProgress(
            goal_id=self.id,
            actual=actual,
            target=self.target,
            ratio=round(min(ratio, 2.0), 3),
            status=status,
        )


class GoalProgress(_Base):
    goal_id: Slug
    actual: float
    target: float
    ratio: float
    status: GoalStatus
    trend: float | None = Field(
        default=None, description="Change versus the previous window, in metric units."
    )


class MetricSeries(_Base):
    """Query result: one metric, bucketed, over a range."""

    metric: Ident
    unit: str | None = None
    bucket: Duration
    points: list[MetricPoint] = Field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(p.value for p in self.points)

    @property
    def mean(self) -> float:
        return self.total / len(self.points) if self.points else 0.0

    def latest(self) -> MetricPoint | None:
        return max(self.points, key=lambda p: p.ts, default=None)


class Streak(_Base):
    """A run of consecutive good days. Framed forward, never used to shame a break."""

    anchor_id: Slug
    metric: Ident
    current_days: int = 0
    best_days: int = 0
    started_at: datetime | None = None
    #: Best-ever is kept, but the report never mentions a *broken* streak. See docs/UX.
    last_break_at: datetime | None = None


class WeeklyReport(_Base):
    """Input to, and record of, the coaching summary."""

    id: ULID = Field(default_factory=new_ulid)
    period_start: datetime
    period_end: datetime
    generated_at: datetime = Field(default_factory=utcnow)

    tasks_created: int = 0
    tasks_resolved: int = 0
    tasks_auto_resolved: int = 0
    alerts_raised: int = 0
    notifications_sent: int = 0

    goals: list[GoalProgress] = Field(default_factory=list)
    streaks: list[Streak] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    #: The LLM is allowed exactly one suggestion per report. More than one reads as a lecture.
    suggestion: str | None = None
    #: Personality-rendered narrative; `plain_summary` is always present as a fallback.
    narrative: str | None = None
    plain_summary: str = ""

    @model_validator(mode="after")
    def _check_period(self) -> Self:
        if self.period_end <= self.period_start:
            raise ValueError("weekly report period_end must be after period_start")
        return self

    @property
    def completion_rate(self) -> float:
        return self.tasks_resolved / self.tasks_created if self.tasks_created else 0.0

    @property
    def nag_index(self) -> float:
        """Notifications per completed task. Rising means the thresholds need work."""
        return self.notifications_sent / self.tasks_resolved if self.tasks_resolved else 0.0


#: Metrics the rollup worker computes out of the box. Skills can define their own via MetricEffect.
BUILTIN_METRICS: dict[str, str] = {
    "clean_streak_hours": "Hours since the last clutter episode on an anchor.",
    "trash_cycles_per_week": "Completed full→empty trash episodes per week.",
    "tv_on_minutes_per_day": "Minutes the screen_on signal was true.",
    "cook_sessions_per_week": "Stove-active episodes lasting over 8 minutes.",
    "task_completion_rate": "Resolved tasks divided by created tasks.",
    "task_auto_resolve_rate": "Share of tasks the camera closed without being asked.",
    "median_time_to_resolve_minutes": "Median open→resolved time.",
    "nag_index": "Notifications sent per completed task. Lower is better.",
    "false_positive_rate": "Share of tasks the user marked as false positives.",
}

__all__ = [
    "BUILTIN_METRICS",
    "Goal",
    "GoalDirection",
    "GoalProgress",
    "GoalStatus",
    "MetricPoint",
    "MetricSeries",
    "Streak",
    "WeeklyReport",
]
