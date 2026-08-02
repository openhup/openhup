"""Metric rollups: turning episodes into numbers you can look at over a month.

Everything here derives from `episodes` — the trigger→resolve cycles the skill engine already
records. Nothing is counted twice and nothing needs a separate ingestion path, which is why
episodes are stored with a denormalised `duration_s`.

The metric worth understanding is `nag_index`: notifications sent per completed task. It is an
*anti*-metric. Rising means the thresholds are wrong and OpenHup is drifting towards being the thing
it was built to avoid. It is computed here, shown in the weekly report, and it is the number to
look at first when something feels off.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from openhup_schemas import Streak, TaskState
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import AlertRow, EpisodeRow, MetricPointRow, NotificationRow, TaskRow

log = logging.getLogger(__name__)
UTC = UTC


@dataclass(frozen=True, slots=True)
class Bucket:
    """One aggregation window."""

    start: datetime
    end: datetime

    @classmethod
    def day(cls, moment: datetime) -> Bucket:
        start = moment.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        return cls(start, start + timedelta(days=1))

    @classmethod
    def week(cls, moment: datetime) -> Bucket:
        day = cls.day(moment)
        start = day.start - timedelta(days=day.start.weekday())
        return cls(start, start + timedelta(days=7))


class Rollup:
    """Computes metric points. Idempotent: re-running a day overwrites rather than accumulates."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def run_daily(self, *, now: datetime | None = None) -> int:
        """Compute yesterday's and today's buckets. Safe to run on a timer."""
        now = now or datetime.now(tz=UTC)
        written = 0
        for offset in (1, 0):
            bucket = Bucket.day(now - timedelta(days=offset))
            written += await self.compute_bucket(bucket)
        written += await self.compute_streaks(now=now)
        return written

    async def compute_bucket(self, bucket: Bucket) -> int:
        """All per-day metrics for one bucket."""
        written = 0

        # Episodes per skill: the basis for "trash cycles per week", "cook sessions", and any
        # user-defined episode metric.
        rows = (
            await self.session.execute(
                select(
                    EpisodeRow.skill_id,
                    EpisodeRow.anchor_id,
                    func.count().label("episodes"),
                    func.coalesce(func.sum(EpisodeRow.duration_s), 0.0).label("total_s"),
                )
                .where(
                    EpisodeRow.opened_at >= bucket.start,
                    EpisodeRow.opened_at < bucket.end,
                )
                .group_by(EpisodeRow.skill_id, EpisodeRow.anchor_id)
            )
        ).all()

        for skill_id, anchor_id, episodes, total_s in rows:
            written += await self._write(
                f"{skill_id}.episodes", bucket, float(episodes), anchor_id, skill_id
            )
            written += await self._write(
                f"{skill_id}.minutes", bucket, float(total_s) / 60.0, anchor_id, skill_id
            )

        created = await self._count(TaskRow, TaskRow.created_at, bucket)
        resolved = await self._count(
            TaskRow,
            TaskRow.completed_at,
            bucket,
            TaskRow.state.in_([TaskState.RESOLVED_AUTO.value, TaskState.RESOLVED_MANUAL.value]),
        )
        auto = await self._count(
            TaskRow, TaskRow.completed_at, bucket, TaskRow.state == TaskState.RESOLVED_AUTO.value
        )
        false_positives = await self._count(
            TaskRow, TaskRow.created_at, bucket, TaskRow.false_positive.is_(True)
        )
        alerts = await self._count(AlertRow, AlertRow.created_at, bucket)
        notifications = await self._count(NotificationRow, NotificationRow.sent_at, bucket)

        written += await self._write("tasks_created", bucket, float(created))
        written += await self._write("tasks_resolved", bucket, float(resolved))
        written += await self._write("alerts_raised", bucket, float(alerts))
        written += await self._write("notifications_sent", bucket, float(notifications))

        if created:
            written += await self._write("task_completion_rate", bucket, resolved / created)
            written += await self._write("false_positive_rate", bucket, false_positives / created)
        if resolved:
            written += await self._write("task_auto_resolve_rate", bucket, auto / resolved)
            # The anti-metric. Watch this one.
            written += await self._write("nag_index", bucket, notifications / resolved)

        median = await self._median_time_to_resolve(bucket)
        if median is not None:
            written += await self._write("median_time_to_resolve_minutes", bucket, median / 60.0)
        return written

    async def compute_streaks(self, *, now: datetime | None = None) -> int:
        """Hours since the last episode, per anchor.

        This is `clean_streak_hours`. Framed forward on purpose: the number is "how long it has been
        good", and a break simply resets it. The UI never announces a broken streak, and no
        personality can mention one - `backlog_counts` is a filtered boundary.
        """
        now = now or datetime.now(tz=UTC)
        rows = (
            await self.session.execute(
                select(EpisodeRow.anchor_id, func.max(EpisodeRow.opened_at)).group_by(
                    EpisodeRow.anchor_id
                )
            )
        ).all()
        bucket = Bucket.day(now)
        written = 0
        for anchor_id, last_opened in rows:
            if last_opened is None:
                continue
            hours = (now - last_opened).total_seconds() / 3600.0
            written += await self._write("clean_streak_hours", bucket, hours, anchor_id)
        return written

    async def streaks(self) -> list[Streak]:
        """Current and best streaks per anchor, for the weekly report."""
        rows = (
            await self.session.execute(
                select(
                    MetricPointRow.anchor_id,
                    func.max(MetricPointRow.value),
                )
                .where(MetricPointRow.metric == "clean_streak_hours")
                .group_by(MetricPointRow.anchor_id)
            )
        ).all()

        out: list[Streak] = []
        for anchor_id, best_hours in rows:
            if not anchor_id:
                continue
            current = (
                await self.session.execute(
                    select(MetricPointRow.value)
                    .where(
                        MetricPointRow.metric == "clean_streak_hours",
                        MetricPointRow.anchor_id == anchor_id,
                    )
                    .order_by(MetricPointRow.ts.desc())
                    .limit(1)
                )
            ).scalar()
            out.append(
                Streak(
                    anchor_id=anchor_id,
                    metric="clean_streak_hours",
                    current_days=int((current or 0) // 24),
                    best_days=int((best_hours or 0) // 24),
                )
            )
        return out

    # -- internals ----------------------------------------------------------------------

    async def _write(
        self,
        metric: str,
        bucket: Bucket,
        value: float,
        anchor_id: str | None = None,
        skill_id: str | None = None,
    ) -> int:
        """Upsert a metric point.

        Idempotent by (metric, ts, anchor_id), so re-running a rollup corrects rather than
        duplicating - which matters because the worker recomputes yesterday on every pass in order
        to catch late episodes.
        """
        values = {
            "metric": metric,
            "ts": bucket.start,
            "value": round(value, 4),
            "anchor_id": anchor_id,
            "skill_id": skill_id,
            "bucket_s": int((bucket.end - bucket.start).total_seconds()),
            "labels": {},
        }
        dialect = self.session.bind.dialect.name if self.session.bind else "postgresql"

        if dialect == "postgresql":
            statement = pg_insert(MetricPointRow).values(**values)
            await self.session.execute(
                statement.on_conflict_do_update(
                    constraint="uq_metric_bucket",
                    set_={"value": statement.excluded.value},
                )
            )
        else:
            existing = (
                await self.session.execute(
                    select(MetricPointRow).where(
                        MetricPointRow.metric == metric,
                        MetricPointRow.ts == bucket.start,
                        MetricPointRow.anchor_id == anchor_id,
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                existing.value = values["value"]
            else:
                self.session.add(MetricPointRow(**values))
        return 1

    async def _count(self, model: type, column, bucket: Bucket, *extra) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(model)
            .where(column >= bucket.start, column < bucket.end, *extra)
        )
        return int(result.scalar() or 0)

    async def _median_time_to_resolve(self, bucket: Bucket) -> float | None:
        """Median seconds from open to resolved.

        Computed in Python over the day's rows rather than with a percentile function, because the
        row counts are small (tens per day) and this keeps the query portable to SQLite.
        """
        rows = (
            await self.session.execute(
                select(TaskRow.created_at, TaskRow.completed_at).where(
                    TaskRow.completed_at >= bucket.start,
                    TaskRow.completed_at < bucket.end,
                    TaskRow.state.in_(
                        [TaskState.RESOLVED_AUTO.value, TaskState.RESOLVED_MANUAL.value]
                    ),
                )
            )
        ).all()
        durations = sorted(
            (completed - created).total_seconds()
            for created, completed in rows
            if created and completed
        )
        if not durations:
            return None
        middle = len(durations) // 2
        if len(durations) % 2:
            return durations[middle]
        return (durations[middle - 1] + durations[middle]) / 2


def summarise(points: Sequence[MetricPointRow]) -> dict[str, float]:
    """Total, mean, and latest for a series. Used by the report and the API."""
    values = [p.value for p in points]
    if not values:
        return {"total": 0.0, "mean": 0.0, "latest": 0.0}
    return {
        "total": round(sum(values), 2),
        "mean": round(sum(values) / len(values), 2),
        "latest": round(max(points, key=lambda p: p.ts).value, 2),
    }


__all__ = ["Bucket", "Rollup", "summarise"]
