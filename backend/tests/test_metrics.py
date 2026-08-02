"""Metric rollup tests over SQLite.

The rollup is intentionally portable to the single-camera SQLite profile, so these tests exercise
The same database path contributors can use without Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from openhup_schemas import TaskState
from sqlalchemy import select

from openhup.core.config import Settings
from openhup.db import (
    EpisodeRow,
    MetricPointRow,
    NotificationRow,
    TaskRow,
    create_all,
    dispose,
    init_engine,
    session_scope,
)
from openhup.metrics import Bucket, Rollup

UTC = UTC


@pytest.fixture
async def database(tmp_path):
    config = Settings(
        state_dir=str(tmp_path),
        database={"url": "sqlite+aiosqlite:///" + str(tmp_path / "metrics.db")},
        llm={"provider": "echo"},
    )
    init_engine(config.database)
    await create_all()
    try:
        yield
    finally:
        await dispose()


async def metric_value(session, metric: str, bucket: Bucket) -> float | None:
    return (
        await session.execute(
            select(MetricPointRow.value).where(
                MetricPointRow.metric == metric,
                MetricPointRow.ts == bucket.start,
                MetricPointRow.anchor_id.is_(None),
            )
        )
    ).scalar_one_or_none()


async def test_sqlite_write_is_idempotent(database) -> None:
    bucket = Bucket.day(datetime(2026, 8, 17, 14, tzinfo=UTC))
    async with session_scope() as session:
        rollup = Rollup(session)
        assert await rollup._write("tasks_created", bucket, 1.25) == 1
        assert await rollup._write("tasks_created", bucket, 3.75) == 1

        rows = (
            (
                await session.execute(
                    select(MetricPointRow).where(
                        MetricPointRow.metric == "tasks_created",
                        MetricPointRow.ts == bucket.start,
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].value == pytest.approx(3.75)


async def test_compute_bucket_writes_nag_index_and_resolution_metrics(database) -> None:
    bucket = Bucket.day(datetime(2026, 8, 17, 14, tzinfo=UTC))
    async with session_scope() as session:
        session.add(
            EpisodeRow(
                id="01K3XQ8V4W7YB2M9C6NZ0PRSEA",
                skill_id="counter-skill",
                anchor_id="kitchen.counter",
                opened_at=bucket.start + timedelta(hours=1),
                duration_s=120,
                trigger_reasons=[],
            )
        )
        session.add(
            TaskRow(
                id="01K3XQ8V4W7YB2M9C6NZ0PRSEB",
                skill_id="counter-skill",
                anchor_id="kitchen.counter",
                episode_id="01K3XQ8V4W7YB2M9C6NZ0PRSEA",
                state=TaskState.RESOLVED_AUTO.value,
                text="Clear the counter.",
                plain_text="Clear the counter.",
                created_at=bucket.start + timedelta(hours=2),
                completed_at=bucket.start + timedelta(hours=2, minutes=1),
            )
        )
        for index in range(2):
            session.add(
                NotificationRow(
                    id=f"01K3XQ8V4W7YB2M9C6NZ0PRS{index:02d}",
                    channel="test",
                    title="Counter",
                    body="Clear the counter.",
                    sent_at=bucket.start + timedelta(hours=3, minutes=index),
                )
            )

        await Rollup(session).compute_bucket(bucket)

        assert await metric_value(session, "tasks_created", bucket) == pytest.approx(1)
        assert await metric_value(session, "tasks_resolved", bucket) == pytest.approx(1)
        assert await metric_value(session, "nag_index", bucket) == pytest.approx(2)
        median = await metric_value(session, "median_time_to_resolve_minutes", bucket)
        assert median == pytest.approx(1)

        episode_minutes = (
            await session.execute(
                select(MetricPointRow.value).where(
                    MetricPointRow.metric == "counter-skill.minutes",
                    MetricPointRow.anchor_id == "kitchen.counter",
                    MetricPointRow.ts == bucket.start,
                )
            )
        ).scalar_one()
        assert episode_minutes == pytest.approx(2)


async def test_nag_index_is_absent_when_nothing_resolved(database) -> None:
    bucket = Bucket.day(datetime(2026, 8, 17, 14, tzinfo=UTC))
    async with session_scope() as session:
        session.add(
            TaskRow(
                id="01K3XQ8V4W7YB2M9C6NZ0PRSFA",
                skill_id="counter-skill",
                anchor_id="kitchen.counter",
                episode_id="01K3XQ8V4W7YB2M9C6NZ0PRSFB",
                state=TaskState.OPEN.value,
                text="Clear the counter.",
                plain_text="Clear the counter.",
                created_at=bucket.start + timedelta(hours=2),
            )
        )
        session.add(
            NotificationRow(
                id="01K3XQ8V4W7YB2M9C6NZ0PRSFC",
                channel="test",
                title="Counter",
                body="Clear the counter.",
                sent_at=bucket.start + timedelta(hours=3),
            )
        )

        await Rollup(session).compute_bucket(bucket)

        assert await metric_value(session, "nag_index", bucket) is None


async def test_median_time_to_resolve_handles_odd_even_and_empty_sets(database) -> None:
    day = Bucket.day(datetime(2026, 8, 17, 14, tzinfo=UTC))
    next_day = Bucket.day(day.start + timedelta(days=1))
    async with session_scope() as session:
        created = day.start + timedelta(hours=1)
        for index, duration in enumerate((60, 180, 300)):
            session.add(
                TaskRow(
                    id=f"01K3XQ8V4W7YB2M9C6NZ0PRSG{index:02d}",
                    skill_id="median-skill",
                    anchor_id="kitchen.counter",
                    episode_id=f"01K3XQ8V4W7YB2M9C6NZ0PRSH{index:02d}",
                    state=TaskState.RESOLVED_MANUAL.value,
                    text="Task",
                    plain_text="Task",
                    created_at=created,
                    completed_at=created + timedelta(seconds=duration),
                )
            )
        next_created = next_day.start + timedelta(hours=1)
        for index, duration in enumerate((60, 180, 300, 420)):
            session.add(
                TaskRow(
                    id=f"01K3XQ8V4W7YB2M9C6NZ0PRSI{index:02d}",
                    skill_id="median-skill",
                    anchor_id="kitchen.counter",
                    episode_id=f"01K3XQ8V4W7YB2M9C6NZ0PRSJ{index:02d}",
                    state=TaskState.RESOLVED_MANUAL.value,
                    text="Task",
                    plain_text="Task",
                    created_at=next_created,
                    completed_at=next_created + timedelta(seconds=duration),
                )
            )
        await session.flush()

        rollup = Rollup(session)
        assert await rollup._median_time_to_resolve(day) == pytest.approx(180)
        assert await rollup._median_time_to_resolve(next_day) == pytest.approx(240)
        empty_day = Bucket.day(day.start - timedelta(days=1))
        assert await rollup._median_time_to_resolve(empty_day) is None


def test_day_and_week_bucket_boundaries() -> None:
    local_moment = datetime(2026, 8, 23, 23, 30, tzinfo=timezone(timedelta(hours=5)))
    day = Bucket.day(local_moment)
    week = Bucket.week(local_moment)

    assert day.start == datetime(2026, 8, 23, tzinfo=UTC)
    assert day.end == datetime(2026, 8, 24, tzinfo=UTC)
    assert week.start == datetime(2026, 8, 17, tzinfo=UTC)
    assert week.end == datetime(2026, 8, 24, tzinfo=UTC)
