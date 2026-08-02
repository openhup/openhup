"""Win moments (ADR-015): the assistant noticing when a place stays clean.

Pure pass first - episodes in, at most one forward-facing claim out - then the executor hook:
dedupe by the ledger, quiet hours suppress the spoken note but never the milestone.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from openhup_schemas import EventType, Skill, TimeWindow
from sqlalchemy import select

from openhup.bus import Bus
from openhup.db import create_all, dispose, init_engine, session_scope
from openhup.db.models import EpisodeRow, TaskRow, WinMilestoneRow
from openhup.llm import PersonalityRenderer
from openhup.llm.render import PLAIN
from openhup.notify import Dispatcher, build_channels
from openhup.tasks import Executor
from openhup.wins import clear_stretch, win_candidates

BASE = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

CLUTTER_SKILL = {
    "id": "kitchen-clutter-buster",
    "enabled": True,
    "description": "Keep the counter clear.",
    "watch": [{"anchor": "kitchen.counter"}],
    "signals": [
        {
            "id": "clutter",
            "detector": "clutter_score",
            "signal": "clutter_level",
            "params": {"reference": "none"},
        }
    ],
    "conditions": {"signal": "clutter", "op": "gte", "value": 0.6, "for": "5m"},
    "effect": {"type": "task", "title_hint": "clear the kitchen counter", "urgency": "low"},
    "resolve": {
        "conditions": {"signal": "clutter", "op": "lte", "value": 0.25, "for": "1m"},
        "grace": "0s",
    },
    "limits": {"cooldown": "30m", "max_per_day": 4},
}


def episode(uid: int, opened: datetime, *, closed: datetime | None = None) -> EpisodeRow:
    return EpisodeRow(
        id=f"01K3XQ8V4W7YB2M9C6NZ0PR{uid:02d}X",
        skill_id="kitchen-clutter-buster",
        anchor_id="kitchen.counter",
        opened_at=opened,
        closed_at=closed,
    )


def chain(*gaps_days: float) -> list[EpisodeRow]:
    """Episodes with the given consecutive clean gaps, newest last.

    gaps_days[-1] is the *current* stretch (the one being celebrated); the earlier entries are
    its history. The newest episode opens at BASE and every episode closes two hours after it
    opens, so episode[i].opened_at - episode[i-1].closed_at == gaps_days[i-1].
    """
    episodes: list[EpisodeRow] = []
    opened = BASE
    for uid in range(len(gaps_days), -1, -1):
        episodes.append(episode(uid, opened, closed=opened + timedelta(hours=2)))
        if uid > 0:
            opened = opened - timedelta(days=gaps_days[uid - 1]) - timedelta(hours=2)
    return list(reversed(episodes))


# ------------------------------------------------------------------ pure pass


def test_no_win_without_two_episodes() -> None:
    assert clear_stretch([episode(0, BASE)]) is None
    assert win_candidates([episode(0, BASE)], label="Kitchen counter", now=BASE) == []


def test_no_win_when_the_previous_episode_never_closed() -> None:
    """Absence of data never resolves anything: a half-recorded cycle is not a win."""
    open_episode = episode(0, BASE - timedelta(days=2), closed=None)
    current = episode(1, BASE)
    assert clear_stretch([open_episode, current]) is None
    assert win_candidates([open_episode, current], label="Kitchen counter", now=BASE) == []


def test_clear_stretch_is_the_gap_between_episodes() -> None:
    previous = episode(0, BASE - timedelta(days=3), closed=BASE - timedelta(days=2))
    current = episode(1, BASE)
    assert clear_stretch([previous, current]) == pytest.approx(2.0)


def test_sub_one_day_clear_is_life_not_a_win() -> None:
    previous = episode(0, BASE - timedelta(days=1), closed=BASE - timedelta(hours=20))
    current = episode(1, BASE)
    assert win_candidates([previous, current], label="Kitchen counter", now=BASE) == []


@pytest.mark.parametrize(
    ("gap", "expected_band"),
    [(1.2, 1), (3.4, 3), (6.9, 3), (12.0, 7), (18.0, 14), (31.0, 30)],
)
def test_band_milestones_celebrate_the_highest_crossed(gap, expected_band) -> None:
    # A prior stretch just under the current one, so this is a band milestone, not a record.
    wins = win_candidates(chain(gap - 0.4, gap), label="Kitchen counter", now=BASE)
    assert len(wins) == 1
    win = wins[0]
    assert win.kind == "clear_days"
    assert win.value == expected_band
    assert not win.record
    assert win.days == pytest.approx(gap)


def test_a_record_beats_the_milestone_claim() -> None:
    """A 5-day stretch after a lifetime of 2-day cycles is 'longest yet', not just '3 days'."""
    wins = win_candidates(chain(2.0, 2.0, 5.2), label="Kitchen counter", now=BASE)
    assert len(wins) == 1
    assert wins[0].kind == "record_clear_days"
    assert wins[0].record is True
    assert "longest clear stretch" in wins[0].summary


def test_no_record_without_beating_the_best_by_a_margin() -> None:
    """A tied or barely-longer stretch is not a new record - float noise must not re-celebrate."""
    wins = win_candidates(chain(5.0, 5.2), label="Kitchen counter", now=BASE)
    assert wins and wins[0].kind == "clear_days"
    assert wins[0].value == 3  # the 3-day band (bands are 1/3/7/14/30), not a record claim


def test_record_beats_the_margin_again_after_a_new_best() -> None:
    """A genuinely longer stretch re-celebrates: each new 90-day best earns its own row."""
    wins = win_candidates(chain(2.0, 5.0, 5.2), label="Kitchen counter", now=BASE)
    assert len(wins) == 1
    # Best prior is 5.0; 5.2 does not clear 5.5, so no record. Push it further and it does.
    wins = win_candidates(chain(2.0, 5.0, 6.1), label="Kitchen counter", now=BASE)
    assert wins[0].kind == "record_clear_days"
    assert wins[0].value == 6.1


def test_first_stretch_is_a_record_because_there_is_no_benchmark() -> None:
    """A single gap has no history to compare against, so it is the longest by definition."""
    wins = win_candidates(chain(3.0), label="Kitchen counter", now=BASE)
    assert len(wins) == 1
    assert wins[0].kind == "record_clear_days"
    assert wins[0].value == 3.0


def test_old_stretches_do_not_count_as_record_benchmarks() -> None:
    """A 6-day stretch from 200 days ago is history, not the bar the household lives with."""
    previous = episode(0, BASE - timedelta(days=200), closed=BASE - timedelta(days=194))
    current = episode(1, BASE)
    wins = win_candidates([previous, current], label="Kitchen counter", now=BASE)
    assert len(wins) == 1
    assert wins[0].kind == "record_clear_days"  # nothing recent to compare against


def test_summaries_are_forward_facing_only() -> None:
    """The only claim shape is 'stayed clear N days'. No code path can produce a backward one."""
    for gap in (1.2, 3.0, 7.0, 12.0, 30.0):
        for win in win_candidates(chain(gap - 0.4, gap), label="Kitchen counter", now=BASE):
            assert "left" not in win.summary.lower()
            assert "missed" not in win.summary.lower()
            assert "overdue" not in win.summary.lower()
            assert "unfinished" not in win.summary.lower()


# ------------------------------------------------------------------ executor hook


@pytest.fixture
async def db(tmp_path):
    from openhup.core.config import Settings

    config = Settings(
        state_dir=str(tmp_path),
        config_dir=str(tmp_path / "config"),
        database={"url": "sqlite+aiosqlite:///" + str(tmp_path / "test.db")},
        bus={"url": "redis://127.0.0.1:6399/0"},
        llm={"provider": "echo"},
        notify={"channels": {}},
    )
    init_engine(config.database)
    await create_all()
    try:
        yield config
    finally:
        await dispose()


def quiet_hours() -> TimeWindow:
    # tz pinned to UTC so the window means the same thing on every test machine.
    return TimeWindow(between=["00:00", "06:00"], tz="UTC")


async def test_resolution_celebrates_and_dedupes(db) -> None:
    received: list = []
    bus = Bus(url=db.bus.url)
    await bus.connect()
    bus.subscribe_local(received.append)
    skill = Skill.model_validate(CLUTTER_SKILL)

    async with session_scope() as session:
        episodes = chain(1.0, 3.0)
        for row in episodes:
            session.add(row)
        row = TaskRow(
            id="01K3XQ8V4W7YB2M9C6NZ0PRSTA",
            skill_id="kitchen-clutter-buster",
            anchor_id="kitchen.counter",
            episode_id=episodes[-1].id,
            text="Clear the counter.",
            plain_text="Clear the counter.",
        )
        session.add(row)

        executor = Executor(
            session=session,
            renderer=PersonalityRenderer(None, personalities={"plain": PLAIN}),
            bus=bus,
            dispatcher=Dispatcher(channels=build_channels({})),
        )
        # Resolve the same task twice: the milestone must be claimed exactly once.
        await executor.complete_manually(row.id, skill=skill, now=BASE + timedelta(days=1))
        await executor.complete_manually(row.id, skill=skill, now=BASE + timedelta(days=1))
        await session.flush()

        milestones = (await session.execute(select(WinMilestoneRow))).scalars().all()
        assert len(milestones) == 1
        assert milestones[0].kind == "record_clear_days"
        assert milestones[0].spoken_at is not None
    await bus.close()

    win_notes = [e for e in received if e.type is EventType.WIN_NOTE]
    assert len(win_notes) == 1
    assert "stayed clear" in win_notes[0].payload["text"]


async def test_quiet_hours_suppress_the_note_but_not_the_milestone(db) -> None:
    received: list = []
    bus = Bus(url=db.bus.url)
    await bus.connect()
    bus.subscribe_local(received.append)
    skill = Skill.model_validate(CLUTTER_SKILL)
    skill.limits.quiet_hours = quiet_hours()

    async with session_scope() as session:
        episodes = chain(1.0, 3.0)
        for row in episodes:
            session.add(row)
        row = TaskRow(
            id="01K3XQ8V4W7YB2M9C6NZ0PRSTB",
            skill_id="kitchen-clutter-buster",
            anchor_id="kitchen.counter",
            episode_id=episodes[-1].id,
            text="Clear the counter.",
            plain_text="Clear the counter.",
        )
        session.add(row)
        executor = Executor(
            session=session,
            renderer=PersonalityRenderer(None, personalities={"plain": PLAIN}),
            bus=bus,
            dispatcher=Dispatcher(channels=build_channels({})),
        )
        at_3am = BASE.replace(hour=3) + timedelta(days=1)
        await executor.complete_manually(row.id, skill=skill, now=at_3am)
        await session.flush()

        milestones = (await session.execute(select(WinMilestoneRow))).scalars().all()
        assert len(milestones) == 1
        assert milestones[0].spoken_at is None
    await bus.close()

    assert [e for e in received if e.type is EventType.WIN_NOTE] == []


async def test_quiet_hours_are_a_window_not_an_off_switch(db) -> None:
    received: list = []
    bus = Bus(url=db.bus.url)
    await bus.connect()
    bus.subscribe_local(received.append)
    skill = Skill.model_validate(CLUTTER_SKILL)
    skill.limits.quiet_hours = quiet_hours()

    async with session_scope() as session:
        episodes = chain(1.0, 3.0)
        for row in episodes:
            session.add(row)
        row = TaskRow(
            id="01K3XQ8V4W7YB2M9C6NZ0PRSTC",
            skill_id="kitchen-clutter-buster",
            anchor_id="kitchen.counter",
            episode_id=episodes[-1].id,
            text="Clear the counter.",
            plain_text="Clear the counter.",
        )
        session.add(row)
        executor = Executor(
            session=session,
            renderer=PersonalityRenderer(None, personalities={"plain": PLAIN}),
            bus=bus,
            dispatcher=Dispatcher(channels=build_channels({})),
        )
        await executor.complete_manually(
            row.id, skill=skill, now=BASE.replace(hour=14) + timedelta(days=1)
        )
        await session.flush()
    await bus.close()

    assert len([e for e in received if e.type is EventType.WIN_NOTE]) == 1


def test_example_skill_shape_is_valid() -> None:
    skill = Skill.model_validate(CLUTTER_SKILL)
    assert skill.effective_personality is None
