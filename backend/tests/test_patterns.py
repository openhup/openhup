"""Learned memory pattern tests.

Four layers, matching the implementation:

* Discovery (`discover_patterns`) is pure: episodes in, candidate claims out. The tests that matter
  prove it claims nothing from too little data, and never produces a backwards-facing claim.
* The due-window logic decides *when* a cadence may speak, and may speak once per episode cycle.
* The API and voice surfaces expose patterns for review, recall, and dismissal.
* The engine nudge pass is the guardrail test: open task, quiet hours, dedupe, and a daily cap all
  stop the system from becoming the nagging it was built to avoid (ADR-013).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from openhup.api.main import create_app
from openhup.core.config import Settings
from openhup.db import create_all, dispose, init_engine, session_scope
from openhup.db.models import EpisodeRow, MemoryPatternRow, TaskRow
from openhup.memory import pattern_due
from openhup.memory.patterns import discover_patterns

UTC = UTC

#: Fixed reference instant, like the engine tests: a Monday afternoon.
BASE = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)

CAMERA = {"id": "kitchen", "name": "Kitchen", "kind": "rtsp", "url": "rtsp://camera.invalid/s"}
ANCHOR = {
    "id": "kitchen.counter",
    "camera_id": "kitchen",
    "label": "Kitchen counter",
    "polygon": [[0.1, 0.3], [0.9, 0.3], [0.9, 0.8], [0.1, 0.8]],
}
SKILL = {
    "id": "trash-cycle",
    "enabled": True,
    "description": "Take out the trash.",
    "watch": [{"anchor": "kitchen.counter"}],
    "signals": [
        {
            "id": "full",
            "detector": "fill_level",
            "signal": "fill_level",
            "params": {"container": "trash_bin"},
        }
    ],
    "conditions": {"signal": "full", "op": "gte", "value": 0.9, "for": "5m"},
    "effect": {"type": "task", "title_hint": "take out the trash", "urgency": "low"},
    "resolve": {"conditions": {"signal": "full", "op": "lte", "value": 0.3, "for": "1m"}},
    "limits": {"cooldown": "30m", "max_per_day": 4},
}


def episodes_at(times, *, skill="trash-cycle", anchor="kitchen.counter") -> list[EpisodeRow]:
    return [EpisodeRow(skill_id=skill, anchor_id=anchor, opened_at=moment) for moment in times]


def every_three_days(*, count: int, ending: datetime = BASE) -> list[datetime]:
    """`count` episodes, one every 3 days, the last one at `ending`."""
    return [ending - timedelta(days=3 * (count - 1 - index)) for index in range(count)]


async def seed_episodes(
    *, skill: str = "trash-cycle", anchor: str = "kitchen.counter", count: int = 5
) -> None:
    """Write regular 3-day episodes into the test database."""
    async with session_scope() as session:
        for moment in every_three_days(count=count):
            session.add(EpisodeRow(skill_id=skill, anchor_id=anchor, opened_at=moment))


# ------------------------------------------------------------------ discovery (pure)


def test_cadence_from_regular_episodes() -> None:
    candidates = discover_patterns(
        episodes_at(every_three_days(count=5)),
        now=BASE,
        labels={"kitchen.counter": "Kitchen counter"},
    )
    cadence = next(c for c in candidates if c.kind == "cadence")
    assert "every 3 days" in cadence.summary
    assert cadence.evidence["median_interval_h"] == pytest.approx(72.0)
    assert cadence.evidence["n_episodes"] == 5
    assert cadence.confidence >= 0.6


def test_too_few_episodes_claims_nothing() -> None:
    """One or two episodes are not a habit, and must never be presented as one."""
    for count in (1, 2, 3):
        candidates = discover_patterns(episodes_at(every_three_days(count=count)), now=BASE)
        assert candidates == [], f"{count} episodes must not produce a pattern"


def test_short_span_claims_nothing() -> None:
    """Four episodes in one frantic day is not a cadence either."""
    times = [BASE - timedelta(hours=hour) for hour in (6, 4, 2, 0)]
    candidates = discover_patterns(episodes_at(times), now=BASE)
    assert not any(c.kind == "cadence" for c in candidates)


def test_everyday_activity_is_not_a_cadence() -> None:
    """Cooking every day is life, not a replenishment cycle worth predicting."""
    times = [BASE - timedelta(days=days) for days in range(14, 0, -1)]
    candidates = discover_patterns(episodes_at(times), now=BASE)
    assert not any(c.kind == "cadence" for c in candidates)


def test_time_of_day_pattern() -> None:
    times = [BASE.replace(hour=19, minute=10) - timedelta(days=days) for days in (12, 9, 6, 3)]
    candidates = discover_patterns(
        episodes_at(times), now=BASE, labels={"kitchen.counter": "Kitchen counter"}
    )
    daypart = next(c for c in candidates if c.kind == "time_of_day")
    assert "evening" in daypart.summary
    assert daypart.evidence["peak_ratio"] == 1.0


def test_spread_out_times_claim_no_daypart() -> None:
    """Episodes across the day have no \"usual\" time, and must not be given one."""
    times = [BASE.replace(hour=hour) for hour in (7, 12, 18, 5)]  # one per daypart
    candidates = discover_patterns(episodes_at(times), now=BASE)
    assert not any(c.kind == "time_of_day" for c in candidates)


def test_confidence_grows_with_evidence() -> None:
    small = discover_patterns(episodes_at(every_three_days(count=5)), now=BASE)
    large = discover_patterns(episodes_at(every_three_days(count=12)), now=BASE)
    small_cadence = next(c for c in small if c.kind == "cadence")
    large_cadence = next(c for c in large if c.kind == "cadence")
    assert large_cadence.confidence > small_cadence.confidence


def test_summaries_never_look_backwards() -> None:
    """The only claim shape is forward: "about every N days". No code path can mention how long
    anything has been undone, so the safety filter never has to catch a backwards claim."""
    candidates = discover_patterns(episodes_at(every_three_days(count=5)), now=BASE)
    for candidate in candidates:
        assert "left" not in candidate.summary.lower()
        assert "days ago" not in candidate.summary.lower()
        assert "missed" not in candidate.summary.lower()


# ------------------------------------------------------------------ due window (pure)


def _cadence_row(
    *, last_at: datetime = BASE - timedelta(days=3), basis: str = "EP1"
) -> MemoryPatternRow:
    return MemoryPatternRow(
        kind="cadence",
        skill_id="trash-cycle",
        anchor_id="kitchen.counter",
        summary="The kitchen counter usually needs attention about every 3 days.",
        confidence=0.8,
        status="active",  # column default applies at flush, not construction; this is a pure test
        evidence={
            "median_interval_h": 72.0,
            "mean_interval_h": 72.0,
            "n_episodes": 5,
            "last_episode_at": last_at.isoformat(),
            "last_episode_id": basis,
        },
    )


def test_pattern_due_inside_the_window() -> None:
    row = _cadence_row(last_at=BASE - timedelta(days=3))
    due, basis = pattern_due(row, now=BASE)  # 3 days after last episode, interval 3 days
    assert due is True
    assert basis == "EP1"


def test_pattern_due_outside_the_window() -> None:
    row = _cadence_row(last_at=BASE - timedelta(days=3))
    assert pattern_due(row, now=BASE + timedelta(days=4)) == (False, None)


def test_pattern_due_once_per_cycle() -> None:
    row = _cadence_row(last_at=BASE - timedelta(days=3))
    _, basis = pattern_due(row, now=BASE)
    row.last_nudge_basis = basis
    assert pattern_due(row, now=BASE) == (False, None)  # already spoken this cycle


def test_pattern_due_only_for_cadence() -> None:
    row = _cadence_row(last_at=BASE - timedelta(days=3))
    row.kind = "time_of_day"
    assert pattern_due(row, now=BASE) == (False, None)


# ------------------------------------------------------------------ the API and refresh


def _settings(tmp_path) -> Settings:
    return Settings(
        state_dir=str(tmp_path),
        config_dir=str(tmp_path / "config"),
        database={"url": "sqlite+aiosqlite:///" + str(tmp_path / "test.db")},
        bus={"url": "redis://127.0.0.1:6399/0"},
        llm={"provider": "echo"},
        snapshots={"directory": str(tmp_path / "snapshots")},
        notify={"channels": {}},
        personality={"default_personality": "plain"},
    )


@pytest.fixture
async def client(tmp_path):
    config = _settings(tmp_path)
    app = create_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        init_engine(config.database)
        await create_all()
        from openhup.api.state import AppState
        from openhup.bus import Bus
        from openhup.llm import PersonalityRenderer, UsageLog
        from openhup.llm.render import PLAIN
        from openhup.notify import Dispatcher, build_channels
        from openhup.voice import VoiceProvider

        bus = Bus(url=config.bus.url)
        await bus.connect()
        state = AppState(
            settings=config,
            bus=bus,
            usage=UsageLog(),
            provider=None,
            renderer=PersonalityRenderer(None, personalities={"plain": PLAIN}),
            dispatcher=Dispatcher(channels=build_channels({})),
            voice=VoiceProvider(config.voice, usage=UsageLog()),
        )
        state.personalities = {"plain": PLAIN}
        app.state.openhup = state
        try:
            yield http, state, config
        finally:
            await bus.close()
            await dispose()


async def test_patterns_endpoint_recomputes_and_lists_evidence(client) -> None:
    http, _, _ = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=SKILL)

    await seed_episodes()

    body = (await http.get("/api/v1/memory/patterns")).json()
    patterns = body["patterns"]
    assert len(patterns) >= 1
    cadence = next(p for p in patterns if p["kind"] == "cadence")
    assert "every 3 days" in cadence["summary"]
    assert cadence["evidence"]["n_episodes"] == 5
    assert cadence["nudge_eligible"] is True


async def test_dismissing_a_pattern_hides_and_persists_it(client) -> None:
    http, _, _ = client
    await seed_episodes()

    listed = (await http.get("/api/v1/memory/patterns")).json()["patterns"]
    assert listed, "seeded episodes should produce patterns"
    dismissed_ids = []
    for pattern in listed:
        assert (await http.delete(f"/api/v1/memory/patterns/{pattern['id']}")).status_code == 204
        dismissed_ids.append(pattern["id"])

    assert (await http.get("/api/v1/memory/patterns")).json()["patterns"] == []
    # Dismissed rows are kept so the pattern is not learned again.
    async with session_scope() as session:
        for pattern_id in dismissed_ids:
            row = await session.get(MemoryPatternRow, pattern_id)
            assert row is not None
            assert row.status == "dismissed"

    assert (await http.delete("/api/v1/memory/patterns/nope")).status_code == 404


async def test_refresh_keeps_dismissed_patterns_dismissed(client) -> None:
    http, _, _ = client
    await seed_episodes()

    listed = (await http.get("/api/v1/memory/patterns")).json()["patterns"]
    for pattern in listed:
        await http.delete(f"/api/v1/memory/patterns/{pattern['id']}")

    # A second recompute must not resurrect them with fresh evidence.
    body = (await http.get("/api/v1/memory/patterns")).json()
    assert body["patterns"] == []


# ------------------------------------------------------------------ voice recall


async def test_voice_recalls_learned_patterns(client) -> None:
    http, _, _ = client
    await seed_episodes()

    body = (await http.post("/api/v1/voice/command", json={"text": "what have you noticed"})).json()
    assert body["intent"] == "memory"
    assert body["action"] == "recall_patterns"
    assert "every 3 days" in body["reply"]

    about = (
        await http.post(
            "/api/v1/voice/command", json={"text": "what have you noticed about the counter"}
        )
    ).json()
    assert "every 3 days" in about["reply"]


async def test_voice_is_honest_when_nothing_learned(client) -> None:
    http, _, _ = client
    body = (await http.post("/api/v1/voice/command", json={"text": "what have you noticed"})).json()
    assert body["intent"] == "memory"
    assert "Nothing yet" in body["reply"]


# ------------------------------------------------------------------ engine nudge pass


def _make_engine(state, config):
    from openhup.engine import Engine
    from openhup.llm.render import PLAIN, PersonalityRenderer
    from openhup.notify import Dispatcher, build_channels

    engine = Engine(
        settings=config,
        bus=state.bus,
        renderer=PersonalityRenderer(None, personalities={"plain": PLAIN}),
        dispatcher=Dispatcher(channels=build_channels({})),
    )
    return engine


async def test_engine_nudges_a_due_pattern_once(client) -> None:
    http, state, config = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=SKILL)

    await seed_episodes()

    engine = _make_engine(state, config)
    await engine.load()
    engine.is_leader = True

    heard: list = []
    engine.bus.subscribe_local(heard.append)

    now = BASE + timedelta(hours=60)  # 2.5 days after the last episode; interval is 3 days
    sent = await engine._pattern_nudge_pass(now=now)
    assert sent == 1
    nudges = [e for e in heard if e.type.value == "system.pattern_nudge"]
    assert len(nudges) == 1
    assert "every 3 days" in nudges[0].payload["text"]

    # Same cycle, second pass: no repeat.
    assert await engine._pattern_nudge_pass(now=now) == 0
    assert len([e for e in heard if e.type.value == "system.pattern_nudge"]) == 1


async def test_engine_skips_when_the_spot_is_already_handled(client) -> None:
    http, state, config = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=SKILL)

    await seed_episodes()
    async with session_scope() as session:
        session.add(
            TaskRow(
                id="01K3XQ8V4W7YB2M9C6NZ0PRSTA",
                skill_id="trash-cycle",
                anchor_id="kitchen.counter",
                episode_id="01K3XQ8V4W7YB2M9C6NZ0PRSTB",
                text="Take out the trash.",
                plain_text="Take out the trash.",
            )
        )

    engine = _make_engine(state, config)
    await engine.load()
    engine.is_leader = True
    assert await engine._pattern_nudge_pass(now=BASE + timedelta(hours=60)) == 0


async def test_engine_respects_quiet_hours(client) -> None:
    http, state, config = client
    quiet_skill = {
        **SKILL,
        "id": "quiet-trash",
        "limits": {
            "cooldown": "30m",
            "max_per_day": 4,
            "quiet_hours": {"between": ["00:00", "23:59"]},
        },
    }
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=quiet_skill)

    await seed_episodes(skill="quiet-trash")

    engine = _make_engine(state, config)
    await engine.load()
    engine.is_leader = True
    # A pattern may never earn a buzz inside the skill's own quiet hours.
    assert await engine._pattern_nudge_pass(now=BASE + timedelta(hours=60)) == 0


async def test_engine_respects_the_skills_own_cooldown(client) -> None:
    """A pattern nudge is a nudge about a skill, so the skill's `cooldown` applies to it."""
    http, state, config = client
    slow_skill = {**SKILL, "id": "slow-trash", "limits": {"cooldown": "5h", "max_per_day": None}}
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=slow_skill)
    await seed_episodes(skill="slow-trash")

    engine = _make_engine(state, config)
    await engine.load()
    engine.is_leader = True
    first = BASE + timedelta(hours=60)
    assert await engine._pattern_nudge_pass(now=first) == 1

    # A new episode arrives and opens a new cycle, but the skill's cooldown is 5h and only 2h
    # have passed: still silent.
    async with session_scope() as session:
        session.add(
            EpisodeRow(
                skill_id="slow-trash",
                anchor_id="kitchen.counter",
                opened_at=BASE + timedelta(minutes=30),
            )
        )
    second = first + timedelta(hours=2)
    assert await engine._pattern_nudge_pass(now=second) == 0

    # Past the cooldown, a due cycle speaks again.
    async with session_scope() as session:
        row = (await session.execute(select(MemoryPatternRow))).scalars().first()
        assert row is not None
        row.last_nudge_at = first - timedelta(hours=6)
    assert await engine._pattern_nudge_pass(now=second + timedelta(hours=4)) == 1


async def test_engine_respects_the_skills_own_max_per_day(client) -> None:
    """`max_per_day` on the skill caps pattern nudges about it, exactly as it caps its triggers."""
    http, state, config = client
    capped_skill = {**SKILL, "id": "capped-trash", "limits": {"cooldown": "1m", "max_per_day": 1}}
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=capped_skill)
    await seed_episodes(skill="capped-trash")

    engine = _make_engine(state, config)
    await engine.load()
    engine.is_leader = True
    first = BASE + timedelta(hours=60)
    assert await engine._pattern_nudge_pass(now=first) == 1

    # A new cycle the same day, but the skill allows one pattern nudge a day.
    async with session_scope() as session:
        session.add(
            EpisodeRow(
                skill_id="capped-trash",
                anchor_id="kitchen.counter",
                opened_at=BASE + timedelta(minutes=30),
            )
        )
    assert await engine._pattern_nudge_pass(now=first + timedelta(hours=2)) == 0


async def test_standby_engine_never_nudges(client) -> None:
    http, state, config = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=SKILL)
    await seed_episodes()

    engine = _make_engine(state, config)
    await engine.load()
    engine.is_leader = False  # warm standby: windows warm, effects never
    assert await engine._pattern_nudge_pass(now=BASE + timedelta(hours=60)) == 0
