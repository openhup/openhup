"""The personality gamble (ADR-014): a voice drawn at setup, a mystery until revealed.

The invariants under test: a draw only ever comes from the intersection of the configured pool
and the loaded personalities (so a mystery voice is never silently clamped or missing), it
persists across restarts, re-draws are explicit and counted, deleting it restores the configured
default exactly, and the whole lifecycle is inspectable through the API.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from openhup_schemas import Personality

from openhup.db import create_all, dispose, init_engine, session_scope
from openhup.personality import (
    GAMBLE_POOL,
    clear,
    draw,
    effective_default_id,
    load_draw,
)

GAMBLE_PRESETS = {
    pid: Personality(
        id=pid,
        display_name=pid.title(),
        intensity=3,
    )
    for pid in GAMBLE_POOL
}


# ------------------------------------------------------------------ draw logic


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


async def test_draw_comes_from_the_pool_intersection(db) -> None:
    """A draw never names a personality that did not load, and never leaves the pool."""
    async with session_scope() as session:
        row = await draw(session, pool=list(GAMBLE_POOL), available=list(GAMBLE_POOL))
        assert row.personality_id in GAMBLE_POOL
        assert await load_draw(session) is not None


async def test_draw_skips_ids_that_are_not_loaded(db) -> None:
    """An operator who widens the pool past the shipped presets gets only the real ones."""
    async with session_scope() as session:
        row = await draw(
            session,
            pool=["friendly", "shy", "does_not_exist"],
            available=["friendly", "shy"],
        )
        assert row.personality_id in {"friendly", "shy"}


async def test_draw_refuses_an_empty_pool(db) -> None:
    async with session_scope() as session:
        with pytest.raises(ValueError):
            await draw(session, pool=[], available=["brief"])


async def test_reroll_counts_and_persists(db) -> None:
    async with session_scope() as session:
        first = await draw(session, pool=["sassy"], available=["sassy"])
        assert first.reroll_count == 0
        second = await draw(session, pool=["sassy"], available=["sassy"])
        assert second.reroll_count == 1
        assert second.personality_id == "sassy"

    # The draw survives the session: a restart finds the same voice and the same count.
    async with session_scope() as session:
        row = await load_draw(session)
        assert row is not None
        assert row.personality_id == "sassy"
        assert row.reroll_count == 1


async def test_clear_restores_the_configured_default(db) -> None:
    async with session_scope() as session:
        await draw(session, pool=["sarcastic"], available=["sarcastic"])
        assert effective_default_id("kind_coach", await load_draw(session)) == "sarcastic"
        assert await clear(session) is True
        assert await load_draw(session) is None
        assert effective_default_id("kind_coach", None) == "kind_coach"


def test_effective_default_is_the_draw_when_there_is_one() -> None:
    from types import SimpleNamespace

    assert effective_default_id("kind_coach", None) == "kind_coach"
    drawn = SimpleNamespace(personality_id="sassy")
    assert effective_default_id("kind_coach", drawn) == "sassy"


# ------------------------------------------------------------------ the API


@pytest.fixture
async def client(tmp_path):
    from openhup.api.main import create_app
    from openhup.bus import Bus
    from openhup.core.config import Settings
    from openhup.llm import PersonalityRenderer, UsageLog
    from openhup.llm.render import PLAIN
    from openhup.notify import Dispatcher, build_channels
    from openhup.voice import VoiceProvider

    settings = Settings(
        state_dir=str(tmp_path),
        config_dir=str(tmp_path / "config"),
        database={"url": "sqlite+aiosqlite:///" + str(tmp_path / "test.db")},
        bus={"url": "redis://127.0.0.1:6399/0"},
        llm={"provider": "echo"},
        notify={"channels": {}},
        personality={"default_personality": "plain"},
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        init_engine(settings.database)
        await create_all()
        from openhup.api.state import AppState

        bus = Bus(url=settings.bus.url)
        await bus.connect()
        state = AppState(
            settings=settings,
            bus=bus,
            usage=UsageLog(),
            provider=None,
            renderer=PersonalityRenderer(None, personalities={"plain": PLAIN}),
            dispatcher=Dispatcher(channels=build_channels({})),
            voice=VoiceProvider(settings.voice, usage=UsageLog()),
        )
        state.personalities = {"plain": PLAIN, **GAMBLE_PRESETS}
        state.renderer.personalities = state.personalities
        app.state.openhup = state
        try:
            yield http, state, settings
        finally:
            await bus.close()
            await dispose()


async def test_draw_lifecycle_through_the_api(client) -> None:
    http, state, _ = client

    empty = (await http.get("/api/v1/personality/draw")).json()
    assert empty["drawn"] is None

    created = (await http.post("/api/v1/personality/draw")).json()
    assert created["drawn"] in GAMBLE_POOL
    assert created["reroll_count"] == 0

    fetched = (await http.get("/api/v1/personality/draw")).json()
    assert fetched["drawn"] == created["drawn"]

    # The drawn voice is now the effective default, visible to operators on /system/info.
    info = (await http.get("/api/v1/system/info")).json()
    assert info["personality"]["default"] == created["drawn"]
    assert info["personality"]["configured_default"] == "plain"

    # Re-drawing is explicit and counted.
    rerolled = (await http.post("/api/v1/personality/draw")).json()
    assert rerolled["reroll_count"] == 1
    assert rerolled["drawn"] in GAMBLE_POOL

    # Deleting returns to the configured default exactly.
    assert (await http.delete("/api/v1/personality/draw")).status_code == 204
    assert (await http.get("/api/v1/personality/draw")).json()["drawn"] is None
    info = (await http.get("/api/v1/system/info")).json()
    assert info["personality"]["default"] == "plain"
    assert state.renderer.settings.default_personality == "plain"


async def test_draw_with_an_empty_pool_is_refused(client) -> None:
    http, state, _ = client
    state.settings.personality.gamble_pool = []
    response = await http.post("/api/v1/personality/draw")
    assert response.status_code == 422


async def test_wins_endpoint_is_reviewable_and_forgettable(client) -> None:
    http, _, _ = client
    assert (await http.get("/api/v1/personality/wins")).json() == {"wins": []}

    from openhup.db import WinMilestoneRow

    async with session_scope() as session:
        session.add(
            WinMilestoneRow(
                id="01K3XQ8V4W7YB2M9C6NZ0PRSTD",
                anchor_id="kitchen.counter",
                kind="record_clear_days",
                value=3.0,
                days=3.0,
                summary="Kitchen counter stayed clear 3 days - its longest clear stretch.",
                spoken_at=None,
            )
        )

    body = (await http.get("/api/v1/personality/wins")).json()
    assert len(body["wins"]) == 1
    assert body["wins"][0]["kind"] == "record_clear_days"
    assert body["wins"][0]["spoken"] is False

    deleted = await http.delete("/api/v1/personality/wins/01K3XQ8V4W7YB2M9C6NZ0PRSTD")
    assert deleted.status_code == 204
    assert (await http.get("/api/v1/personality/wins")).json() == {"wins": []}
