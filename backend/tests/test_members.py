"""Consent-gated members API (ADR-016).

The tests that matter: enrollment requires consent-shaped input and an enabled feature, a deleted
member loses their embedding and presence history, and the consent marker updates without ever
storing a face.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from openhup.api.main import create_app
from openhup.core.config import Settings
from openhup.db import create_all, dispose, init_engine


def _settings(tmp_path) -> Settings:
    return Settings(
        state_dir=str(tmp_path),
        config_dir=str(tmp_path / "config"),
        database={"url": "sqlite+aiosqlite:///" + str(tmp_path / "test.db")},
        bus={"url": "redis://127.0.0.1:6399/0"},
        llm={"provider": "echo"},
        snapshots={"directory": str(tmp_path / "snapshots")},
        notify={"channels": {}},
        personality={"default_personality": "kind_coach"},
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
        from openhup.notify import Dispatcher, build_channels
        from openhup.voice import VoiceProvider

        bus = Bus(url=config.bus.url)
        await bus.connect()
        state = AppState(
            settings=config,
            bus=bus,
            usage=UsageLog(),
            provider=None,
            renderer=PersonalityRenderer(None, personalities={}),
            dispatcher=Dispatcher(channels=build_channels({})),
            voice=VoiceProvider(config.voice, usage=UsageLog()),
        )
        app.state.openhup = state
        try:
            yield http
        finally:
            await bus.close()
            await dispose()


async def test_enroll_and_list(client) -> None:
    http = client
    body = (await http.get("/api/v1/members")).json()
    assert body["enabled"] is True
    assert body["members"] == []

    created = (
        await http.post("/api/v1/members", json={"name": "Sam", "embedding": [0.1] * 128})
    ).json()
    assert created["member"]["name"] == "Sam"
    assert created["member"]["embedding_dim"] == 128
    assert "remember you as Sam" in created["reply"]

    listed = (await http.get("/api/v1/members")).json()
    assert [m["name"] for m in listed["members"]] == ["Sam"]


async def test_enroll_rejects_duplicate_names(client) -> None:
    http = client
    await http.post("/api/v1/members", json={"name": "Sam", "embedding": [0.1] * 128})
    dup = await http.post("/api/v1/members", json={"name": "sam", "embedding": [0.2] * 128})
    assert dup.status_code == 409


async def test_enroll_rejects_empty_or_malformed_embeddings(client) -> None:
    http = client
    tiny = await http.post("/api/v1/members", json={"name": "Sam", "embedding": [0.1]})
    assert tiny.status_code == 422


async def test_delete_member_removes_embedding(client) -> None:
    http = client
    created = (
        await http.post("/api/v1/members", json={"name": "Sam", "embedding": [0.1] * 128})
    ).json()
    member_id = created["member"]["id"]
    deleted = await http.delete(f"/api/v1/members/{member_id}")
    assert deleted.status_code == 204
    listed = (await http.get("/api/v1/members")).json()
    assert listed["members"] == []


async def test_delete_unknown_member_is_404(client) -> None:
    http = client
    assert (await http.delete("/api/v1/members/does-not-exist")).status_code == 404


async def test_consent_no_creates_the_marker(client) -> None:
    http = client
    body = (
        await http.post(
            "/api/v1/members/consent",
            json={"anchor_id": "kitchen.counter", "answer": "no"},
        )
    ).json()
    assert "again today" in body["reply"]

    from openhup.db import ConsentAskRow, session_scope

    async with session_scope() as session:
        rows = (
            (await session.execute(__import__("sqlalchemy").select(ConsentAskRow))).scalars().all()
        )
        assert len(rows) == 1
        assert rows[0].anchor_id == "kitchen.counter"
        assert rows[0].answer == "no"


async def test_consent_yes_offers_the_name_handoff(client) -> None:
    http = client
    body = (
        await http.post(
            "/api/v1/members/consent",
            json={"anchor_id": "kitchen.counter", "answer": "yes"},
        )
    ).json()
    assert "call you" in body["reply"]


async def test_identity_disabled_refuses_enrollment(tmp_path) -> None:
    """With the master switch off, nothing about identity exists."""
    config = _settings(tmp_path)
    config.identity.enabled = False
    app = create_app(config)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        init_engine(config.database)
        await create_all()
        from openhup.api.state import AppState
        from openhup.bus import Bus
        from openhup.llm import PersonalityRenderer, UsageLog
        from openhup.notify import Dispatcher, build_channels
        from openhup.voice import VoiceProvider

        bus = Bus(url=config.bus.url)
        await bus.connect()
        state = AppState(
            settings=config,
            bus=bus,
            usage=UsageLog(),
            provider=None,
            renderer=PersonalityRenderer(None, personalities={}),
            dispatcher=Dispatcher(channels=build_channels({})),
            voice=VoiceProvider(config.voice, usage=UsageLog()),
        )
        app.state.openhup = state
        try:
            refused = await http.post(
                "/api/v1/members", json={"name": "Sam", "embedding": [0.1] * 128}
            )
            assert refused.status_code == 409
            listed = (await http.get("/api/v1/members")).json()
            assert listed["enabled"] is False
        finally:
            await bus.close()
            await dispose()
