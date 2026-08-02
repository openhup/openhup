"""Voice interface tests.

Two halves, matching the split in the code:

* The pure detection helpers (`navigation_target`, `command_action`, `snooze_minutes`, `is_query`)
  are the whole deterministic brain; they are tested without any I/O.
* `VoiceSettings` enforces the same egress gate as the LLM, so a remote speech provider cannot be
  enabled by accident.
* The `/voice/config` and `/voice/command` endpoints are driven over the API against SQLite, the
  same way `test_api.py` does, to prove a transcript turns into a side effect and a spoken reply.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from openhup.api.main import create_app
from openhup.core.config import Settings, VoiceSettings
from openhup.db import create_all, dispose, init_engine
from openhup.voice import (
    command_action,
    is_query,
    navigation_target,
    snooze_minutes,
)

CAMERA = {"id": "kitchen", "name": "Kitchen", "kind": "rtsp", "url": "rtsp://camera.invalid/s"}
ANCHOR = {
    "id": "kitchen.counter",
    "camera_id": "kitchen",
    "label": "Kitchen counter",
    "polygon": [[0.1, 0.3], [0.9, 0.3], [0.9, 0.8], [0.1, 0.8]],
}
SKILL = {
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
    "resolve": {"conditions": {"signal": "clutter", "op": "lte", "value": 0.25, "for": "1m"}},
    "limits": {"cooldown": "30m", "max_per_day": 4},
}


# ------------------------------------------------------------------ pure helpers


def test_navigation_target() -> None:
    assert navigation_target("show tasks") == "/tasks"
    assert navigation_target("open cameras") == "/cameras"
    assert navigation_target("go to metrics") == "/metrics"
    assert navigation_target("skills") == "/skills"
    assert navigation_target("read my task") is None  # a query, not a destination


def test_command_action() -> None:
    assert command_action("done") == ("complete", 0)
    assert command_action("start") == ("start", 0)
    assert command_action("snooze for 30 minutes") == ("snooze", 30)
    assert command_action("not a real task") == ("false_positive", 0)
    assert command_action("dismiss") == ("dismiss", 0)
    assert command_action("what should i do") is None


def test_snooze_minutes() -> None:
    assert snooze_minutes("snooze for 30 minutes") == 30
    assert snooze_minutes("later for 2 hours") == 120
    assert snooze_minutes("snooze for 1 day") == 1440
    assert snooze_minutes("snooze") == 60


def test_is_query() -> None:
    assert is_query("what should i do") is True
    assert is_query("read my task") is True
    assert is_query("done") is False


# ------------------------------------------------------------------ egress gate


def test_browser_voice_needs_no_consent() -> None:
    settings = VoiceSettings()
    assert settings.stt_remote is False
    assert settings.tts_remote is False


def test_remote_stt_is_refused_without_consent() -> None:
    with pytest.raises(ValueError, match="allow_remote_voice"):
        VoiceSettings(stt_provider="openai")
    # The explicit opt-in is enough, whatever the key situation.
    VoiceSettings(stt_provider="openai", allow_remote_voice=True)


def test_remote_tts_is_refused_without_consent() -> None:
    with pytest.raises(ValueError, match="allow_remote_voice"):
        VoiceSettings(tts_provider="openai_compatible", base_url="http://voice:8080/v1")
    # A local gateway is not egress and must not be gated.
    VoiceSettings(
        tts_provider="openai_compatible", base_url="http://voice:8080/v1", treat_as_local=True
    )


# ------------------------------------------------------------------ the API


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
            yield http
        finally:
            await bus.close()
            await dispose()


async def test_voice_config_reports_browser_by_default(client) -> None:
    http = client
    body = (await http.get("/api/v1/voice/config")).json()
    assert body["enabled"] is True
    assert body["stt_provider"] == "browser"
    assert body["tts_provider"] == "browser"
    assert body["stt_remote"] is False
    assert body["wake_word"] == "hey openhup"


async def test_command_with_no_task_is_honest(client) -> None:
    http = client
    body = (await http.post("/api/v1/voice/command", json={"text": "done"})).json()
    assert body["intent"] == "task_command"
    assert body["reply"] == "There's nothing to do right now."

    query = (await http.post("/api/v1/voice/command", json={"text": "what should i do"})).json()
    assert query["intent"] == "query"
    assert query["reply"] == "Nothing right now. You're clear."


async def test_command_navigates(client) -> None:
    http = client
    body = (await http.post("/api/v1/voice/command", json={"text": "show tasks"})).json()
    assert body["intent"] == "navigate"
    assert body["target"] == "/tasks"


async def test_unknown_command_is_not_guessed(client) -> None:
    http = client
    body = (await http.post("/api/v1/voice/command", json={"text": "flargl blargl"})).json()
    assert body["intent"] == "unknown"
    assert "didn't catch" in body["reply"]


async def test_command_completes_the_next_task(client) -> None:
    http = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=SKILL)

    from openhup.db import TaskRow, session_scope

    async with session_scope() as session:
        session.add(
            TaskRow(
                id="01K3XQ8V4W7YB2M9C6NZ0PRSTA",
                skill_id="kitchen-clutter-buster",
                anchor_id="kitchen.counter",
                episode_id="01K3XQ8V4W7YB2M9C6NZ0PRSTB",
                text="Clear the counter.",
                plain_text="Clear the counter.",
            )
        )

    body = (await http.post("/api/v1/voice/command", json={"text": "done"})).json()
    assert body["intent"] == "task_command"
    assert body["action"] == "complete"
    assert body["task_id"] == "01K3XQ8V4W7YB2M9C6NZ0PRSTA"

    done = (await http.get("/api/v1/tasks?state=done")).json()
    assert done[0]["state"] == "resolved_manual"


async def test_command_reads_the_next_task(client) -> None:
    http = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=SKILL)

    from openhup.db import TaskRow, session_scope

    async with session_scope() as session:
        session.add(
            TaskRow(
                id="01K3XQ8V4W7YB2M9C6NZ0PRSTA",
                skill_id="kitchen-clutter-buster",
                anchor_id="kitchen.counter",
                episode_id="01K3XQ8V4W7YB2M9C6NZ0PRSTB",
                text="Clear the counter.",
                plain_text="Clear the counter.",
            )
        )

    body = (await http.post("/api/v1/voice/command", json={"text": "what should i do"})).json()
    assert body["intent"] == "query"
    assert body["reply"] == "Clear the counter."


# ------------------------------------------------------------------ identity (ADR-016)


async def test_self_id_against_unknown_name_is_honest(client) -> None:
    http = client
    body = (await http.post("/api/v1/voice/command", json={"text": "it's Sam"})).json()
    assert body["intent"] == "identity"
    assert "don't know anyone called" in body["reply"]
    assert body["speaker_id"] is None


async def test_self_id_confirms_an_enrolled_member(client) -> None:
    http = client
    await http.post(
        "/api/v1/members",
        json={"name": "Sam", "embedding": [0.1] * 64},
    )
    body = (await http.post("/api/v1/voice/command", json={"text": "call me Sam"})).json()
    assert body["intent"] == "identity"
    assert "Sam" in body["reply"]
    assert body["speaker_id"] is not None


async def test_who_am_i_without_a_speaker_asks_who_is_asking(client) -> None:
    http = client
    body = (await http.post("/api/v1/voice/command", json={"text": "who am i"})).json()
    assert body["intent"] == "identity"
    assert "don't know who's asking" in body["reply"]


async def test_consent_yes_leads_to_the_name_handoff(client) -> None:
    http = client
    body = (await http.post("/api/v1/voice/command", json={"text": "yes remember me"})).json()
    assert body["intent"] == "identity"
    assert "call me" in body["reply"]


async def test_query_without_speaker_in_a_multi_member_house_is_honest(client) -> None:
    """The next task belongs to the household, not to whoever happened to ask."""
    http = client
    await http.post("/api/v1/members", json={"name": "Sam", "embedding": [0.1] * 64})
    await http.post("/api/v1/members", json={"name": "Lee", "embedding": [0.2] * 64})
    body = (await http.post("/api/v1/voice/command", json={"text": "what should i do"})).json()
    assert body["intent"] == "query"
    assert "don't know who's asking" in body["reply"]

    # Declared identity unlocks the personal query.
    body = (
        await http.post(
            "/api/v1/voice/command", json={"text": "what should i do", "speaker": "sam"}
        )
    ).json()
    assert body["intent"] == "query"
    assert "don't know who's asking" not in body["reply"]
