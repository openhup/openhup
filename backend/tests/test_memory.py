"""Household memory tests.

Three layers, matching the implementation:

* The retrieval helper (`relevant_facts`) is keyword matching over a local store - no model, no
  embeddings, and an empty or stop-word-only query must not dump the store into a prompt.
* The API is the review screen's backend: list, add, per-row delete. A memory the user cannot
  inspect and delete is not a memory.
* The voice intents teach, recall, and forget - deterministic, like every other intent - and the
  phrasing renderer injects retrieved facts into the prompt as context, never into the plain line.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from openhup_schemas import Personality, PersonalitySettings, Templates, TextSource

from openhup.api.main import create_app
from openhup.core.config import Settings
from openhup.db import create_all, dispose, init_engine
from openhup.voice import is_query, navigation_target


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


# ------------------------------------------------------------------ retrieval


async def test_relevant_facts_matches_by_fact_and_topic(client) -> None:
    from openhup.db import MemoryFactRow, session_scope
    from openhup.memory import relevant_facts

    async with session_scope() as session:
        session.add(MemoryFactRow(fact="bin day is tuesday", topic="trash", source="voice"))
        session.add(
            MemoryFactRow(fact="the spare room is the junk room", topic="naming", source="voice")
        )
        session.add(
            MemoryFactRow(fact="water the plants on sundays", topic="plants", source="voice")
        )

        hits = await relevant_facts(session, query="bin day trash")
        assert hits == ["bin day is tuesday"]

        about = await relevant_facts(session, query="junk room")
        assert "the spare room is the junk room" in about


async def test_relevant_facts_ignores_empty_and_stopword_queries(client) -> None:
    from openhup.db import MemoryFactRow, session_scope
    from openhup.memory import relevant_facts

    async with session_scope() as session:
        session.add(MemoryFactRow(fact="bin day is tuesday", topic="trash", source="voice"))
        assert await relevant_facts(session, query="") == []
        assert await relevant_facts(session, query="is the") == []
        assert await relevant_facts(session, query="   ") == []


# ------------------------------------------------------------------ the API


async def test_memory_api_lists_adds_and_deletes(client) -> None:
    http = client
    created = (await http.post("/api/v1/memory", json={"fact": "bin day is tuesday"})).json()
    assert "created" in created

    listed = (await http.get("/api/v1/memory")).json()
    assert len(listed) == 1
    assert listed[0]["fact"] == "bin day is tuesday"
    assert listed[0]["source"] == "settings"
    assert listed[0]["topic"] is None

    deleted = await http.delete(f"/api/v1/memory/{created['created']}")
    assert deleted.status_code == 204
    assert (await http.get("/api/v1/memory")).json() == []

    again = await http.delete(f"/api/v1/memory/{created['created']}")
    assert again.status_code == 404


async def test_memory_api_accepts_topic_and_rejects_blank(client) -> None:
    http = client
    created = (
        await http.post("/api/v1/memory", json={"fact": "junk room", "topic": "naming"})
    ).json()
    listed = (await http.get("/api/v1/memory")).json()
    assert listed[0]["topic"] == "naming"
    assert listed[0]["id"] == created["created"]

    blank = await http.post("/api/v1/memory", json={"fact": "   "})
    assert blank.status_code == 422


# ------------------------------------------------------------------ voice


async def test_voice_teaches_a_fact(client) -> None:
    http = client
    body = (
        await http.post("/api/v1/voice/command", json={"text": "remember that bin day is tuesday"})
    ).json()
    assert body["intent"] == "memory"
    assert body["action"] == "remember"
    assert "bin day is tuesday" in body["reply"]

    listed = (await http.get("/api/v1/memory")).json()
    assert len(listed) == 1
    assert listed[0]["fact"] == "bin day is tuesday"
    assert listed[0]["source"] == "voice"


async def test_voice_recalls_by_topic(client) -> None:
    http = client
    await http.post("/api/v1/memory", json={"fact": "bin day is tuesday", "topic": "trash"})

    body = (
        await http.post("/api/v1/voice/command", json={"text": "what do you remember about bin"})
    ).json()
    assert body["intent"] == "memory"
    assert body["action"] == "recall"
    assert "bin day is tuesday" in body["reply"]

    none = (
        await http.post("/api/v1/voice/command", json={"text": "what do you remember about plants"})
    ).json()
    assert "don't remember anything about plants" in none["reply"]


async def test_voice_recalls_the_whole_store(client) -> None:
    http = client
    # An empty store is answered honestly before anything is taught.
    empty = (await http.post("/api/v1/voice/command", json={"text": "what do you know"})).json()
    assert empty["intent"] == "memory"
    assert "don't remember anything yet" in empty["reply"]

    await http.post("/api/v1/memory", json={"fact": "bin day is tuesday"})
    body = (await http.post("/api/v1/voice/command", json={"text": "what do you remember"})).json()
    assert body["intent"] == "memory"
    assert "bin day is tuesday" in body["reply"]


async def test_voice_forgets_one_fact(client) -> None:
    http = client
    await http.post("/api/v1/memory", json={"fact": "bin day is tuesday"})
    await http.post("/api/v1/memory", json={"fact": "water the plants on sundays"})

    body = (
        await http.post("/api/v1/voice/command", json={"text": "forget that bin day is tuesday"})
    ).json()
    assert body["intent"] == "memory"
    assert body["action"] == "forget"
    assert body["reply"] == "Forgotten."

    listed = (await http.get("/api/v1/memory")).json()
    assert [f["fact"] for f in listed] == ["water the plants on sundays"]


async def test_voice_forget_everything_wipes_the_store(client) -> None:
    http = client
    await http.post("/api/v1/memory", json={"fact": "bin day is tuesday"})
    await http.post("/api/v1/memory", json={"fact": "water the plants on sundays"})

    body = (await http.post("/api/v1/voice/command", json={"text": "forget everything"})).json()
    assert body["reply"] == "Forgotten."
    assert (await http.get("/api/v1/memory")).json() == []

    none = (await http.post("/api/v1/voice/command", json={"text": "forget the shed"})).json()
    assert none["reply"] == "I don't remember that."


async def test_remember_to_falls_through_to_skill_dictation(client) -> None:
    """\"remember to ...\" is a reminder, not a fact - it must reach the skill parser."""
    http = client
    body = (
        await http.post(
            "/api/v1/voice/command", json={"text": "remember to remind me when the trash is full"}
        )
    ).json()
    assert body["intent"] == "skill_dictation"
    assert (await http.get("/api/v1/memory")).json() == []


async def test_memory_phrases_do_not_collide_with_other_intents(client) -> None:
    assert navigation_target("show tasks") == "/tasks"
    assert is_query("what should i do") is True


# ------------------------------------------------------------------ renderer injection


async def test_memory_facts_reach_the_phrasing_prompt() -> None:
    from openhup.llm import EchoProvider, PersonalityRenderer

    chatty = Personality(id="chatty", display_name="Chatty", intensity=2)
    provider = EchoProvider({"clear the kitchen counter": "Clear the counter, as requested."})
    render = PersonalityRenderer(
        provider,
        personalities={"chatty": chatty},
        settings=PersonalitySettings(default_personality="chatty"),
    )
    result = await render.task(
        title_hint="clear the kitchen counter",
        anchor_label="Kitchen counter",
        personality_id="chatty",
        memory=["I call the kitchen counter the junk counter"],
    )
    assert result.source is TextSource.LLM
    prompt = "\n".join(message.content for message in provider.calls[0])
    assert "Things to keep in mind" in prompt
    assert "junk counter" in prompt


async def test_memory_never_reaches_plain_text() -> None:
    from openhup.llm import PersonalityRenderer
    from openhup.llm.render import PLAIN

    render = PersonalityRenderer(None, personalities={"plain": PLAIN})
    result = await render.task(
        title_hint="clear the kitchen counter",
        anchor_label="Kitchen counter",
        memory=["I call the kitchen counter the junk counter"],
    )
    assert result.text == "Clear the kitchen counter."
    assert result.source is TextSource.TEMPLATE


async def test_memory_reaches_template_facts_without_a_model() -> None:
    from openhup.llm import PersonalityRenderer

    chatty = Personality(
        id="chatty",
        display_name="Chatty",
        intensity=2,
        templates=Templates(task="{title_hint} ({facts})"),
    )
    render = PersonalityRenderer(
        None,  # no provider at all - templates are the whole voice
        personalities={"chatty": chatty},
        settings=PersonalitySettings(default_personality="chatty"),
    )
    result = await render.task(
        title_hint="clear the counter",
        anchor_label="Kitchen counter",
        personality_id="chatty",
        memory=["burner goes off at ten"],
    )
    assert result.source is TextSource.TEMPLATE
    assert "burner goes off at ten" in result.text
