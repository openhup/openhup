"""End-to-end API and engine tests over SQLite, with no Redis, no cameras, and no models.

That this is possible is a design property, not a testing trick: the bus degrades to in-process
queues, the LLM degrades to templates, and observations can be posted in directly. It means a
contributor can run the full loop on a laptop before touching any hardware.

The headline test is `test_full_loop_mess_to_task_to_resolution`, which drives the real engine with
synthetic observations and asserts that a task appears, carries a snapshot, and closes itself.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from openhup_schemas import (
    DetectorInfo,
    Observation,
    ObservationSource,
    Signal,
    SignalKind,
    load_skill_yaml,
)

from openhup.api.main import create_app
from openhup.core.config import Settings
from openhup.db import create_all, dispose, init_engine, session_scope

UTC = UTC

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
    "effect": {
        "type": "task",
        "title_hint": "clear the kitchen counter",
        "urgency": "low",
        "micro_steps": "none",
    },
    "resolve": {
        "conditions": {"signal": "clutter", "op": "lte", "value": 0.25, "for": "1m"},
        "grace": "0s",
    },
    "limits": {"cooldown": "30m", "max_per_day": 4},
}

CAMERA = {
    "id": "kitchen",
    "name": "Kitchen",
    "kind": "rtsp",
    "url": "rtsp://camera.invalid/stream",
    "substream_url": "rtsp://camera.invalid/sub",
    "username": "openhup",
    # The config names an env var; the secret itself lives only in the environment.
    "password_env": "TEST_KITCHEN_CAM_PASSWORD",
}

ANCHOR = {
    "id": "kitchen.counter",
    "camera_id": "kitchen",
    "label": "Kitchen counter",
    "polygon": [[0.1, 0.3], [0.9, 0.3], [0.9, 0.8], [0.1, 0.8]],
}


def settings(tmp_path) -> Settings:
    return Settings(
        state_dir=str(tmp_path),
        config_dir=str(tmp_path / "config"),
        database={"url": "sqlite+aiosqlite:///" + str(tmp_path / "test.db")},
        # No Redis in tests: the bus falls back to in-process queues, which is a supported mode.
        bus={"url": "redis://127.0.0.1:6399/0"},
        llm={"provider": "echo"},
        snapshots={"directory": str(tmp_path / "snapshots")},
        notify={"channels": {}},
        personality={"default_personality": "plain"},
    )


@pytest.fixture
async def client(tmp_path):
    config = settings(tmp_path)
    app = create_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        # Lifespan is not run by ASGITransport, so set up exactly what it would.
        init_engine(config.database)
        await create_all()
        from openhup.api.state import AppState
        from openhup.bus import Bus
        from openhup.llm import PersonalityRenderer, UsageLog
        from openhup.llm.render import PLAIN
        from openhup.notify import Dispatcher, build_channels
        from openhup.voice import VoiceProvider

        bus = Bus(url=config.bus.url)
        await bus.connect()  # fails over to local queues, by design
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


# ------------------------------------------------------------------ basics


async def test_health_and_readiness(client) -> None:
    http, _, _ = client
    assert (await http.get("/healthz")).json()["status"] == "ok"
    assert (await http.get("/readyz")).json()["status"] == "ready"


async def test_detector_registry_is_served(client) -> None:
    """The skill builder renders its inputs from this, so it has to be complete."""
    http, _, _ = client
    body = (await http.get("/api/v1/detectors")).json()
    names = {d["name"] for d in body["detectors"]}
    assert {"clutter_score", "zero_shot_state", "object_inventory", "screen_on"} <= names
    clutter = next(d for d in body["detectors"] if d["name"] == "clutter_score")
    assert any(s["key"] == "clutter_level" for s in clutter["signals"])
    assert clutter["cost"] in {"trivial", "low", "medium", "high"}


async def test_metric_catalog_includes_the_anti_metric(client) -> None:
    http, _, _ = client
    catalog = (await http.get("/api/v1/metrics/catalog")).json()
    assert "nag_index" in catalog
    assert "Lower is better" in catalog["nag_index"]


# ------------------------------------------------------------------ cameras and anchors


async def test_create_camera_and_anchor(client) -> None:
    http, _state, _ = client
    assert (await http.post("/api/v1/cameras", json=CAMERA)).status_code == 201
    assert (await http.post("/api/v1/anchors", json=ANCHOR)).status_code == 201

    cameras = (await http.get("/api/v1/cameras")).json()
    assert cameras[0]["id"] == "kitchen"
    assert cameras[0]["anchors"] == ["kitchen.counter"]
    # The API returns the *name* of the env var holding the password, never a secret value.
    assert cameras[0]["password_env"] == "TEST_KITCHEN_CAM_PASSWORD"
    assert "password" not in cameras[0]

    anchors = (await http.get("/api/v1/anchors")).json()
    assert anchors[0]["subregions"] == []
    assert anchors[0]["has_baseline"] is False


async def test_deleting_a_camera_keeps_its_anchors(client) -> None:
    """ADR-010: replacing hardware must not destroy the history of the places it watched."""
    http, _, _ = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)

    assert (await http.delete("/api/v1/cameras/kitchen")).status_code == 204
    anchors = (await http.get("/api/v1/anchors")).json()
    assert [a["id"] for a in anchors] == ["kitchen.counter"]


async def test_duplicate_camera_is_rejected(client) -> None:
    http, _, _ = client
    await http.post("/api/v1/cameras", json=CAMERA)
    assert (await http.post("/api/v1/cameras", json=CAMERA)).status_code == 409


# ------------------------------------------------------------------ skills


async def test_create_skill_and_read_it_back(client) -> None:
    http, _, _ = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)

    response = await http.post("/api/v1/skills", json=CLUTTER_SKILL)
    assert response.status_code == 201, response.text

    listed = (await http.get("/api/v1/skills")).json()
    assert listed[0]["id"] == "kitchen-clutter-buster"
    # The plain-language explanation is a first-class field, not a debug aid.
    assert "Watching kitchen.counter" in listed[0]["explanation"]
    assert "clears itself" in listed[0]["explanation"]

    detail = (await http.get("/api/v1/skills/kitchen-clutter-buster")).json()
    assert detail["compiled"]["signal_keys"] == ["kitchen.counter/clutter_score.clutter_level"]


async def test_flapping_skill_is_rejected_with_every_finding(client) -> None:
    """A skill whose thresholds overlap is refused at save time, with an actionable message."""
    http, _, _ = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)

    broken = {
        **CLUTTER_SKILL,
        "id": "flapper",
        "resolve": {
            "conditions": {"signal": "clutter", "op": "lte", "value": 0.7, "for": "1m"},
            "grace": "0s",
        },
    }
    response = await http.post("/api/v1/skills", json=broken)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "skill_compile_failed"
    assert any(f["code"] == "no_hysteresis" for f in body["findings"])
    assert any("open and close repeatedly" in f["message"] for f in body["findings"])


async def test_skill_referencing_a_missing_anchor_is_rejected(client) -> None:
    http, _, _ = client
    response = await http.post("/api/v1/skills", json=CLUTTER_SKILL)
    assert response.status_code == 422
    assert any(f["code"] == "unknown_anchor" for f in response.json()["findings"])


async def test_parse_without_an_llm_falls_back_to_heuristics(client) -> None:
    """A deployment with no LLM is still usable, and says plainly that it guessed."""
    http, _, _ = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)

    response = await http.post(
        "/api/v1/skills/parse", json={"text": "remind me when the kitchen counter is a mess"}
    )
    body = response.json()
    assert body["ok"] is True
    assert body["heuristic"] is True
    assert body["needs_confirmation"] is True
    assert body["skill"]["enabled"] is False  # drafts never arm themselves
    assert body["confidence"] < 0.5


async def test_import_leaves_skills_disabled_by_default(client) -> None:
    """Someone else's thresholds are almost never right for your house."""
    http, _, _ = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)

    import yaml

    body = yaml.safe_dump(CLUTTER_SKILL)
    response = await http.post(
        "/api/v1/skills/import", content=body, headers={"Content-Type": "text/plain"}
    )
    assert response.json()["imported"] == ["kitchen-clutter-buster"]
    listed = (await http.get("/api/v1/skills")).json()
    assert listed[0]["enabled"] is False


# ------------------------------------------------------------------ the vision plan


async def test_plan_is_empty_until_a_skill_is_enabled(client) -> None:
    """The claim in the README: disable the skills and the skill detectors stop consuming CPU.

    Identity is ambient context, not a skill: with identity on (the default) the plan still
    carries the face_id detector, but no skill detectors run until a skill is enabled.
    """
    http, _, _ = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)

    plan = (await http.get("/api/v1/vision/plan")).json()
    identity_detectors = [d for d in plan["anchors"][0]["detectors"] if d["detector"] != "face_id"]
    assert identity_detectors == []
    assert any(d["detector"] == "face_id" for d in plan["anchors"][0]["detectors"])

    await http.post("/api/v1/skills", json=CLUTTER_SKILL)
    plan = (await http.get("/api/v1/vision/plan")).json()
    anchor_plan = plan["anchors"][0]
    assert [d["detector"] for d in anchor_plan["detectors"] if d["detector"] != "face_id"] == [
        "clutter_score"
    ]
    clutter = next(d for d in anchor_plan["detectors"] if d["detector"] == "clutter_score")
    assert clutter["wanted_signals"] == ["clutter_level"]
    assert anchor_plan["snapshot_attach"] is True


async def test_plan_revision_changes_when_skills_change(client) -> None:
    http, _, _ = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    before = (await http.get("/api/v1/vision/plan")).json()["revision"]

    await http.post("/api/v1/skills", json=CLUTTER_SKILL)
    after = (await http.get("/api/v1/vision/plan")).json()["revision"]
    assert before != after


async def test_plan_takes_the_strictest_snapshot_policy(client) -> None:
    """One skill wanting ephemeral snapshots makes the whole anchor ephemeral."""
    http, _, _ = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=CLUTTER_SKILL)

    private = {
        **CLUTTER_SKILL,
        "id": "private-metric",
        "effect": {
            "type": "metric",
            "metric": "counter_busy_minutes",
            "aggregation": "duration_minutes",
        },
        "resolve": {"conditions": {"signal": "clutter", "op": "lte", "value": 0.25, "for": "1m"}},
        "snapshot": {"attach": False, "mode": "ephemeral"},
    }
    assert (await http.post("/api/v1/skills", json=private)).status_code == 201

    plan = (await http.get("/api/v1/vision/plan")).json()
    assert plan["anchors"][0]["snapshot_attach"] is False
    assert plan["anchors"][0]["snapshot_mode"] == "ephemeral"


# ------------------------------------------------------------------ the full loop


async def test_full_loop_mess_to_task_to_resolution(client) -> None:
    """Drive the real engine with synthetic observations, end to end.

    Mess appears, is sustained past the trigger window, a task is created with the snapshot from the
    observation that caused it - and when the counter clears, the task closes itself.
    """
    http, state, config = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=CLUTTER_SKILL)

    from openhup.engine import Engine
    from openhup.llm.render import PLAIN, PersonalityRenderer
    from openhup.notify import Dispatcher, build_channels
    from openhup.skills.window import Sample

    engine = Engine(
        settings=config,
        bus=state.bus,
        renderer=PersonalityRenderer(None, personalities={"plain": PLAIN}),
        dispatcher=Dispatcher(channels=build_channels({})),
    )
    await engine.load()
    engine.is_leader = True
    assert engine.compiled, "the enabled skill should have compiled"

    key = next(iter(engine.windows.tracked_keys()))
    # A fixed clock, injected into the engine, so the test asserts about logic rather than timing.
    base = datetime.now(tz=UTC) - timedelta(hours=1)
    messy_until = base + timedelta(minutes=20)

    # Twenty minutes of mess, one sample a minute, ending exactly at `messy_until`.
    for minute in range(21):
        engine.windows.get(key).append(Sample(ts=base + timedelta(minutes=minute), value=0.8))
    engine.context_cache["kitchen.counter"] = {
        "snapshot_ref": "snap://2026/08/17/kitchen.counter/before.jpg",
        "objects": ("cup", "cereal box"),
    }
    await engine._evaluate(now=messy_until)

    tasks = (await http.get("/api/v1/tasks")).json()
    assert len(tasks) == 1, tasks
    task = tasks[0]
    assert task["state"] == "open"
    assert task["plain_text"] == "Clear the kitchen counter."
    assert task["before_snapshot"].endswith("before.jpg")
    assert task["anchor_label"] == "Kitchen counter"

    # Evaluating again must not produce a second task for the same mess.
    await engine._evaluate(now=messy_until)
    assert len((await http.get("/api/v1/tasks")).json()) == 1

    # Now it gets cleaned: three minutes below the resolve threshold, which is longer than the
    # skill's `for: 1m` resolve window.
    for minute in range(1, 4):
        engine.windows.get(key).append(
            Sample(ts=messy_until + timedelta(minutes=minute), value=0.05)
        )
    engine.context_cache["kitchen.counter"]["snapshot_ref"] = (
        "snap://2026/08/17/kitchen.counter/after.jpg"
    )
    await engine._evaluate(now=messy_until + timedelta(minutes=3))

    done = (await http.get("/api/v1/tasks?state=done")).json()
    assert len(done) == 1
    assert done[0]["state"] == "resolved_auto"
    assert done[0]["after_snapshot"].endswith("after.jpg")


async def test_no_task_for_a_brief_mess(client) -> None:
    """`for: 5m` exists so a mug put down and picked up again is not a task."""
    http, state, config = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=CLUTTER_SKILL)

    from openhup.engine import Engine
    from openhup.llm.render import PLAIN, PersonalityRenderer
    from openhup.notify import Dispatcher, build_channels
    from openhup.skills.window import Sample

    engine = Engine(
        settings=config,
        bus=state.bus,
        renderer=PersonalityRenderer(None, personalities={"plain": PLAIN}),
        dispatcher=Dispatcher(channels=build_channels({})),
    )
    await engine.load()
    engine.is_leader = True

    key = next(iter(engine.windows.tracked_keys()))
    now = datetime.now(tz=UTC)
    # Two minutes of mess against a `for: 5m` trigger.
    for seconds in (120, 60, 0):
        engine.windows.get(key).append(Sample(ts=now - timedelta(seconds=seconds), value=0.9))
    await engine._evaluate(now=now)
    assert (await http.get("/api/v1/tasks")).json() == []


async def test_stale_data_reports_a_problem_rather_than_silence(client) -> None:
    """A dead camera must not look like a tidy house."""
    http, _, _ = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=CLUTTER_SKILL)

    health = (await http.get("/api/v1/system/health")).json()
    assert health["status"] == "degraded"
    assert any("no frames recently" in problem for problem in health["problems"])


# ------------------------------------------------------------------ task actions


async def test_task_actions(client) -> None:
    http, _state, _config = client
    await http.post("/api/v1/cameras", json=CAMERA)
    await http.post("/api/v1/anchors", json=ANCHOR)
    await http.post("/api/v1/skills", json=CLUTTER_SKILL)

    async with session_scope() as session:
        from openhup.db import TaskRow

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

    task_id = "01K3XQ8V4W7YB2M9C6NZ0PRSTA"
    assert (await http.get("/api/v1/tasks/next")).json()["id"] == task_id

    started = await http.patch(f"/api/v1/tasks/{task_id}", json={"action": "start"})
    assert started.json()["state"] == "in_progress"

    snoozed = await http.patch(f"/api/v1/tasks/{task_id}", json={"action": "snooze", "minutes": 90})
    assert snoozed.json()["state"] == "snoozed"
    # A snoozed task is not the next thing to do.
    assert (await http.get("/api/v1/tasks/next")).json() is None

    marked = await http.patch(
        f"/api/v1/tasks/{task_id}",
        json={"action": "false_positive", "note": "that is a fruit bowl, it lives there"},
    )
    assert marked.json()["state"] == "dismissed"


async def test_task_list_carries_no_total_count(client) -> None:
    """Deliberate: a number next to unfinished work is the fastest way to lose a user."""
    http, _, _ = client
    response = await http.get("/api/v1/tasks")
    assert isinstance(response.json(), list)
    assert "total" not in response.text


# ------------------------------------------------------------------ personalities and system


async def test_personality_preview_shows_that_alerts_stay_plain(client) -> None:
    http, state, _ = client
    from openhup_schemas import Personality

    goblin = Personality(
        id="chaos_goblin",
        display_name="Chaos Goblin",
        intensity=4,
        templates={"task": "The {anchor} has grown a civilisation. Evict it."},
    )
    state.personalities["chaos_goblin"] = goblin
    state.renderer.personalities = state.personalities

    preview = (await http.post("/api/v1/personalities/chaos_goblin/preview")).json()
    assert "civilisation" in preview["task"]
    # The alert sample must be factual regardless of personality.
    assert preview["alert"].startswith("Front burner still on:")
    assert "always read like" in preview["note"]


async def test_system_info_reports_the_privacy_posture(client) -> None:
    http, _, _ = client
    info = (await http.get("/api/v1/system/info")).json()
    assert info["llm"]["remote_allowed"] is False
    assert info["llm"]["redaction_profile"] == "text_only"
    assert info["ux"]["hide_task_counts"] is True


async def test_snapshot_path_traversal_is_refused(client) -> None:
    http, _, _ = client
    response = await http.get("/api/v1/snapshots/../../../etc/passwd")
    assert response.status_code in {400, 404}


async def test_weekly_report_has_no_shame_metrics(client) -> None:
    http, _, _ = client
    report = (await http.get("/api/v1/metrics/report/weekly")).json()
    assert "nag_index" in report
    assert "tasks_missed" not in report
    assert "overdue" not in str(report)
    assert report["plain_summary"].startswith("This week:")


def test_example_skills_load_against_the_api_schema() -> None:
    """The shipped examples must satisfy the same validation the API applies."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples/skills/kitchen-clutter-buster.yaml"
    skill = load_skill_yaml(path.read_text())
    assert skill.id == "kitchen-clutter-buster"


def test_observation_round_trip() -> None:
    observation = Observation(
        source=ObservationSource(camera_id="kitchen", anchor_id="kitchen.counter"),
        detector=DetectorInfo(name="clutter_score", version="test@1"),
        signals=[Signal(key="clutter_level", kind=SignalKind.SCALAR, value=0.72)],
    )
    again = Observation.model_validate_json(observation.model_dump_json(by_alias=True))
    assert again.signal("clutter_level").value == pytest.approx(0.72)
