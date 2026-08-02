"""Shared application state: compiled skills, personalities, anchors, and the WebSocket hub.

The compile cache is the point of this module. Compiling a skill validates it against the detector
registry and the real anchors, which is cheap but not free, and both the API and the vision plan
endpoint need the result on every request. Skills change rarely, so they are compiled on write and
cached, with a revision string the vision service can compare to decide whether to reconfigure.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from openhup_schemas import (
    BUILTIN_DETECTORS,
    Anchor,
    Camera,
    Envelope,
    Personality,
    Skill,
)
from sqlalchemy import select

from ..bus import Bus
from ..core.config import Settings
from ..db import (
    AnchorRow,
    CameraRow,
    PersonalityDrawRow,
    PersonalityRow,
    SkillRow,
    session_scope,
)
from ..llm import PersonalityRenderer, UsageLog
from ..notify import Dispatcher
from ..personality import draw as draw_personality
from ..personality import effective_default_id, load_draw
from ..skills.compile import CompiledSkill, compile_all
from ..voice import VoiceProvider

log = logging.getLogger(__name__)
UTC = UTC


@dataclass
class WebSocketHub:
    """Fan-out to connected browsers.

    Server-side topic filtering, and a dead socket is dropped rather than retried: a phone that went
    to sleep must not be able to slow down task creation.
    """

    connections: dict[Any, set[str]] = field(default_factory=dict)

    async def connect(self, socket: Any, topics: set[str]) -> None:
        self.connections[socket] = topics

    def disconnect(self, socket: Any) -> None:
        self.connections.pop(socket, None)

    async def broadcast(self, envelope: Envelope) -> int:
        family = envelope.type.value.split(".")[0]
        payload = envelope.model_dump(mode="json")
        dead: list[Any] = []
        delivered = 0
        for socket, topics in list(self.connections.items()):
            if topics and family not in topics and "all" not in topics:
                continue
            try:
                await socket.send_json(payload)
                delivered += 1
            except Exception:
                dead.append(socket)
        for socket in dead:
            self.disconnect(socket)
        return delivered


@dataclass
class AppState:
    """Everything request handlers need, assembled once at startup."""

    settings: Settings
    bus: Bus
    usage: UsageLog
    provider: Any
    renderer: PersonalityRenderer
    dispatcher: Dispatcher
    voice: VoiceProvider

    skills: dict[str, Skill] = field(default_factory=dict)
    compiled: dict[str, CompiledSkill] = field(default_factory=dict)
    compile_failures: dict[str, list[str]] = field(default_factory=dict)
    anchors: dict[str, Anchor] = field(default_factory=dict)
    cameras: dict[str, Camera] = field(default_factory=dict)
    personalities: dict[str, Personality] = field(default_factory=dict)
    #: The personality gamble (ADR-014): the drawn voice shadows the configured default until
    #: revealed, re-drawn, or deleted. Loaded at startup, kept current by the API.
    personality_draw: PersonalityDrawRow | None = None
    hub: WebSocketHub = field(default_factory=WebSocketHub)
    #: Changes whenever an enabled skill, an anchor, or an enrolled member changes. The vision
    #: service compares it, so enrollment and deletion re-pull the face gallery (ADR-016).
    plan_revision: str = "0"
    members_revision: int = 0
    loaded_at: datetime | None = None

    # -- loading ------------------------------------------------------------------------

    async def load_registry(self) -> None:
        """Load cameras, anchors, skills, and personalities, then compile.

        Seeds from `config_dir` on an empty database so a fresh install with a cameras.yaml works
        immediately rather than presenting an empty screen.
        """
        async with session_scope() as session:
            cameras = (await session.execute(select(CameraRow))).scalars().all()
            anchors = (await session.execute(select(AnchorRow))).scalars().all()
            skills = (await session.execute(select(SkillRow))).scalars().all()
            personalities = (await session.execute(select(PersonalityRow))).scalars().all()

            if not cameras and not anchors:
                await self._seed_from_disk(session)
                cameras = (await session.execute(select(CameraRow))).scalars().all()
                anchors = (await session.execute(select(AnchorRow))).scalars().all()
            fresh_install = not personalities
            if not personalities:
                await self._seed_personalities(session)
                personalities = (await session.execute(select(PersonalityRow))).scalars().all()

            # The gamble happens once, at first seed, and is never announced - the voice is
            # discovered by living with it (ADR-014).
            if fresh_install and self.settings.personality.gamble and personalities:
                try:
                    await draw_personality(
                        session,
                        pool=self.settings.personality.gamble_pool,
                        available=[row.id for row in personalities],
                    )
                    log.info("personality gamble: a voice was drawn at first setup")
                except ValueError as exc:
                    log.warning("personality gamble skipped: %s", exc)
            self.personality_draw = await load_draw(session)

        self.cameras = {row.id: Camera.model_validate(row.config) for row in cameras}
        self.anchors = {row.id: Anchor.model_validate(row.config) for row in anchors}
        self.personalities = {
            row.id: Personality.model_validate(row.definition) for row in personalities
        }
        self.renderer.personalities = self.personalities
        self._apply_draw()

        self.skills = {}
        for row in skills:
            try:
                self.skills[row.id] = Skill.model_validate(row.definition)
            except Exception as exc:
                self.compile_failures[row.id] = [f"stored definition is invalid: {exc}"]

        self.recompile()
        self.loaded_at = datetime.now(tz=UTC)

    async def _seed_from_disk(self, session: Any) -> None:
        """Import cameras.yaml on first run."""
        path = Path(self.settings.config_dir) / "cameras.yaml"
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except OSError:
            log.info("no %s to seed from; add cameras through the UI", path)
            return

        for entry in raw.get("cameras", []):
            camera = Camera.model_validate(entry)
            session.add(
                CameraRow(
                    id=camera.id,
                    name=camera.name,
                    enabled=camera.enabled,
                    kind=camera.kind.value,
                    config=camera.model_dump(mode="json"),
                )
            )
        for entry in raw.get("anchors", []):
            anchor = Anchor.model_validate(entry)
            session.add(
                AnchorRow(
                    id=anchor.id,
                    camera_id=anchor.camera_id,
                    label=anchor.label,
                    enabled=anchor.enabled,
                    config=anchor.model_dump(mode="json"),
                    baseline_ref=anchor.baseline_ref,
                )
            )
        log.info("seeded %s", path)

    async def _seed_personalities(self, session: Any) -> None:
        """Install the shipped presets. User-created ones are never touched by this."""
        for candidate in (
            Path(self.settings.config_dir) / "personalities.yaml",
            Path("examples/personalities/personalities.yaml"),
        ):
            try:
                entries = yaml.safe_load(candidate.read_text()) or []
            except OSError:
                continue
            for entry in entries:
                personality = Personality.model_validate(entry)
                session.add(
                    PersonalityRow(
                        id=personality.id,
                        display_name=personality.display_name,
                        definition=personality.model_dump(mode="json"),
                        builtin=True,
                    )
                )
            log.info("seeded personalities from %s", candidate)
            return

    # -- personality -------------------------------------------------------------------

    def _apply_draw(self) -> None:
        """Make the drawn personality the effective default, when one exists and is valid.

        Config's `default_personality` stays the operator's choice; the draw shadows it in
        memory. An unknown drawn id (a preset that was removed) falls back to the config default
        rather than breaking the renderer.
        """
        effective = effective_default_id(
            self.settings.personality.default_personality, self.personality_draw
        )
        if effective not in self.personalities:
            effective = self.settings.personality.default_personality
        if self.renderer.settings.default_personality != effective:
            self.renderer.settings = self.renderer.settings.model_copy(
                update={"default_personality": effective}
            )

    def effective_default_personality(self) -> str:
        """What /system/info and the renderer actually use - draw shadows config."""
        effective = effective_default_id(
            self.settings.personality.default_personality, self.personality_draw
        )
        if effective not in self.personalities:
            return self.settings.personality.default_personality
        return effective

    # -- compilation --------------------------------------------------------------------

    def recompile(self) -> None:
        """Recompile every skill and bump the plan revision."""
        compiled, failures = compile_all(
            list(self.skills.values()), registry=BUILTIN_DETECTORS, anchors=self.anchors
        )
        self.compiled = {c.skill.id: c for c in compiled}
        self.compile_failures = {
            skill_id: [f.message for f in error.findings] for skill_id, error in failures.items()
        }
        if failures:
            log.warning(
                "%d skill(s) failed to compile and are not running: %s",
                len(failures),
                ", ".join(sorted(failures)),
            )
        self.plan_revision = self._revision()

    def _revision(self) -> str:
        """Hash of everything the vision service's plan depends on."""
        parts = []
        for skill_id in sorted(self.compiled):
            skill = self.compiled[skill_id].skill
            if skill.enabled:
                parts.append(f"{skill_id}:{skill.version}")
        for anchor_id in sorted(self.anchors):
            anchor = self.anchors[anchor_id]
            parts.append(f"{anchor_id}:{anchor.enabled}:{len(anchor.polygon)}:{anchor.sensitivity}")
        parts.append(f"members:{self.members_revision}")
        digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
        return digest[:12]

    def compile_one(self, skill: Skill) -> CompiledSkill:
        """Compile a single skill, raising `SkillCompileError` for the API to turn into a 422."""
        from ..skills.compile import compile_skill

        return compile_skill(skill, registry=BUILTIN_DETECTORS, anchors=self.anchors)

    def enabled_compiled(self) -> list[CompiledSkill]:
        return [c for c in self.compiled.values() if c.skill.enabled]

    def anchors_for_camera(self, camera_id: str) -> list[Anchor]:
        return [a for a in self.anchors.values() if a.camera_id == camera_id]

    def skills_watching(self, anchor_id: str) -> list[CompiledSkill]:
        return [c for c in self.enabled_compiled() if anchor_id in c.anchor_ids]

    # -- events -------------------------------------------------------------------------

    def wire_bus_to_websockets(self) -> None:
        """Forward in-process events to browsers.

        Only covers events published by this process. Events from the engine arrive over Redis and
        are relayed by `relay_bus_events`.
        """
        self.bus.subscribe_local(lambda envelope: self.hub.broadcast(envelope))

    async def relay_bus_events(self) -> None:
        """Relay engine events from Redis to connected browsers.

        Runs as a background task in the API process. Reads with `$` (new messages only) rather
        than a consumer group: a browser that was not connected does not need the backlog, and the
        UI refetches on reconnect anyway.
        """
        if not self.bus.connected:
            return
        from openhup_schemas import Topic

        streams = {
            Topic.TASK_EVENTS.value: "$",
            Topic.ALERT_EVENTS.value: "$",
            Topic.SKILL_EVENTS.value: "$",
            Topic.SYSTEM_EVENTS.value: "$",
        }
        while True:
            try:
                entries = await self.bus._redis.xread(streams, block=5000, count=100)
            except Exception as exc:
                log.debug("event relay paused: %s", exc)
                await asyncio.sleep(5)
                continue
            for stream, messages in entries or []:
                for message_id, fields in messages:
                    streams[stream] = message_id
                    with contextlib.suppress(Exception):
                        await self.hub.broadcast(_envelope_from_fields(fields))


def _envelope_from_fields(fields: dict[str, str]) -> Envelope:
    import json

    payload = json.loads(fields.get("payload") or "{}").get("payload", {})
    return Envelope(
        id=fields["id"],
        type=fields["type"],  # type: ignore[arg-type]
        ts=fields["ts"],  # type: ignore[arg-type]
        payload=payload,
        skill_id=fields.get("skill_id") or None,
        anchor_id=fields.get("anchor_id") or None,
        episode_id=fields.get("episode_id") or None,
        source=fields.get("source", "engine"),
    )


__all__ = ["AppState", "WebSocketHub"]
