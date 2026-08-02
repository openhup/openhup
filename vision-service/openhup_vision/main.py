"""The vision service loop.

    plan pull → per-camera source → sampler gate → detector graph → observation → bus

One asyncio task per camera. Decoding happens on source threads (PyAV and OpenCV both release the
GIL during I/O), while detection runs in a thread pool so a slow ONNX call cannot stall the event
loop or the plan refresh.

Nothing here decides what is worth looking at. That comes from the plan, which comes from the
enabled skills (ADR-003).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
from openhup_schemas import Anchor, SensorReading, Signal, SignalKind, Topic, VisionCommand

from . import detectors as detector_impls
from .agent_server import AgentServer
from .backends import ModelRegistry, ModelUnavailable, SessionCache
from .config import DEFAULT_CONFIG_PATHS, AnchorPlan, VisionPlan, VisionSettings
from .emitter import ObservationEmitter, SnapshotPolicy, SnapshotStore
from .fusion import Weights
from .roi import Region, region_from_anchor
from .sampler import AnchorSampler, Cadence
from .sensor_feed import SensorBinding, SensorFeed, SensorMqtt

log = logging.getLogger("openhup.vision")
UTC = UTC


@dataclass
class AnchorRuntime:
    """Per-anchor state: geometry, sampler, baseline, and the plan that produced it."""

    plan: AnchorPlan
    region: Region
    subregions: tuple[Region, ...]
    sampler: AnchorSampler
    anchor: Anchor | None = None
    baseline: object | None = None
    errors: int = 0

    @property
    def policy(self) -> SnapshotPolicy:
        from datetime import timedelta

        return SnapshotPolicy(
            attach=self.plan.snapshot_attach,
            mode=self.plan.snapshot_mode,
            retention=self.plan.snapshot_retention or timedelta(days=7),
            redact=tuple(self.plan.snapshot_redact),
        )


@dataclass
class Service:
    settings: VisionSettings
    plan: VisionPlan | None = None
    sessions: SessionCache | None = None
    emitter: ObservationEmitter | None = None
    registry: dict[str, object] = field(default_factory=dict)
    runtimes: dict[str, AnchorRuntime] = field(default_factory=dict)
    #: `camera_id` → PushSource, for frames submitted over HTTP by camera-agents.
    agent_sources: dict[str, object] = field(default_factory=dict)
    _agent_server: AgentServer | None = None
    _frigate_source: object | None = None
    sensor_feed: SensorFeed | None = None
    _redis: object | None = None
    _stopping: asyncio.Event = field(default_factory=asyncio.Event)
    _started_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    # -- setup ------------------------------------------------------------------------

    async def setup(self) -> None:
        model_registry = ModelRegistry.load(self.settings.inference.registry)
        if self.settings.inference.model_dir:
            model_registry.directory = self.settings.inference.model_dir  # type: ignore[assignment]

        self.sessions = SessionCache(
            model_registry,
            providers=tuple(self.settings.inference.providers) or None,
            allow_unverified=not self.settings.inference.require_verified_models,
        )
        os.environ.setdefault("OPENHUP_ORT_THREADS", str(self.settings.inference.threads))
        self.sensor_feed = SensorFeed()
        self.registry = detector_impls.build_registry(self.sessions, sensor_feed=self.sensor_feed)

        store = SnapshotStore(
            self.settings.snapshots.directory, quality=self.settings.snapshots.jpeg_quality
        )
        self._redis = await self._connect_bus()
        self.emitter = ObservationEmitter(store=store, redis=self._redis)

        log.info(
            "node=%s provider=%s detectors=%s",
            self.settings.node_id,
            self.sessions.chosen_provider(),
            ",".join(sorted(self.registry)),
        )
        if self.settings.dry_run:
            log.warning("dry-run: detectors will run but no observations will be published")

    async def _connect_bus(self) -> object | None:
        if self.settings.dry_run:
            return None
        try:
            from redis.asyncio import Redis

            client = Redis.from_url(self.settings.bus.url, decode_responses=True)
            await client.ping()
            return client
        except Exception as exc:
            log.error("bus unavailable (%s); running without publishing", exc)
            return None

    # -- plan -------------------------------------------------------------------------

    async def refresh_plan(self) -> bool:
        """Pull the detector plan. Returns True when it changed.

        On failure the previous plan is kept: a backend restart must not blind the cameras.
        """
        token = os.environ.get(self.settings.api_token_env, "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.settings.backend_url}/api/v1/vision/plan",
                    headers=headers,
                    params={"node": self.settings.node_id},
                )
                response.raise_for_status()
                fresh = VisionPlan.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            if self.plan is None:
                log.error("cannot fetch initial plan: %s", exc)
            else:
                log.warning(
                    "plan refresh failed (%s); keeping revision %s", exc, self.plan.revision
                )
            return False

        if self.plan is not None and fresh.revision == self.plan.revision:
            return False

        log.info(
            "plan %s: %d active anchor(s), detectors: %s",
            fresh.revision,
            fresh.active_anchor_count,
            ",".join(sorted(fresh.detector_names())) or "none",
        )
        self.plan = fresh
        self._rebuild_runtimes()
        return True

    def _rebuild_runtimes(self) -> None:
        assert self.plan is not None
        sampling = self.settings.sampling
        cadence = Cadence(
            active=sampling.active_interval,
            idle=sampling.idle_interval,
            dormant=sampling.dormant_interval,
            settle=sampling.settle_after,
            hibernate=sampling.hibernate_after,
            heartbeat=sampling.heartbeat,
        )

        surviving: dict[str, AnchorRuntime] = {}
        for anchor_plan in self.plan.anchors:
            if anchor_plan.idle:
                continue
            configured = next(
                (a for a in self.settings.anchors if a.id == anchor_plan.anchor_id), None
            )
            region = (
                region_from_anchor(anchor_plan.anchor_id, anchor_plan.label, configured.polygon)
                if configured
                else Region(id=anchor_plan.anchor_id, label=anchor_plan.label, points=())
            )
            subregions = (
                tuple(
                    region_from_anchor(sub.id, sub.label, sub.polygon)
                    for sub in (configured.ordered_subregions() if configured else [])
                )
                if anchor_plan.score_subregions
                else ()
            )

            previous = self.runtimes.get(anchor_plan.anchor_id)
            sampler = (
                previous.sampler
                if previous
                else AnchorSampler(
                    anchor_id=anchor_plan.anchor_id,
                    cadence=cadence,
                    motion_threshold=sampling.motion_threshold,
                    region=region,
                )
            )
            sampler.force_next()  # the plan changed; do not wait out the old interval

            surviving[anchor_plan.anchor_id] = AnchorRuntime(
                plan=anchor_plan,
                region=region,
                subregions=subregions,
                sampler=sampler,
                anchor=configured,
                baseline=previous.baseline if previous else None,
            )

        dropped = set(self.runtimes) - set(surviving)
        if dropped:
            log.info("anchors no longer watched: %s", ", ".join(sorted(dropped)))
        self.runtimes = surviving

    # -- the loop ---------------------------------------------------------------------

    async def run(self) -> int:
        await self.setup()
        if not await self.refresh_plan() and self.plan is None:
            log.error("no plan available; exiting")
            return 1

        tasks = [
            asyncio.create_task(self._camera_loop(camera), name=f"camera-{camera.id}")
            for camera in self.settings.enabled_cameras()
            if camera.kind != "frigate"
        ]

        frigate = self._setup_frigate()
        if frigate is not None:
            tasks.append(asyncio.create_task(self._frigate_loop(frigate), name="frigate"))

        sensors = self._setup_sensors()
        if sensors is not None:
            tasks.append(asyncio.create_task(self._sensor_loop(), name="sensors"))

        if not tasks:
            log.error("no cameras configured for this node")
            return 1

        tasks.append(asyncio.create_task(self._plan_loop(), name="plan"))
        tasks.append(asyncio.create_task(self._reaper_loop(), name="reaper"))
        tasks.append(asyncio.create_task(self._command_loop(), name="commands"))

        if self.agent_sources and self.settings.agent.enabled:
            token = os.environ.get(self.settings.agent.token_env, "") or None
            self._agent_server = AgentServer(
                self._agent_submit,
                host=self.settings.agent.host,
                port=self.settings.agent.port,
                token=token,
            )
            await self._agent_server.start()

        await self._stopping.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._agent_server is not None:
            await self._agent_server.stop()
        if self._redis is not None:
            with contextlib.suppress(Exception):
                await self._redis.aclose()  # type: ignore[attr-defined]
        log.info("stopped")
        return 0

    async def _camera_loop(self, camera: object) -> None:
        from .sources import build_source

        try:
            source = build_source(camera)
        except ValueError as exc:
            log.error("%s", exc)
            return

        source.start()
        camera_id = str(camera.id)
        if str(getattr(camera, "kind", "")) == "agent_push":
            self.agent_sources[camera_id] = source
        log.info("camera %s: capture started", camera_id)
        try:
            while not self._stopping.is_set():
                captured = source.latest()
                if captured is None:
                    await asyncio.sleep(0.5)
                    continue

                now = datetime.now(tz=UTC)
                for runtime in self._runtimes_for(camera_id):
                    decision = runtime.sampler.consider(captured.frame, now)
                    if not decision.run:
                        continue
                    await self._detect(camera_id, runtime, captured, now)
                # Poll at roughly twice the fastest cadence; the sampler enforces the real interval.
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        finally:
            source.stop()
            log.info("camera %s: capture stopped (%s)", camera_id, source.stats)

    def _runtimes_for(self, camera_id: str) -> list[AnchorRuntime]:
        return [r for r in self.runtimes.values() if r.plan.camera_id == camera_id]

    async def _agent_submit(self, camera_id: str, jpeg: bytes) -> None:
        """Hand a camera-agent's JPEG to the right PushSource.

        Raises KeyError for an unknown camera (→ 404) and lets ValueError from a bad JPEG
        propagate (→ 400). The source itself is a single-slot buffer, so a flood of frames simply
        overwrites the newest one - exactly the drop-don't-queue behaviour every source shares.
        """
        source = self.agent_sources.get(camera_id)
        if source is None:
            raise KeyError(camera_id)
        source.submit(jpeg)  # type: ignore[attr-defined]

    # -- frigate ---------------------------------------------------------------------

    def _setup_frigate(self) -> object | None:
        """Bridge every anchor on a Frigate camera to the MQTT detection feed.

        Frigate cameras have no frame source of their own: their `object_inventory` signals arrive
        over MQTT, translated from Frigate's own detections. Returns None when nothing to bridge,
        so the broker is never touched unless a camera is actually configured `kind: frigate`.
        """
        from .sources import FrigateBridge, FrigateSource

        assert self.plan is not None
        bridges: list[FrigateBridge] = []
        for camera in self.settings.enabled_cameras():
            if camera.kind != "frigate":
                continue
            frigate_camera = camera.frigate_camera or camera.id
            for anchor_plan in self.plan.anchors:
                if anchor_plan.camera_id != camera.id or anchor_plan.idle:
                    continue
                bridges.append(
                    FrigateBridge(
                        camera_id=camera.id,
                        frigate_camera=frigate_camera,
                        anchor_id=anchor_plan.anchor_id,
                    )
                )
        if not bridges:
            return None

        mqtt = self.settings.mqtt
        password = os.environ.get(mqtt.password_env) if mqtt.password_env else None
        source = FrigateSource(
            bridges,
            host=mqtt.host,
            port=mqtt.port,
            topic=mqtt.frigate_topic,
            username=mqtt.username,
            password=password,
        )
        source.start()
        self._frigate_source = source
        log.info(
            "frigate: %d bridge(s) via %s:%d (%s)",
            len(bridges),
            mqtt.host,
            mqtt.port,
            mqtt.frigate_topic,
        )
        return source

    async def _frigate_loop(self, source: object) -> None:
        try:
            while not self._stopping.is_set():
                for translated in source.drain():  # type: ignore[attr-defined]
                    await self._publish_translated(translated)
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        finally:
            source.stop()  # type: ignore[attr-defined]

    async def _publish_translated(self, translated: dict) -> None:
        """Publish a Frigate-origin observation, filtered to the signals the plan wants."""
        assert self.emitter is not None
        anchor_id = str(translated["anchor_id"])
        camera_id = str(translated["camera_id"])
        detector = str(translated["detector"])
        signals = list(translated["signals"])

        wanted = self._wanted_signals(anchor_id, detector)
        if wanted:
            signals = [s for s in signals if s.key in wanted]
        if not signals:
            return

        observation = self.emitter.build(
            camera_id=camera_id,
            anchor_id=anchor_id,
            detector=detector,
            detector_version=f"frigate@{self.settings.node_id}",
            backend="frigate",
            signals=signals,
            frame=None,
            policy=None,
            now=datetime.now(tz=UTC),
        )
        if self.settings.dry_run:
            log.info("[dry-run] %s/%s %s", anchor_id, detector, {s.key: s.value for s in signals})
            return
        await self.emitter.publish(observation)

    def _wanted_signals(self, anchor_id: str, detector: str) -> set[str]:
        assert self.plan is not None
        for anchor_plan in self.plan.anchors:
            if anchor_plan.anchor_id != anchor_id:
                continue
            for detector_plan in anchor_plan.detectors:
                if detector_plan.detector == detector:
                    return set(detector_plan.wanted_signals)
        return set()

    # -- external sensors ------------------------------------------------------------

    def _setup_sensors(self) -> object | None:
        """Subscribe to every MQTT topic a `sensor` binding in the plan declares.

        Sensor values arrive over MQTT and are published directly (camera-independent), so a door
        contact switch is not gated on the camera frame loop. Bindings are fixed at startup; a
        skill that changes a sensor topic takes effect on the next plan-triggered restart.
        """
        assert self.plan is not None and self.sensor_feed is not None
        bindings: list[SensorBinding] = []
        for anchor_plan in self.plan.anchors:
            for detector_plan in anchor_plan.detectors:
                if detector_plan.detector != "sensor":
                    continue
                topic = detector_plan.params.get("topic")
                if not topic:
                    continue
                try:
                    kind = SignalKind(str(detector_plan.params.get("kind", "scalar")))
                except ValueError:
                    log.warning(
                        "anchor %s: unknown sensor kind %r; skipping",
                        anchor_plan.anchor_id,
                        detector_plan.params.get("kind"),
                    )
                    continue
                key = str(detector_plan.params.get("emit_as") or "sensor")
                bindings.append(
                    SensorBinding(
                        topic=str(topic),
                        anchor_id=anchor_plan.anchor_id,
                        key=key,
                        kind=kind,
                    )
                )
        if not bindings:
            return None

        mqtt = self.settings.mqtt
        password = os.environ.get(mqtt.password_env) if mqtt.password_env else None
        source = SensorMqtt(
            self.sensor_feed,
            bindings,
            host=mqtt.host,
            port=mqtt.port,
            username=mqtt.username,
            password=password,
        )
        source.start()
        log.info("sensors: %d binding(s) via %s:%d", len(bindings), mqtt.host, mqtt.port)
        return source

    async def _sensor_loop(self) -> None:
        """Publish changed sensor values promptly, independent of the camera frame loop."""
        assert self.emitter is not None and self.sensor_feed is not None
        while not self._stopping.is_set():
            for anchor_id, sig in self.sensor_feed.drain():
                await self._publish_sensor(anchor_id, sig)
            await asyncio.sleep(0.2)

    async def _publish_sensor(self, anchor_id: str, sig: Signal) -> None:
        assert self.emitter is not None
        reading = SensorReading(
            anchor_id=anchor_id,
            key=sig.key,
            kind=sig.kind,
            value=sig.value,
            origin="mqtt",
        )
        if self.settings.dry_run:
            log.info("[dry-run] sensor %s/%s = %s", anchor_id, sig.key, sig.value)
            return
        await self.emitter.publish(reading.to_observation())

    async def _detect(
        self, camera_id: str, runtime: AnchorRuntime, captured: object, now: datetime
    ) -> None:
        """Run every planned detector for one anchor and publish the results."""
        assert self.emitter is not None and self.sessions is not None
        weights = (
            Weights(
                baseline_diff=runtime.anchor.clutter_weights.baseline_diff,
                object_density=runtime.anchor.clutter_weights.object_density,
                semantic=runtime.anchor.clutter_weights.semantic,
            )
            if runtime.anchor
            else Weights()
        )

        for plan in runtime.plan.detectors:
            implementation = self.registry.get(plan.detector)
            if implementation is None:
                if plan.detector in detector_impls.NOT_YET_IMPLEMENTED:
                    log.warning(
                        "anchor %s wants %s, which is declared but not implemented in this build",
                        runtime.plan.anchor_id,
                        plan.detector,
                    )
                self.registry[plan.detector] = None  # type: ignore[assignment]
                continue

            gallery = tuple(
                (m["id"], m["embedding"]) for m in (self.plan.members if self.plan else [])
            )
            context = detector_impls.DetectorContext(
                anchor_id=runtime.plan.anchor_id,
                anchor_label=runtime.plan.label,
                region=runtime.region,
                params=dict(plan.params),
                baseline=runtime.baseline,  # type: ignore[arg-type]
                sensitivity=runtime.anchor.sensitivity if runtime.anchor else 0.5,
                clutter_weights=weights,
                subregions=runtime.subregions,
                gallery=gallery,
            )

            started = time.perf_counter()
            try:
                result = await asyncio.to_thread(
                    implementation.detect,
                    captured.frame,
                    context,  # type: ignore[attr-defined]
                )
            except ModelUnavailable as exc:
                runtime.errors += 1
                if runtime.errors in (1, 10, 100):  # log once, then rarely
                    log.warning("%s on %s: %s", plan.detector, runtime.plan.anchor_id, exc)
                continue
            except Exception as exc:
                runtime.errors += 1
                log.exception("%s on %s failed: %s", plan.detector, runtime.plan.anchor_id, exc)
                continue

            cost_ms = (time.perf_counter() - started) * 1000
            signals = (
                [s for s in result.signals if s.key in plan.wanted_signals]
                if plan.wanted_signals
                else result.signals
            )
            if not signals:
                continue

            observation = self.emitter.build(
                camera_id=camera_id,
                anchor_id=runtime.plan.anchor_id,
                detector=plan.detector,
                detector_version=self._version_for(plan.detector),
                backend=self.sessions.chosen_provider(),
                signals=signals,
                frame=captured.frame,  # type: ignore[attr-defined]
                policy=runtime.policy,
                redact_boxes=result.redact_boxes,
                frame_seq=captured.sequence,  # type: ignore[attr-defined]
                cost_ms=round(cost_ms, 2),
                now=now,
            )
            if self.settings.dry_run:
                log.info(
                    "[dry-run] %s/%s %s",
                    runtime.plan.anchor_id,
                    plan.detector,
                    {s.key: s.value for s in signals},
                )
                continue
            await self.emitter.publish(observation)

    def _version_for(self, detector: str) -> str:
        """Model identity recorded on every observation, so a metric series stays interpretable
        across a model upgrade that shifts the scale."""
        implementation = self.registry.get(detector)
        models = getattr(implementation, "models", ())
        return f"{'+'.join(models) or detector}@{self.settings.node_id}"

    # -- background tasks -------------------------------------------------------------

    async def _plan_loop(self) -> None:
        interval = self.settings.plan_refresh.total_seconds()
        while not self._stopping.is_set():
            await asyncio.sleep(interval)
            with contextlib.suppress(Exception):
                await self.refresh_plan()

    async def _reaper_loop(self) -> None:
        assert self.emitter is not None
        interval = self.settings.snapshots.reap_interval.total_seconds()
        while not self._stopping.is_set():
            await asyncio.sleep(interval)
            removed = await asyncio.to_thread(self.emitter.store.reap)
            if removed:
                log.info("reaped %d expired snapshot(s)", removed)
            used = self.emitter.store.usage_bytes()
            if used > self.settings.snapshots.max_bytes:
                log.error(
                    "snapshot store is %.1f GiB, over the %.1f GiB limit - shorten retention",
                    used / 1024**3,
                    self.settings.snapshots.max_bytes / 1024**3,
                )

    async def _command_loop(self) -> None:
        """React to backend commands: reload the plan, recapture a baseline, grab a snapshot."""
        if self._redis is None:
            return
        last_id = "$"
        while not self._stopping.is_set():
            try:
                entries = await self._redis.xread(  # type: ignore[attr-defined]
                    {Topic.VISION_COMMANDS.value: last_id}, block=5000, count=10
                )
            except Exception as exc:
                log.warning("command stream error: %s", exc)
                await asyncio.sleep(5)
                continue

            for _stream, messages in entries or []:
                for message_id, fields in messages:
                    last_id = message_id
                    with contextlib.suppress(Exception):
                        command = VisionCommand.model_validate_json(fields.get("payload", "{}"))
                        await self._handle_command(command)

    async def _handle_command(self, command: VisionCommand) -> None:
        log.info("command: %s %s", command.action, command.anchor_id or command.camera_id or "")
        if command.action == "reload_plan":
            self.plan = None if command.args.get("force") else self.plan
            await self.refresh_plan()
        elif command.action == "capture_baseline" and command.anchor_id:
            runtime = self.runtimes.get(command.anchor_id)
            if runtime is not None:
                runtime.sampler.force_next()
                # The backend stores the authoritative baseline; this refreshes the in-memory
                # copy on the next frame so scoring uses the new reference immediately.
                runtime.baseline = None
        elif command.action == "shutdown":
            self._stopping.set()

    def stop(self) -> None:
        self._stopping.set()

    def health(self) -> dict[str, object]:
        return {
            "node_id": self.settings.node_id,
            "uptime_s": int((datetime.now(tz=UTC) - self._started_at).total_seconds()),
            "provider": self.sessions.chosen_provider() if self.sessions else None,
            "plan_revision": self.plan.revision if self.plan else None,
            "anchors": {
                anchor_id: runtime.sampler.stats() for anchor_id, runtime in self.runtimes.items()
            },
            "emitter": self.emitter.stats() if self.emitter else {},
            "models_loaded": self.sessions.loaded() if self.sessions else {},
        }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenHup vision service")
    parser.add_argument("--config", action="append", default=[], help="YAML config (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="detect but publish nothing")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    settings = VisionSettings.load(*(args.config or DEFAULT_CONFIG_PATHS))
    if args.dry_run:
        settings.dry_run = True
    logging.basicConfig(
        level=(args.log_level or settings.log_level).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    service = Service(settings=settings)

    async def main() -> int:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, service.stop)
        return await service.run()

    return asyncio.run(main())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())


__all__ = ["AnchorRuntime", "Service", "run"]
