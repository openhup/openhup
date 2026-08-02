"""Frame sources.

Five ways frames arrive, behind one interface. The important shared behaviour is in `FrameSource`:
**frames are dropped, never queued**. A late frame is worthless for deciding whether a counter is
cluttered right now, and a growing queue converts a transient hiccup into an outage that ends with
the OOM killer. Every source therefore keeps only the newest frame.

PyAV and OpenCV are imported lazily so this module - and the pure maths that depends on nothing -
stays importable on a machine with no video stack.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from openhup_schemas import Signal, SignalKind

from .roi import Frame

log = logging.getLogger(__name__)
UTC = UTC


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    frame: Frame
    ts: datetime
    sequence: int
    camera_id: str


@dataclass
class SourceStats:
    frames_decoded: int = 0
    frames_dropped: int = 0
    reconnects: int = 0
    last_frame_at: datetime | None = None
    last_error: str | None = None

    @property
    def healthy(self) -> bool:
        if self.last_frame_at is None:
            return False
        return (datetime.now(tz=UTC) - self.last_frame_at).total_seconds() < 60


class FrameSource(ABC):
    """Base class: a background reader thread and a single-slot latest-frame buffer."""

    def __init__(self, camera_id: str, *, max_fps: float = 5.0) -> None:
        self.camera_id = camera_id
        self.max_fps = max_fps
        self.stats = SourceStats()
        self._latest: CapturedFrame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0

    # -- lifecycle --------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_forever, name=f"source-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> FrameSource:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- reading ----------------------------------------------------------------------

    def latest(self) -> CapturedFrame | None:
        """The newest frame, or None. Never blocks."""
        with self._lock:
            return self._latest

    def _publish(self, frame: Frame) -> None:
        self._sequence += 1
        captured = CapturedFrame(
            frame=frame, ts=datetime.now(tz=UTC), sequence=self._sequence, camera_id=self.camera_id
        )
        with self._lock:
            if self._latest is not None:
                # The previous frame was never consumed. That is normal and expected: the sampler
                # runs at a fraction of the decode rate. Counted, not warned about.
                self.stats.frames_dropped += 1
            self._latest = captured
        self.stats.frames_decoded += 1
        self.stats.last_frame_at = captured.ts

    def _run_forever(self) -> None:
        """Read loop with exponential backoff. A camera that is down must not spin the CPU."""
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._read_loop()
                backoff = 1.0
            except Exception as exc:
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                self.stats.reconnects += 1
                log.warning("%s: %s; reconnecting in %.0fs", self.camera_id, exc, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)

    @abstractmethod
    def _read_loop(self) -> None:
        """Decode frames and call `_publish`. Return to trigger a reconnect."""


class RTSPSource(FrameSource):
    """RTSP via PyAV.

    Two things worth knowing. TCP transport is the default because UDP on wifi produces torn frames
    that look like motion and waste inference. And this should be pointed at the camera's
    *substream*: decoding 4K at 5fps to find a coffee mug is the usual reason a home server melts.
    """

    def __init__(
        self,
        camera_id: str,
        url: str,
        *,
        max_fps: float = 5.0,
        transport: str = "tcp",
        hwaccel: str = "none",
        timeout_s: float = 10.0,
    ) -> None:
        super().__init__(camera_id, max_fps=max_fps)
        self.url = url
        self.transport = transport
        self.hwaccel = hwaccel
        self.timeout_s = timeout_s

    def _read_loop(self) -> None:
        import av  # lazy: keeps this module importable without a video stack

        options = {
            "rtsp_transport": self.transport,
            "stimeout": str(int(self.timeout_s * 1_000_000)),  # microseconds
            "fflags": "nobuffer",
            "flags": "low_delay",
        }
        if self.hwaccel != "none":
            options["hwaccel"] = self.hwaccel

        container = av.open(self.url, options=options, timeout=self.timeout_s)
        try:
            stream = container.streams.video[0]
            # Let FFmpeg drop frames rather than buffering them for us.
            stream.thread_type = "AUTO"
            min_interval = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
            last = 0.0

            for packet in container.demux(stream):
                if self._stop.is_set():
                    return
                for frame in packet.decode():
                    now = time.monotonic()
                    if now - last < min_interval:
                        continue  # rate limit before the expensive ndarray conversion
                    last = now
                    self._publish(frame.to_ndarray(format="bgr24"))
        finally:
            container.close()


class USBSource(FrameSource):
    """USB or CSI camera via OpenCV. Used directly, or inside a camera-agent on another host."""

    def __init__(
        self,
        camera_id: str,
        device: str = "/dev/video0",
        *,
        max_fps: float = 2.0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        super().__init__(camera_id, max_fps=max_fps)
        self.device = device
        self.width = width
        self.height = height

    def _read_loop(self) -> None:
        import cv2

        index: int | str = int(self.device) if str(self.device).isdigit() else self.device
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {self.device}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Depth 1: we want the newest frame, not the oldest.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            interval = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("read failed")
                self._publish(frame)
                self._stop.wait(interval)
        finally:
            capture.release()


class SnapshotURLSource(FrameSource):
    """Periodic still from an HTTP endpoint - ESP32-CAM and similar.

    Fine for binary states like "is the door open". Not useful for clutter scoring: the frame rate
    and quality are too low for the baseline comparison to mean much.
    """

    def __init__(
        self, camera_id: str, url: str, *, max_fps: float = 0.5, timeout_s: float = 10.0
    ) -> None:
        super().__init__(camera_id, max_fps=max_fps)
        self.url = url
        self.timeout_s = timeout_s

    def _read_loop(self) -> None:
        import cv2
        import httpx

        interval = 1.0 / self.max_fps if self.max_fps > 0 else 2.0
        with httpx.Client(timeout=self.timeout_s) as client:
            while not self._stop.is_set():
                response = client.get(self.url)
                response.raise_for_status()
                buffer = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("undecodable image")
                self._publish(frame)
                self._stop.wait(interval)


class PushSource(FrameSource):
    """Frames pushed in by a camera-agent over HTTP.

    For hosts that own a camera but cannot be reached from the vision service - a Pi Zero on wifi,
    anything behind NAT. The agent posts JPEGs; `submit` is called by the HTTP handler, so there is
    no read loop to run.
    """

    def __init__(self, camera_id: str, *, max_fps: float = 2.0) -> None:
        super().__init__(camera_id, max_fps=max_fps)

    def start(self) -> None:  # no background thread needed
        return

    def stop(self) -> None:
        self._stop.set()

    def _read_loop(self) -> None:  # pragma: no cover - never called
        raise NotImplementedError

    def submit(self, jpeg: bytes) -> None:
        import cv2

        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("undecodable JPEG from agent")
        self._publish(frame)


@dataclass
class FrigateBridge:
    """Consume an existing Frigate install's detections over MQTT instead of decoding twice.

    The most practical adoption path for anyone already running Frigate: OpenHup contributes the
    skill engine, tasks, and personality on top of detections Frigate is already computing, at a
    fraction of the CPU. Frigate's detection events are translated straight into observations, so no
    OpenHup detector runs for these cameras at all.

    Frigate answers "what objects are there". It does not answer "is this surface cluttered", so
    clutter skills still need a native source - which is why this is a bridge rather than a source.
    """

    camera_id: str
    frigate_camera: str
    anchor_id: str
    stats: SourceStats = field(default_factory=SourceStats)

    def translate(self, payload: dict) -> dict | None:
        """Frigate MQTT `frigate/events` payload → observation-shaped dict, or None to ignore."""
        after = payload.get("after") or payload.get("before") or {}
        if after.get("camera") != self.frigate_camera:
            return None
        label = after.get("label")
        if not label:
            return None
        self.stats.frames_decoded += 1
        self.stats.last_frame_at = datetime.now(tz=UTC)
        return {
            "camera_id": self.camera_id,
            "anchor_id": self.anchor_id,
            "detector": "object_inventory",
            "signals": [
                Signal(key="objects", kind=SignalKind.SET, value=[label]),
                Signal(key="object_count", kind=SignalKind.COUNT, value=1),
                Signal(
                    key="person_count",
                    kind=SignalKind.COUNT,
                    value=1 if label == "person" else 0,
                ),
            ],
            "score": after.get("top_score", after.get("score")),
        }


class FrigateSource:
    """Subscribe to a Frigate install's MQTT events and hand them to the service loop.

    Frigate publishes object detections on `frigate/events`; this consumes them and turns each one
    into an observation-shaped dict via `FrigateBridge`, so a home already running Frigate can add
    the skill engine without decoding a single stream twice. Runs its own network thread (paho),
    and hands results to the asyncio loop through a thread-safe queue.
    """

    def __init__(
        self,
        bridges: list[FrigateBridge],
        *,
        host: str,
        port: int = 1883,
        topic: str = "frigate/events",
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.bridges = bridges
        self.host = host
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._client: object | None = None

    def start(self) -> None:
        import json

        import paho.mqtt.client as mqtt

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if self.username:
            client.username_pw_set(self.username, self.password)

        def on_connect(_client, _userdata, _flags, reason_code, _properties) -> None:  # type: ignore[no-untyped-def]
            if reason_code != 0:
                log.warning("frigate MQTT connect failed: %s", reason_code)
                return
            client.subscribe(self.topic)
            log.info("subscribed to %s on %s:%d", self.topic, self.host, self.port)

        def on_message(_client, _userdata, message) -> None:  # type: ignore[no-untyped-def]
            try:
                payload = json.loads(message.payload)
            except (ValueError, TypeError):
                return
            for bridge in self.bridges:
                translated = bridge.translate(payload)
                if translated is not None:
                    self.queue.put(translated)

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(self.host, self.port, keepalive=60)
        client.loop_start()
        self._client = client

    def stop(self) -> None:
        if self._client is not None:
            self._client.loop_stop()  # type: ignore[attr-defined]
            self._client.disconnect()  # type: ignore[attr-defined]
            self._client = None

    def drain(self) -> list[dict[str, Any]]:
        """Everything translated since the last drain, oldest first."""
        out: list[dict[str, Any]] = []
        while True:
            try:
                out.append(self.queue.get_nowait())
            except queue.Empty:
                return out


def build_source(camera: object) -> FrameSource:
    """Construct the right source for a `Camera` config object."""
    kind = str(getattr(camera, "kind", "rtsp"))
    camera_id = str(camera.id)
    max_fps = float(getattr(camera, "max_fps", 5.0))

    if kind == "rtsp":
        url = getattr(camera, "detect_url", None) or camera.url
        return RTSPSource(
            camera_id,
            _with_credentials(url, camera),
            max_fps=max_fps,
            transport=str(getattr(camera, "transport", "tcp")),
            hwaccel=str(getattr(camera, "hwaccel", "none")),
        )
    if kind == "usb":
        return USBSource(camera_id, str(camera.device), max_fps=max_fps)
    if kind == "snapshot_url":
        return SnapshotURLSource(camera_id, _with_credentials(camera.url, camera), max_fps=max_fps)
    if kind == "agent_push":
        return PushSource(camera_id, max_fps=max_fps)
    raise ValueError(f"camera {camera_id}: source kind {kind!r} has no native source (frigate?)")


def _with_credentials(url: str, camera: object) -> str:
    """Inject credentials from the environment variable named by `password_env`.

    Passwords live in the environment, never in the camera config, so config files stay safe to
    commit and to paste into a bug report.
    """
    import os
    from urllib.parse import urlsplit, urlunsplit

    username = getattr(camera, "username", None)
    env_name = getattr(camera, "password_env", None)
    if not username or not env_name:
        return url

    password = os.environ.get(env_name)
    if not password:
        log.warning(
            "camera %s: %s is not set in the environment; connecting without a password",
            getattr(camera, "id", "?"),
            env_name,
        )
        return url

    parts = urlsplit(url)
    netloc = f"{username}:{password}@{parts.hostname}"
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


__all__ = [
    "CapturedFrame",
    "FrameSource",
    "FrigateBridge",
    "FrigateSource",
    "PushSource",
    "RTSPSource",
    "SnapshotURLSource",
    "SourceStats",
    "USBSource",
    "build_source",
]
