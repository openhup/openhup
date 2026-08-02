"""External sensors (MQTT, Zigbee2MQTT, Home Assistant) normalised into the same pipeline.

A lid contact switch answers "is the trash open" far more cheaply than a camera, and skills should
not have to care which one answered. This module feeds external values into the shared
`SensorFeed`, which the `sensor` detector reads through exactly the same observation path as
camera data.

The feed itself is a thread-safe single-slot store: MQTT callbacks (a paho network thread) write to
it, the asyncio loop reads from it, and neither ever blocks the other. Values are only surfaced
when they actually change, so a door sensor that has been closed for three hours does not re-publish
"closed" three times a second.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

from openhup_schemas import Signal, SignalKind

log = logging.getLogger(__name__)


#: MQTT topic → sensor binding. `key` is the signal name the skill reads; `kind` is how the payload
#: is parsed and typed.
@dataclass(frozen=True, slots=True)
class SensorBinding:
    topic: str
    anchor_id: str
    key: str
    kind: SignalKind


def coerce_sensor_value(raw: str | bytes, kind: SignalKind) -> float | int | bool | str | list[str]:
    """Parse an MQTT payload into the value shape its `kind` declares.

    Sensors are dumb: they say "1" or "ON" or "22.4". This is where that becomes a typed signal.
    Raises ValueError when the payload cannot be read as the declared kind, so a misconfigured
    topic fails loudly in the log rather than silently feeding nonsense into the skill engine.
    """
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    text = text.strip()

    if kind is SignalKind.SCALAR:
        return float(text)
    if kind is SignalKind.COUNT:
        return int(float(text))
    if kind is SignalKind.BOOLEAN:
        lowered = text.lower()
        if lowered in {"true", "on", "yes", "1", "open", "detected"}:
            return True
        if lowered in {"false", "off", "no", "0", "closed", "clear"}:
            return False
        raise ValueError(f"cannot parse {text!r} as a boolean")
    if kind is SignalKind.ENUM:
        return text
    if kind is SignalKind.SET:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (ValueError, TypeError):
            pass
        return [part.strip() for part in text.split(",") if part.strip()]
    raise ValueError(f"unsupported sensor kind {kind!r}")


class SensorFeed:
    """Thread-safe store of the latest value per (anchor, signal key)."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], tuple[SignalKind, Any]] = {}
        self._dirty: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def set_value(self, anchor_id: str, key: str, kind: SignalKind, value: Any) -> bool:
        """Record a value. Returns True only when it actually changed from the stored one."""
        with self._lock:
            previous = self._values.get((anchor_id, key))
            if previous is not None and previous[1] == value:
                return False
            self._values[(anchor_id, key)] = (kind, value)
            self._dirty.add((anchor_id, key))
            return True

    def take(self, anchor_id: str, key: str) -> Signal | None:
        """The changed value as a Signal, or None when nothing new is pending.

        Taking clears the pending flag, so an unchanged sensor is read once and then goes quiet.
        """
        with self._lock:
            if (anchor_id, key) not in self._dirty:
                return None
            self._dirty.discard((anchor_id, key))
            kind, value = self._values[(anchor_id, key)]
            return Signal(key=key, kind=kind, value=value)

    def latest(self, anchor_id: str, key: str) -> Signal | None:
        """The most recent value regardless of whether it has been consumed, or None."""
        with self._lock:
            entry = self._values.get((anchor_id, key))
            if entry is None:
                return None
            kind, value = entry
            return Signal(key=key, kind=kind, value=value)

    def drain(self) -> list[tuple[str, Signal]]:
        """Every changed value since the last drain, as (anchor_id, Signal), oldest first."""
        out: list[tuple[str, Signal]] = []
        with self._lock:
            for anchor_id, key in list(self._dirty):
                self._dirty.discard((anchor_id, key))
                kind, value = self._values[(anchor_id, key)]
                out.append((anchor_id, Signal(key=key, kind=kind, value=value)))
        return out


class SensorMqtt:
    """Subscribe to sensor topics and hand parsed values to a `SensorFeed`.

    Mirrors `FrigateSource`: paho runs its own network thread and calls the feed through the lock,
    so a slow broker never stalls the asyncio loop. No bindings means no connection is opened.
    """

    def __init__(
        self,
        feed: SensorFeed,
        bindings: list[SensorBinding],
        *,
        host: str,
        port: int = 1883,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.feed = feed
        self.bindings = bindings
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self._client: object | None = None

    def start(self) -> None:
        if not self.bindings:
            return
        import paho.mqtt.client as mqtt

        topics = sorted({binding.topic for binding in self.bindings})
        by_topic: dict[str, list[SensorBinding]] = {}
        for binding in self.bindings:
            by_topic.setdefault(binding.topic, []).append(binding)

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if self.username:
            client.username_pw_set(self.username, self.password)

        def on_connect(_client, _userdata, _flags, reason_code, _properties) -> None:  # type: ignore[no-untyped-def]
            if reason_code != 0:
                log.warning("sensor MQTT connect failed: %s", reason_code)
                return
            for topic in topics:
                client.subscribe(topic)
            log.info(
                "sensor MQTT subscribed to %d topic(s) on %s:%d",
                len(topics),
                self.host,
                self.port,
            )

        def on_message(_client, _userdata, message) -> None:  # type: ignore[no-untyped-def]
            for binding in by_topic.get(str(message.topic), []):
                try:
                    value = coerce_sensor_value(message.payload, binding.kind)
                except (ValueError, TypeError) as exc:
                    log.warning("sensor %s/%s: %s", binding.anchor_id, binding.key, exc)
                    continue
                self.feed.set_value(binding.anchor_id, binding.key, binding.kind, value)

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


__all__ = [
    "SensorBinding",
    "SensorFeed",
    "SensorMqtt",
    "coerce_sensor_value",
]
