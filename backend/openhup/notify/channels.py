"""Notification channels.

A channel takes a `NotificationRequest` and delivers it. The interesting behaviour is not in any
individual channel but in the dispatcher above them:

* **Quiet hours hold, they do not drop.** A held notification is recorded as `held`, appears in the
  UI immediately, and is delivered when the window ends. Silently discarding it would be worse than
  waking someone up.
* **High urgency ignores every limit.** Quiet hours, rate limits, and dedupe windows all yield to a
  burner left on. A safety alert that got rate-limited would be an unforgivable failure.
* **Rate limits are per channel.** One miscalibrated skill must not be able to empty a phone
  battery, and the ceiling is low by default.

Channels are built from config, not registered by import side effects, so a broken third-party
channel cannot prevent startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from openhup_schemas import NotificationRequest, TimeWindow, Urgency

log = logging.getLogger(__name__)
UTC = UTC


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    channel: str
    ok: bool
    status: str  # sent | held | failed | suppressed
    detail: str = ""


class Channel(ABC):
    """One delivery target."""

    #: Channels that can carry an image get the snapshot; the rest get a link to it.
    supports_images: bool = False
    #: Config keys without which this channel cannot possibly work. Validated at construction so a
    #: missing `topic` is an error at startup rather than a silent failure at 3am when it matters.
    required_config: tuple[str, ...] = ()

    def __init__(self, channel_id: str, config: dict[str, Any]) -> None:
        self.id = channel_id
        self.config = config
        missing = [key for key in self.required_config if not config.get(key)]
        if missing:
            raise KeyError(
                f"channel {channel_id!r} of type {type(self).__name__} is missing "
                f"required setting(s): {', '.join(missing)}"
            )
        self.enabled = bool(config.get("enabled", True))
        #: Only deliver at or above this urgency. Lets someone put chores on ntfy and safety on SMS.
        self.min_urgency = Urgency(config.get("min_urgency", "info"))

    def accepts(self, urgency: Urgency) -> bool:
        return self.enabled and urgency.rank >= self.min_urgency.rank

    @abstractmethod
    async def send(self, request: NotificationRequest, *, image: bytes | None = None) -> None:
        """Deliver, or raise. The dispatcher records the outcome."""

    async def test(self) -> DeliveryResult:
        """Send a test message. Wired to POST /notify/channels/{id}/test."""
        try:
            await self.send(
                NotificationRequest(
                    channels=[self.id],
                    title="OpenHup test",
                    body="If you can read this, this channel works.",
                    urgency=Urgency.INFO,
                )
            )
        except Exception as exc:
            return DeliveryResult(self.id, False, "failed", str(exc))
        return DeliveryResult(self.id, True, "sent")


class NtfyChannel(Channel):
    """ntfy.sh or a self-hosted ntfy. The recommended default: self-hostable, no account, images."""

    supports_images = True
    required_config = ("topic",)

    async def send(self, request: NotificationRequest, *, image: bytes | None = None) -> None:
        base = self.config.get("url", "https://ntfy.sh").rstrip("/")
        topic = self.config["topic"]
        headers = {
            "Title": request.title,
            "Priority": _ntfy_priority(request.urgency),
            "Tags": self.config.get("tags", "house"),
        }
        if request.link:
            headers["Click"] = f"{self.config.get('ui_url', '').rstrip('/')}{request.link}"
        if token := self.config.get("token"):
            headers["Authorization"] = f"Bearer {token}"
        if request.dedupe_key:
            # ntfy replaces a message with the same id rather than stacking it up.
            headers["X-Message-ID"] = request.dedupe_key[:64]

        async with httpx.AsyncClient(timeout=15) as client:
            if image and self.config.get("attach_images", True):
                headers["Filename"] = "snapshot.jpg"
                headers["Message"] = request.body
                response = await client.put(f"{base}/{topic}", content=image, headers=headers)
            else:
                response = await client.post(
                    f"{base}/{topic}", content=request.body.encode(), headers=headers
                )
            response.raise_for_status()


class WebhookChannel(Channel):
    """POST the whole notification as JSON. The escape hatch for anything not covered."""

    required_config = ("url",)

    async def send(self, request: NotificationRequest, *, image: bytes | None = None) -> None:
        payload = {
            "id": request.id,
            "title": request.title,
            "body": request.body,
            "urgency": request.urgency.value,
            "link": request.link,
            "snapshot_ref": request.snapshot_ref,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                self.config["url"],
                json=payload,
                headers=self.config.get("headers", {}),
            )
            response.raise_for_status()


class DiscordChannel(Channel):
    supports_images = True
    required_config = ("webhook_url",)

    async def send(self, request: NotificationRequest, *, image: bytes | None = None) -> None:
        colour = {"critical": 0xE03131, "high": 0xE8590C, "normal": 0x1971C2}.get(
            request.urgency.value, 0x868E96
        )
        embed = {"title": request.title, "description": request.body, "color": colour}
        async with httpx.AsyncClient(timeout=15) as client:
            if image:
                embed["image"] = {"url": "attachment://snapshot.jpg"}
                response = await client.post(
                    self.config["webhook_url"],
                    data={"payload_json": _json({"embeds": [embed]})},
                    files={"file": ("snapshot.jpg", image, "image/jpeg")},
                )
            else:
                response = await client.post(self.config["webhook_url"], json={"embeds": [embed]})
            response.raise_for_status()


class MatrixChannel(Channel):
    """Matrix via the client-server API. Text only here; image upload needs a media POST first."""

    required_config = ("homeserver", "room_id", "access_token")

    async def send(self, request: NotificationRequest, *, image: bytes | None = None) -> None:
        homeserver = self.config["homeserver"].rstrip("/")
        room = self.config["room_id"]
        token = self.config["access_token"]
        body = f"**{request.title}**\n{request.body}"
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.put(
                f"{homeserver}/_matrix/client/v3/rooms/{room}/send/m.room.message/{request.id}",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "msgtype": "m.text",
                    "body": body,
                    "format": "org.matrix.custom.html",
                    "formatted_body": body.replace("\n", "<br/>"),
                },
            )
            response.raise_for_status()


class MQTTChannel(Channel):
    """Publish to MQTT, for Home Assistant and the rest of a home-automation setup.

    This is the integration surface referred to in ADR-002: MQTT is not the internal bus, but it is
    how OpenHup talks to everything else in a house.
    """

    required_config = ("host",)

    async def send(self, request: NotificationRequest, *, image: bytes | None = None) -> None:
        import json as _json_mod

        try:
            from aiomqtt import Client
        except ImportError as exc:
            raise RuntimeError("MQTT channel needs `aiomqtt` installed") from exc

        topic = self.config.get("topic", "openhup/notifications")
        async with Client(
            hostname=self.config["host"],
            port=int(self.config.get("port", 1883)),
            username=self.config.get("username"),
            password=self.config.get("password"),
        ) as client:
            await client.publish(
                topic,
                _json_mod.dumps(
                    {
                        "title": request.title,
                        "body": request.body,
                        "urgency": request.urgency.value,
                        "link": request.link,
                    }
                ),
                retain=bool(self.config.get("retain", False)),
            )


class SMTPChannel(Channel):
    required_config = ("host", "from", "to")

    async def send(self, request: NotificationRequest, *, image: bytes | None = None) -> None:
        from email.message import EmailMessage

        import aiosmtplib

        message = EmailMessage()
        message["From"] = self.config["from"]
        message["To"] = ", ".join(
            self.config["to"] if isinstance(self.config["to"], list) else [self.config["to"]]
        )
        message["Subject"] = request.title
        message.set_content(request.body)
        if image:
            message.add_attachment(image, maintype="image", subtype="jpeg", filename="snapshot.jpg")

        await aiosmtplib.send(
            message,
            hostname=self.config["host"],
            port=int(self.config.get("port", 587)),
            username=self.config.get("username"),
            password=self.config.get("password"),
            start_tls=bool(self.config.get("start_tls", True)),
        )


class LogChannel(Channel):
    """Writes to the log. The default when nothing is configured, so notifications are never
    silently swallowed during setup."""

    async def send(self, request: NotificationRequest, *, image: bytes | None = None) -> None:
        log.info("[notify:%s] %s - %s", request.urgency.value, request.title, request.body)


CHANNEL_TYPES: dict[str, type[Channel]] = {
    "ntfy": NtfyChannel,
    "webhook": WebhookChannel,
    "discord": DiscordChannel,
    "matrix": MatrixChannel,
    "mqtt": MQTTChannel,
    "smtp": SMTPChannel,
    "log": LogChannel,
}


def build_channels(config: dict[str, dict[str, Any]]) -> dict[str, Channel]:
    """Construct channels from config, skipping and reporting any that are broken.

    A typo in one channel's config must not stop the other four from working, and must certainly not
    stop the process from starting.
    """
    channels: dict[str, Channel] = {}
    for channel_id, entry in (config or {}).items():
        kind = entry.get("type")
        factory = CHANNEL_TYPES.get(kind or "")
        if factory is None:
            log.error("channel %r: unknown type %r; skipping", channel_id, kind)
            continue
        try:
            channels[channel_id] = factory(channel_id, entry)
        except (KeyError, ValueError) as exc:
            log.error("channel %r is misconfigured (%s); skipping", channel_id, exc)
    if not channels:
        channels["log"] = LogChannel("log", {})
    return channels


# --------------------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------------------


@dataclass
class Dispatcher:
    """Applies policy, then fans out to channels."""

    channels: dict[str, Channel]
    quiet_hours: TimeWindow | None = None
    max_per_hour: int = 12
    #: Suppress an identical notification seen within this window.
    dedupe_window: timedelta = field(default_factory=lambda: timedelta(minutes=10))

    _sent_at: dict[str, list[datetime]] = field(default_factory=dict)
    _recent_keys: dict[str, datetime] = field(default_factory=dict)
    held: list[NotificationRequest] = field(default_factory=list)

    async def dispatch(
        self,
        request: NotificationRequest,
        *,
        image: bytes | None = None,
        now: datetime | None = None,
    ) -> list[DeliveryResult]:
        now = now or datetime.now(tz=UTC)
        urgent = request.urgency.bypasses_personality  # high and above

        if not urgent and self._is_duplicate(request, now):
            return [DeliveryResult("*", True, "suppressed", "duplicate within dedupe window")]

        if not urgent and self.quiet_hours is not None and self.quiet_hours.contains(now):
            # Held, not dropped. It is already visible in the UI; this only defers the buzz.
            self.held.append(request)
            return [
                DeliveryResult("*", True, "held", f"quiet hours until {self.quiet_hours.end:%H:%M}")
            ]

        targets = [
            self.channels[cid]
            for cid in (request.channels or list(self.channels))
            if cid in self.channels and self.channels[cid].accepts(request.urgency)
        ]
        if not targets:
            return [DeliveryResult("*", False, "suppressed", "no channel accepts this urgency")]

        results = await asyncio.gather(
            *(self._send_one(channel, request, image, now, urgent) for channel in targets)
        )
        if request.dedupe_key:
            self._recent_keys[request.dedupe_key] = now
        return list(results)

    async def _send_one(
        self,
        channel: Channel,
        request: NotificationRequest,
        image: bytes | None,
        now: datetime,
        urgent: bool,
    ) -> DeliveryResult:
        if not urgent and self._rate_limited(channel.id, now):
            return DeliveryResult(
                channel.id, False, "suppressed", f"rate limit ({self.max_per_hour}/hour) reached"
            )
        try:
            await channel.send(request, image=image if channel.supports_images else None)
        except Exception as exc:
            log.warning("channel %s failed: %s", channel.id, exc)
            return DeliveryResult(channel.id, False, "failed", str(exc))
        self._sent_at.setdefault(channel.id, []).append(now)
        return DeliveryResult(channel.id, True, "sent")

    def _rate_limited(self, channel_id: str, now: datetime) -> bool:
        window = now - timedelta(hours=1)
        recent = [ts for ts in self._sent_at.get(channel_id, []) if ts > window]
        self._sent_at[channel_id] = recent
        return len(recent) >= self.max_per_hour

    def _is_duplicate(self, request: NotificationRequest, now: datetime) -> bool:
        if not request.dedupe_key:
            return False
        last = self._recent_keys.get(request.dedupe_key)
        return last is not None and now - last < self.dedupe_window

    async def release_held(self, *, now: datetime | None = None) -> list[DeliveryResult]:
        """Deliver everything held during quiet hours. Called when the window closes."""
        now = now or datetime.now(tz=UTC)
        if self.quiet_hours is not None and self.quiet_hours.contains(now):
            return []
        pending, self.held = self.held, []
        results: list[DeliveryResult] = []
        for request in pending:
            with contextlib.suppress(Exception):
                results.extend(await self.dispatch(request, now=now))
        return results


def _ntfy_priority(urgency: Urgency) -> str:
    return {"info": "2", "low": "2", "normal": "3", "high": "4", "critical": "5"}[urgency.value]


def _json(value: Any) -> str:
    import json

    return json.dumps(value)


__all__ = [
    "CHANNEL_TYPES",
    "Channel",
    "DeliveryResult",
    "DiscordChannel",
    "Dispatcher",
    "LogChannel",
    "MQTTChannel",
    "MatrixChannel",
    "NtfyChannel",
    "SMTPChannel",
    "WebhookChannel",
    "build_channels",
]
