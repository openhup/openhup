"""Redis Streams: the event bus.

Delivery is at-least-once, which is the right trade for this system - losing an observation is
invisible (another arrives in seconds), while duplicating a task is not. Safety therefore comes from
idempotency at the effect layer (`UniqueConstraint("episode_id")` in the database), not from
pretending the bus is exactly-once.

Three things this module provides beyond thin wrapping:

* **Consumer groups with recovery.** A crashed engine's unacknowledged messages are reclaimed by the
  next one via XAUTOCLAIM, so a restart mid-batch does not lose observations.
* **Graceful absence.** With no Redis, `Bus` degrades to a local in-process queue. The API stays
  up, the UI works, and the engine processes whatever it produces itself. A single-box install with
  a broken Redis should be degraded, not dead.
* **One envelope everywhere.** The same `Envelope` goes on the bus and out over the WebSocket, so
  the frontend and the engine agree on shapes by construction.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from openhup_schemas import ConsumerGroup, Envelope, EventType, Observation, Topic

log = logging.getLogger(__name__)


@dataclass
class BusMessage:
    """A message plus the handle needed to acknowledge it."""

    id: str
    topic: str
    envelope: Envelope | None
    raw: dict[str, str]

    def observation(self) -> Observation | None:
        payload = self.raw.get("payload")
        if not payload:
            return None
        try:
            return Observation.model_validate_json(payload)
        except ValueError as exc:
            log.warning("undecodable observation %s: %s", self.id, exc)
            return None


@dataclass
class Bus:
    """Publisher and consumer over Redis Streams, with a local fallback."""

    url: str = "redis://127.0.0.1:6379/0"
    observation_maxlen: int = 100_000
    block_ms: int = 2_000
    claim_after: timedelta = field(default_factory=lambda: timedelta(minutes=2))
    consumer_name: str = "openhup-1"

    _redis: Any = None
    _local: dict[str, asyncio.Queue[BusMessage]] = field(default_factory=dict)
    _subscribers: list[Callable[[Envelope], Any]] = field(default_factory=list)
    published: int = 0
    consumed: int = 0

    @property
    def connected(self) -> bool:
        return self._redis is not None

    async def connect(self) -> bool:
        """Connect, or fall back to local queues. Never raises."""
        try:
            from redis.asyncio import Redis

            client = Redis.from_url(self.url, decode_responses=True)
            await client.ping()
        except Exception as exc:
            log.warning("bus unavailable (%s); using in-process queues", exc)
            self._redis = None
            return False
        self._redis = client
        log.info("bus connected: %s", self.url)
        return True

    async def close(self) -> None:
        if self._redis is not None:
            with contextlib.suppress(Exception):
                await self._redis.aclose()
        self._redis = None

    # -- publishing --------------------------------------------------------------------

    async def publish(self, topic: Topic, envelope: Envelope) -> str | None:
        """Publish an envelope. Local subscribers are always notified, bus or no bus."""
        self.published += 1
        for callback in self._subscribers:
            with contextlib.suppress(Exception):
                result = callback(envelope)
                if asyncio.iscoroutine(result):
                    await result

        if self._redis is None:
            queue = self._local.setdefault(topic.value, asyncio.Queue(maxsize=10_000))
            message = BusMessage(id=envelope.id, topic=topic.value, envelope=envelope, raw={})
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(message)
            return envelope.id

        maxlen = self.observation_maxlen if topic is Topic.OBSERVATIONS else 10_000
        return await self._redis.xadd(
            topic.value, envelope.redis_fields(), maxlen=maxlen, approximate=True
        )

    async def publish_observation(self, observation: Observation) -> str | None:
        """Observations bypass the envelope: they are high-volume and have their own schema."""
        self.published += 1
        if self._redis is None:
            queue = self._local.setdefault(Topic.OBSERVATIONS.value, asyncio.Queue(maxsize=10_000))
            message = BusMessage(
                id=observation.id,
                topic=Topic.OBSERVATIONS.value,
                envelope=None,
                raw={"payload": observation.model_dump_json(by_alias=True)},
            )
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(message)
            return observation.id
        return await self._redis.xadd(
            Topic.OBSERVATIONS.value,
            {"payload": observation.model_dump_json(by_alias=True)},
            maxlen=self.observation_maxlen,
            approximate=True,
        )

    def subscribe_local(self, callback: Callable[[Envelope], Any]) -> None:
        """Register an in-process subscriber - used by the WebSocket hub inside the API process."""
        self._subscribers.append(callback)

    # -- consuming ---------------------------------------------------------------------

    async def ensure_group(self, topic: Topic, group: ConsumerGroup) -> None:
        if self._redis is None:
            return
        with contextlib.suppress(Exception):  # BUSYGROUP when it already exists
            await self._redis.xgroup_create(topic.value, group.value, id="0", mkstream=True)

    async def consume(
        self,
        topic: Topic,
        group: ConsumerGroup,
        *,
        count: int = 50,
    ) -> AsyncIterator[list[BusMessage]]:
        """Yield batches from a consumer group, forever.

        Each iteration first reclaims messages abandoned by a dead consumer, then reads new ones.
        Doing the reclaim first means a crashed engine's backlog is drained before fresh data,
        preserving temporal order as far as possible - which matters because the engine's operators
        are all about ordering.
        """
        await self.ensure_group(topic, group)

        while True:
            if self._redis is None:
                queue = self._local.setdefault(topic.value, asyncio.Queue(maxsize=10_000))
                message = await queue.get()
                batch = [message]
                while not queue.empty() and len(batch) < count:
                    batch.append(queue.get_nowait())
                self.consumed += len(batch)
                yield batch
                continue

            reclaimed = await self._reclaim(topic, group, count)
            if reclaimed:
                self.consumed += len(reclaimed)
                yield reclaimed
                continue

            try:
                entries = await self._redis.xreadgroup(
                    group.value,
                    self.consumer_name,
                    {topic.value: ">"},
                    count=count,
                    block=self.block_ms,
                )
            except Exception as exc:
                log.warning("consume error on %s: %s", topic.value, exc)
                await asyncio.sleep(1)
                continue

            batch = [
                BusMessage(id=message_id, topic=topic.value, envelope=None, raw=fields)
                for _stream, messages in entries or []
                for message_id, fields in messages
            ]
            if batch:
                self.consumed += len(batch)
                yield batch

    async def _reclaim(self, topic: Topic, group: ConsumerGroup, count: int) -> list[BusMessage]:
        """Take over messages a dead consumer never acknowledged."""
        try:
            _cursor, messages, _deleted = await self._redis.xautoclaim(
                topic.value,
                group.value,
                self.consumer_name,
                min_idle_time=int(self.claim_after.total_seconds() * 1000),
                count=count,
            )
        except Exception:
            return []
        if messages:
            log.info("reclaimed %d abandoned message(s) from %s", len(messages), topic.value)
        return [
            BusMessage(id=message_id, topic=topic.value, envelope=None, raw=fields)
            for message_id, fields in messages
        ]

    async def ack(self, topic: Topic, group: ConsumerGroup, *ids: str) -> None:
        if self._redis is None or not ids:
            return
        with contextlib.suppress(Exception):
            await self._redis.xack(topic.value, group.value, *ids)

    # -- leader election ---------------------------------------------------------------

    async def acquire_leadership(self, key: str, ttl: timedelta) -> bool:
        """Try to become the engine leader.

        Only one engine may run: two would each create a task for every mess. Without Redis there is
        nothing to coordinate through, so we assume single-process operation - correct for the
        single-box default, and documented as a constraint for split deployments.
        """
        if self._redis is None:
            return True
        return bool(
            await self._redis.set(key, self.consumer_name, nx=True, ex=int(ttl.total_seconds()))
        )

    async def renew_leadership(self, key: str, ttl: timedelta) -> bool:
        """Extend the lock, but only if we still hold it.

        The ownership check matters: a paused process whose lock expired must not steal it back from
        the engine that legitimately took over, or both would run at once.
        """
        if self._redis is None:
            return True
        current = await self._redis.get(key)
        if current != self.consumer_name:
            return False
        await self._redis.expire(key, int(ttl.total_seconds()))
        return True

    async def release_leadership(self, key: str) -> None:
        if self._redis is None:
            return
        with contextlib.suppress(Exception):
            if await self._redis.get(key) == self.consumer_name:
                await self._redis.delete(key)

    # -- helpers -----------------------------------------------------------------------

    async def emit(
        self,
        topic: Topic,
        event: EventType,
        payload: dict[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        """Convenience: build an envelope and publish it."""
        await self.publish(topic, Envelope(type=event, payload=payload or {}, **fields))

    def stats(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "published": self.published,
            "consumed": self.consumed,
            "local_queues": {k: v.qsize() for k, v in self._local.items()},
        }


__all__ = ["Bus", "BusMessage"]
