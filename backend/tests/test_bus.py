"""Event bus tests without Redis.

The local queue is a supported deployment mode, not merely a test stub: a one-box install should
remain useful when Redis is unavailable. These tests pin its delivery and degradation semantics.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from openhup_schemas import (
    ConsumerGroup,
    DetectorInfo,
    Envelope,
    EventType,
    Observation,
    ObservationSource,
    Signal,
    SignalKind,
    Topic,
)

from openhup.bus import Bus, BusMessage


def observation() -> Observation:
    return Observation(
        source=ObservationSource(camera_id="kitchen", anchor_id="kitchen.counter"),
        detector=DetectorInfo(name="sensor", version="test@1"),
        signals=[Signal(key="fill", kind=SignalKind.SCALAR, value=0.75)],
    )


@pytest.mark.asyncio
async def test_local_bus_publish_notifies_sync_and_async_subscribers() -> None:
    bus = Bus()
    received: list[Envelope] = []
    async_received: list[Envelope] = []

    async def async_callback(envelope: Envelope) -> None:
        async_received.append(envelope)

    bus.subscribe_local(received.append)
    bus.subscribe_local(async_callback)
    envelope = Envelope(type=EventType.TASK_CREATED, payload={"title": "Clear counter"})

    message_id = await bus.publish(Topic.TASK_EVENTS, envelope)

    assert message_id == envelope.id
    assert received == [envelope]
    assert async_received == [envelope]
    assert bus.stats()["published"] == 1


@pytest.mark.asyncio
async def test_local_bus_continues_when_a_subscriber_fails() -> None:
    bus = Bus()
    received: list[Envelope] = []

    def broken(_envelope: Envelope) -> None:
        raise RuntimeError("subscriber failed")

    bus.subscribe_local(broken)
    bus.subscribe_local(received.append)

    await bus.publish(Topic.TASK_EVENTS, Envelope(type=EventType.TASK_CREATED))

    assert len(received) == 1


@pytest.mark.asyncio
async def test_local_bus_consumes_envelopes_in_batches() -> None:
    bus = Bus()
    for index in range(3):
        await bus.publish(
            Topic.TASK_EVENTS,
            Envelope(type=EventType.TASK_CREATED, payload={"index": index}),
        )

    stream = bus.consume(Topic.TASK_EVENTS, ConsumerGroup.WS_FANOUT, count=2)
    first = await anext(stream)
    second = await anext(stream)

    assert [message.envelope.payload["index"] for message in first] == [0, 1]
    assert [message.envelope.payload["index"] for message in second] == [2]
    assert bus.stats()["consumed"] == 3
    await stream.aclose()


@pytest.mark.asyncio
async def test_local_bus_round_trips_observations() -> None:
    bus = Bus()
    original = observation()

    message_id = await bus.publish_observation(original)
    stream = bus.consume(Topic.OBSERVATIONS, ConsumerGroup.SKILL_ENGINE)
    message = (await anext(stream))[0]

    assert message_id == original.id
    assert message.observation() == original
    assert message.topic == Topic.OBSERVATIONS.value
    await stream.aclose()


@pytest.mark.asyncio
async def test_local_bus_emit_builds_the_expected_topic_and_event() -> None:
    bus = Bus()
    await bus.emit(
        Topic.ALERT_EVENTS,
        EventType.ALERT_RAISED,
        {"plain_text": "Burner on"},
        anchor_id="kitchen.stove",
    )

    stream = bus.consume(Topic.ALERT_EVENTS, ConsumerGroup.WS_FANOUT)
    message = (await anext(stream))[0]

    assert message.envelope is not None
    assert message.envelope.type is EventType.ALERT_RAISED
    assert message.envelope.anchor_id == "kitchen.stove"
    assert message.envelope.payload == {"plain_text": "Burner on"}
    await stream.aclose()


@pytest.mark.asyncio
async def test_local_bus_assumes_single_process_leadership() -> None:
    bus = Bus()

    assert await bus.acquire_leadership("engine", timedelta(seconds=5))
    assert await bus.renew_leadership("engine", timedelta(seconds=5))
    await bus.release_leadership("engine")


@pytest.mark.asyncio
async def test_connect_falls_back_when_redis_is_unavailable() -> None:
    bus = Bus(url="redis://127.0.0.1:6399/0")

    assert await bus.connect() is False
    assert bus.connected is False
    await bus.close()


@pytest.mark.asyncio
async def test_close_is_safe_without_a_redis_connection() -> None:
    bus = Bus()

    await bus.close()
    await bus.close()
    assert bus.connected is False


def test_bus_message_returns_none_for_missing_or_invalid_payload() -> None:
    missing = BusMessage(id="1", topic="observations", envelope=None, raw={})
    invalid = BusMessage(id="2", topic="observations", envelope=None, raw={"payload": "not-json"})

    assert missing.observation() is None
    assert invalid.observation() is None


@pytest.mark.asyncio
async def test_local_queue_does_not_block_on_overflow() -> None:
    bus = Bus()
    for _ in range(10_100):
        await bus.publish(Topic.TASK_EVENTS, Envelope(type=EventType.TASK_CREATED))

    assert bus.stats()["local_queues"][Topic.TASK_EVENTS.value] == 10_000
