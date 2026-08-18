"""WebSocket hub tests.

The hub is deliberately independent from network setup. Testing it with small fake sockets catches
filtering regressions without needing a browser or a running Redis instance.
"""

from __future__ import annotations

import pytest
from openhup_schemas import Envelope, EventType

from openhup.api.state import WebSocketHub


class FakeSocket:
    def __init__(self, *, fail: bool = False) -> None:
        self.messages: list[dict] = []
        self.fail = fail

    async def send_json(self, payload: dict) -> None:
        if self.fail:
            raise RuntimeError("socket closed")
        self.messages.append(payload)


@pytest.mark.asyncio
async def test_hub_delivers_only_to_matching_event_families() -> None:
    hub = WebSocketHub()
    tasks = FakeSocket()
    alerts = FakeSocket()
    everything = FakeSocket()
    await hub.connect(tasks, {"task"})
    await hub.connect(alerts, {"alert"})
    await hub.connect(everything, {"all"})

    delivered = await hub.broadcast(Envelope(type=EventType.TASK_CREATED, payload={"id": "t"}))

    assert delivered == 2
    assert len(tasks.messages) == 1
    assert alerts.messages == []
    assert len(everything.messages) == 1


@pytest.mark.asyncio
async def test_hub_removes_dead_connections_after_send_failure() -> None:
    hub = WebSocketHub()
    dead = FakeSocket(fail=True)
    live = FakeSocket()
    await hub.connect(dead, {"task"})
    await hub.connect(live, {"task"})

    assert await hub.broadcast(Envelope(type=EventType.TASK_UPDATED)) == 1
    assert dead not in hub.connections
    assert live in hub.connections


@pytest.mark.asyncio
async def test_hub_disconnect_is_idempotent() -> None:
    hub = WebSocketHub()
    socket = FakeSocket()
    await hub.connect(socket, {"system"})

    hub.disconnect(socket)
    hub.disconnect(socket)

    assert hub.connections == {}
